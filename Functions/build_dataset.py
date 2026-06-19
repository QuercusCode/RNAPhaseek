"""
build_dataset.py
================
Combines all positive and negative sequences, deduplicates, and splits
into train/val/test sets.

Steps:
  1. Merge positives: rps2_reviewed_rna.fasta + synthetic_rna.fasta
                       + RNAPhaSep/rnaphasep.fasta
  2. Normalise all sequences (uppercase, T→U, AUGCN only)
  3. Filter by length (50–10000 nt)
  4. Deduplicate positives at 100% identity (exact match on normalised seq)
  5. Cluster positives at 80% identity via CD-HIT-EST (if available)
     else simple length+hash dedup
  6. Load negatives: negatives_ensembl.fasta
  7. Remove any negative that overlaps with a positive gene
  8. Write combined FASTA: Data/splits/positives.fasta, negatives.fasta
  9. Split 80/10/10 stratified by organism
  10. Write: Data/splits/{train,val,test}_{pos,neg}.fasta
  11. Print summary statistics

Usage:
    python Functions/build_dataset.py [--no-cdhit]
"""

import argparse
import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ─────────────────────── config ────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_RAW      = PROJECT_ROOT / "Data" / "raw"
SPLITS_DIR    = PROJECT_ROOT / "Data" / "splits"

POS_SOURCES = [
    DATA_RAW / "all_positives_dedup.fasta",
]
NEG_SOURCE    = DATA_RAW / "negatives_ensembl.fasta"

_HEADER_NOISE = {
    "RPS2", "PARKERSG", "SMOOPS", "RNAPHASEP", "POS", "NEG", "HOMO", "SAPIENS",
    "HUMAN", "MOUSE", "UNDEFINED", "NATURAL", "DESIGNED", "LNCRNA", "MRNA",
    "MIRNA", "RRNA", "SNORNA", "SNRNA", "PIRNA", "SIRNA", "VIRYSRNA", "TOTALRNA",
}

CDHIT_EST     = shutil.which("cd-hit-est")
IDENTITY      = 0.80   # CD-HIT clustering identity
SEQ_MIN       = 50
SEQ_MAX       = 10_000
SEED          = 42

# ─────────────────────── helpers ────────────────────────────────

def normalise(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    seq = re.sub(r"[^AUGCN]", "", seq)
    return seq

def seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()

def parse_fasta(path: Path):
    """Yield (header, sequence) pairs."""
    if not path.exists():
        return
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        hdr, chunks = None, []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(chunks)
                hdr, chunks = line[1:], []
            else:
                chunks.append(line)
        if hdr is not None:
            yield hdr, "".join(chunks)

def organism_from_header(header: str) -> str:
    """Extract organism token from header for stratification."""
    low = header.lower()
    if "homo_sapiens" in low or "homo sapiens" in low:
        return "human"
    if "mus_musculus" in low or "mus musculus" in low:
        return "mouse"
    if "saccharomyces" in low or "cerevisiae" in low or "yeast" in low:
        return "yeast"
    if "caenorhabditis" in low or "elegans" in low or "celegans" in low:
        return "celegans"
    if "sars" in low or "coronavirus" in low or "covid" in low:
        return "sars2"
    if "drosophila" in low or "melanogaster" in low:
        return "drosophila"
    if "synthetic" in low or "poly" in low or "syn_" in low:
        return "synthetic"
    return "other"

def run_cdhit(in_fasta: Path, out_fasta: Path, identity: float = 0.80) -> Path:
    """Run cd-hit-est and return path to clustered FASTA."""
    if not CDHIT_EST:
        print("  [warn] cd-hit-est not found — skipping clustering")
        return in_fasta
    word_size = 8 if identity >= 0.9 else 6 if identity >= 0.8 else 4
    cmd = [
        CDHIT_EST,
        "-i", str(in_fasta),
        "-o", str(out_fasta),
        "-c", str(identity),
        "-n", str(word_size),
        "-T", "4",
        "-M", "4000",
        "-d", "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [warn] cd-hit-est failed: {result.stderr[:200]}")
        return in_fasta
    return out_fasta

def stratified_split(records: list, splits=(0.80, 0.10, 0.10), seed: int = SEED):
    """Split list of (header, seq) into train/val/test maintaining organism balance."""
    rng = random.Random(seed)
    # Group by organism
    by_org = defaultdict(list)
    for rec in records:
        org = organism_from_header(rec[0])
        by_org[org].append(rec)

    train, val, test = [], [], []
    for org, recs in by_org.items():
        rng.shuffle(recs)
        n = len(recs)
        n_val  = max(1, int(n * splits[1]))
        n_test = max(1, int(n * splits[2]))
        test.extend(recs[:n_test])
        val.extend(recs[n_test:n_test + n_val])
        train.extend(recs[n_test + n_val:])

    return train, val, test

def write_fasta(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for hdr, seq in records:
            f.write(f">{hdr}\n{seq}\n")
    return len(records)

# ─────────────────────── main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cdhit", action="store_true",
                        help="Skip CD-HIT clustering")
    args = parser.parse_args()

    random.seed(SEED)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load positives from all sources
    print("=== Loading positives ===")
    seen_hashes = set()
    positives   = []

    for src in POS_SOURCES:
        n_added = 0
        if not src.exists():
            print(f"  [skip] {src.name} not found")
            continue
        for hdr, seq in parse_fasta(src):
            seq = normalise(seq)
            if not SEQ_MIN <= len(seq) <= SEQ_MAX:
                continue
            h = seq_hash(seq)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            positives.append((hdr, seq))
            n_added += 1
        print(f"  {src.name}: +{n_added} (total: {len(positives)})")

    print(f"Total unique positives (pre-cluster): {len(positives)}")

    # 2. Write merged positives FASTA
    merged_pos = DATA_RAW / "merged_positives_raw.fasta"
    write_fasta(positives, merged_pos)

    # 3. CD-HIT clustering
    clustered_pos = DATA_RAW / "merged_positives_cdhit.fasta"
    if not args.no_cdhit and CDHIT_EST:
        print(f"\n=== Clustering positives at {IDENTITY*100:.0f}% identity ===")
        clustered_pos = run_cdhit(merged_pos, clustered_pos, IDENTITY)
        # Reload clustered
        positives = [(h, normalise(s)) for h, s in parse_fasta(clustered_pos)
                     if SEQ_MIN <= len(normalise(s)) <= SEQ_MAX]
        print(f"  After clustering: {len(positives)} sequences")
    else:
        if not args.no_cdhit:
            print("  [info] cd-hit-est not in PATH — using exact-dedup only")
        clustered_pos = merged_pos

    # 4. Load negatives
    print(f"\n=== Loading negatives ===")
    # Collect positive gene tokens (symbols + ENSG/ENST/etc.) to exclude from negatives.
    pos_gene_syms = set()
    for hdr, _ in positives:
        for tok in hdr.split("|"):
            t = tok.strip().upper()
            if not t or t in _HEADER_NOISE:
                continue
            if t.startswith(("ENSG", "ENST", "ENSMUSG", "ENSMUST", "NM_", "NR_", "XM_", "XR_")):
                pos_gene_syms.add(t.split(".")[0])
            elif re.fullmatch(r"[A-Z][A-Z0-9-]{1,15}", t):
                pos_gene_syms.add(t)

    negatives = []
    seen_neg_hashes = set()
    n_excluded = 0

    for hdr, seq in parse_fasta(NEG_SOURCE):
        seq = normalise(seq)
        if not SEQ_MIN <= len(seq) <= SEQ_MAX:
            continue
        # prepare_negatives writes headers as: neg|<enst>|<gene>|len=..|gc=..
        parts = hdr.split("|")
        excluded = False
        for p in parts:
            t = p.strip().upper().split(".")[0]
            if t in pos_gene_syms:
                excluded = True
                break
        if excluded:
            n_excluded += 1
            continue
        h = seq_hash(seq)
        if h in seen_neg_hashes:
            continue
        seen_neg_hashes.add(h)
        negatives.append((hdr, seq))

    print(f"  Negatives loaded: {len(negatives)} (excluded {n_excluded} gene overlaps)")

    # 5. Balance if too many negatives (max 3:1 ratio)
    max_neg = min(len(negatives), len(positives) * 3)
    if len(negatives) > max_neg:
        random.shuffle(negatives)
        negatives = negatives[:max_neg]
        print(f"  Subsampled to {len(negatives)} negatives (3:1 ratio)")

    # 6. Split
    print(f"\n=== Splitting dataset ===")
    pos_train, pos_val, pos_test = stratified_split(positives)
    neg_train, neg_val, neg_test = stratified_split(negatives)

    print(f"  Positives  — train: {len(pos_train)} | val: {len(pos_val)} | test: {len(pos_test)}")
    print(f"  Negatives  — train: {len(neg_train)} | val: {len(neg_val)} | test: {len(neg_test)}")

    # 7. Write splits
    split_counts = {}
    for split_name, pos_recs, neg_recs in [
        ("train", pos_train, neg_train),
        ("val",   pos_val,   neg_val),
        ("test",  pos_test,  neg_test),
    ]:
        n_pos = write_fasta(pos_recs, SPLITS_DIR / f"{split_name}_pos.fasta")
        n_neg = write_fasta(neg_recs, SPLITS_DIR / f"{split_name}_neg.fasta")
        split_counts[split_name] = (n_pos, n_neg)

    # Also write full positives/negatives
    write_fasta(positives, SPLITS_DIR / "all_positives.fasta")
    write_fasta(negatives, SPLITS_DIR / "all_negatives.fasta")

    # 8. Summary
    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")
    total_pos = len(positives)
    total_neg = len(negatives)
    print(f"Total positives : {total_pos}")
    print(f"Total negatives : {total_neg}")
    print(f"Ratio neg/pos   : {total_neg/max(total_pos,1):.2f}")
    print()
    for split, (np, nn) in split_counts.items():
        print(f"  {split:5s}: {np} pos + {nn} neg = {np+nn} total")

    # Organism distribution of positives
    print("\nOrganism distribution (positives):")
    org_counts = defaultdict(int)
    for hdr, _ in positives:
        org_counts[organism_from_header(hdr)] += 1
    for org, n in sorted(org_counts.items(), key=lambda x: -x[1]):
        print(f"  {org:15s}: {n}")

    # Length stats
    lengths = [len(s) for _, s in positives]
    if lengths:
        lengths.sort()
        print(f"\nPositive lengths: min={min(lengths)} "
              f"median={lengths[len(lengths)//2]} max={max(lengths)}")

    print(f"\n✓ Splits written to {SPLITS_DIR}")

if __name__ == "__main__":
    main()
