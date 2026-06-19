"""
Build species-matched GC + length-matched negative pools for each species'
LLPS positives.

Each species' positives are paired with negatives drawn from THAT species'
reference transcriptome, with:
  - GC within ±6% of each positive
  - length within ±25% of each positive
  - gene-symbol / gene-ID overlap with positives excluded

Output (per species):
  Data/raw/multispecies/negatives/{species}_negatives.fasta

This script assumes:
  - Reference transcriptomes already downloaded by
    download_reference_transcriptomes.py
  - Per-species positives FASTAs already built by the corresponding
    collect_<species>_llps.py scripts

Run:
    python -m Functions.data_collection_multispecies.prepare_multispecies_negatives
"""

import argparse
import gzip
import io
import os
import random
import re
import subprocess
import sys
from pathlib import Path

POSITIVES_DIR  = Path("Data/raw/multispecies")
REFS_DIR       = Path("Data/raw/multispecies/refs")
NEGATIVES_DIR  = Path("Data/raw/multispecies/negatives")

# Per-species config: (positives_fasta, reference_fasta_gz, species_label)
SPECIES_CONFIG = [
    ("smoops_mouse_positives.fasta",    "mus_musculus_cdna.fa.gz",                "mouse"),
    ("celegans_positives.fasta",        "caenorhabditis_elegans_cdna.fa.gz",      "celegans"),
    ("drosophila_positives.fasta",      "drosophila_melanogaster_cdna.fa.gz",     "drosophila"),
    ("yeast_positives.fasta",           "saccharomyces_cerevisiae_cdna.fa.gz",    "yeast"),
    ("spombe_positives.fasta",          "schizosaccharomyces_pombe_cdna.fa.gz",   "spombe"),
    ("arabidopsis_positives.fasta",     "arabidopsis_thaliana_cdna.fa.gz",        "arabidopsis"),
    ("rice_positives.fasta",            "oryza_sativa_cdna.fa.gz",                "rice"),
    # Viral RNAs: skip species-matched negatives (small set; could pair with
    # dinucleotide-shuffled negatives if class balance matters later).
]

SEQ_MIN, SEQ_MAX = 50, 10_000
GC_TOL           = 0.06
LEN_TOL          = 0.25
SAMPLE_CAP       = 30_000
SEED             = 42


def normalise(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    return re.sub(r"[^AUGCN]", "", seq)


def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in "GC") / len(seq)


def parse_fasta_stream(path: str):
    """Yield (header, seq) from a (possibly gzipped) FASTA via system gzip -cd."""
    if path.endswith(".gz"):
        proc = subprocess.Popen(
            ["gzip", "-cd", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=1 << 22,
        )
        lines = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
    else:
        proc = None
        lines = open(path, "rt", encoding="utf-8")

    hdr, chunks = None, []
    for line in lines:
        line = line.rstrip()
        if line.startswith(">"):
            if hdr is not None:
                yield hdr, "".join(chunks)
            hdr, chunks = line[1:], []
        else:
            chunks.append(line)
    if hdr is not None:
        yield hdr, "".join(chunks)

    if proc is not None:
        proc.stdout.close()
        proc.wait()


# ── Per-species header parsers (to extract gene symbol / gene ID / biotype) ──

ENSEMBL_GENE_RE   = re.compile(r"gene:(\S+)")
ENSEMBL_SYMBOL_RE = re.compile(r"gene_symbol:(\S+)")
ENSEMBL_BIOTYPE_RE= re.compile(r"transcript_biotype:(\S+)")


def parse_header_for_keys(hdr: str, species: str) -> tuple[str, str, str]:
    """
    Return (transcript_id, gene_symbol_or_id, biotype) from a reference cDNA header.

    Ensembl format (mouse, worm, fly):
      ENST... cdna chromosome:... gene:ENSG... transcript_biotype:protein_coding gene_symbol:NEAT1

    SGD yeast format:
      YAL068C PAU8 SGDID:S000002142, Chr I from 1...
    """
    if species == "yeast":
        toks = hdr.split(",", 1)[0].split()
        sys_id   = toks[0] if toks else ""
        std_name = toks[1] if len(toks) > 1 and not toks[1].startswith("SGDID:") and toks[1] != sys_id else sys_id
        return sys_id, std_name, "protein_coding"   # SGD ORFs are all protein-coding
    # Ensembl
    tx_id = hdr.split()[0] if hdr else ""
    gm = ENSEMBL_GENE_RE.search(hdr);    gene_id = gm.group(1).split(".")[0] if gm else ""
    sm = ENSEMBL_SYMBOL_RE.search(hdr);  gene_sym = sm.group(1) if sm else gene_id
    bm = ENSEMBL_BIOTYPE_RE.search(hdr); biotype = bm.group(1) if bm else "protein_coding"
    return tx_id, gene_sym, biotype


# ── Extract positive identifiers (for overlap exclusion in negatives) ────────

def positive_gene_keys(positives_path: Path) -> set[str]:
    keys: set[str] = set()
    if not positives_path.exists():
        return keys
    for hdr, _ in parse_fasta_stream(str(positives_path)):
        # Header format: llps_<species>|<gene_id>|<tx_id>|<source>
        toks = hdr.split("|")
        for t in toks[1:3]:
            t = t.upper().strip()
            if t:
                keys.add(t)
                keys.add(t.split(".")[0])     # strip Ensembl version suffix
    return keys


# ── Build negative pool and match ────────────────────────────────────────────

def build_negative_pool(reference_path: Path, species: str,
                        exclude_keys: set[str], cap: int = SAMPLE_CAP,
                        rng: random.Random = None,
                        biotype_filter: tuple[str, ...] = ("protein_coding",)) -> list[tuple[str, str, float, int, str]]:
    """
    Reservoir-sample sequences from the reference transcriptome.
    Returns list of (tx_id, seq, gc, length, gene_symbol).
    """
    rng = rng or random.Random(SEED)
    pool: list = []
    n_seen = 0
    for hdr, seq in parse_fasta_stream(str(reference_path)):
        tx_id, gene_sym, biotype = parse_header_for_keys(hdr, species)
        if biotype_filter and biotype not in biotype_filter:
            continue
        seq_n = normalise(seq)
        L = len(seq_n)
        if not SEQ_MIN <= L <= SEQ_MAX:
            n_seen += 1
            continue
        # Skip if this gene is in the positives set
        if gene_sym and gene_sym.upper() in exclude_keys:
            n_seen += 1
            continue
        if tx_id and tx_id.upper().split(".")[0] in exclude_keys:
            n_seen += 1
            continue

        gc = gc_content(seq_n)
        n_seen += 1
        item = (tx_id, seq_n, gc, L, gene_sym)
        if len(pool) < cap:
            pool.append(item)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < cap:
                pool[j] = item
    return pool, n_seen


def match_negatives_to_positives(positives_path: Path, pool: list, rng: random.Random,
                                  gc_tol: float = GC_TOL, len_tol: float = LEN_TOL,
                                  ) -> list:
    """For each positive, pick one matching negative; return list of negative tuples."""
    used_tx = set()
    used_gene = set()
    chosen = []
    for hdr, seq in parse_fasta_stream(str(positives_path)):
        seq_n = normalise(seq)
        L = len(seq_n)
        if not SEQ_MIN <= L <= SEQ_MAX:
            continue
        gc = gc_content(seq_n)
        gc_lo, gc_hi   = gc - gc_tol, gc + gc_tol
        len_lo, len_hi = L * (1 - len_tol), L * (1 + len_tol)
        candidates = [
            e for e in pool
            if gc_lo <= e[2] <= gc_hi
            and len_lo <= e[3] <= len_hi
            and e[0] not in used_tx
            and (not e[4] or e[4] not in used_gene)
        ]
        if not candidates:
            continue
        c = rng.choice(candidates)
        used_tx.add(c[0])
        if c[4]:
            used_gene.add(c[4])
        chosen.append(c)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    NEGATIVES_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for pos_file, ref_file, species in SPECIES_CONFIG:
        pos_path = POSITIVES_DIR / pos_file
        ref_path = REFS_DIR / ref_file
        out_path = NEGATIVES_DIR / f"{species}_negatives.fasta"

        if not pos_path.exists():
            print(f"  [skip {species}] {pos_path} not found")
            continue
        if not ref_path.exists():
            print(f"  [skip {species}] {ref_path} not found")
            continue

        n_pos = sum(1 for _ in parse_fasta_stream(str(pos_path)))
        if n_pos == 0:
            print(f"  [skip {species}] {pos_path} is empty")
            continue

        print(f"\n=== {species} ===")
        print(f"  positives FASTA: {pos_path} ({n_pos} records)")
        excl = positive_gene_keys(pos_path)
        print(f"  exclude keys collected: {len(excl)}")

        pool, n_seen = build_negative_pool(ref_path, species, excl, rng=rng)
        print(f"  reference sampled: {n_seen} screened -> {len(pool)} kept in pool")

        chosen = match_negatives_to_positives(pos_path, pool, rng)
        print(f"  matched negatives: {len(chosen)} / {n_pos} positives "
              f"({100*len(chosen)/n_pos:.1f}%)")

        with open(out_path, "w") as f:
            for tx_id, seq, gc, L, gene_sym in chosen:
                f.write(f">neg_{species}|{tx_id}|{gene_sym}|len={L}|gc={gc:.3f}\n{seq}\n")
        print(f"  wrote {out_path}")
        summary.append((species, n_pos, len(chosen)))

    print("\n" + "=" * 60)
    print(f"{'Species':<14} {'Positives':>10} {'Negatives':>10} {'Match%':>8}")
    print("-" * 60)
    for sp, np, nn in summary:
        pct = 100*nn/np if np else 0
        print(f"{sp:<14} {np:>10} {nn:>10} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
