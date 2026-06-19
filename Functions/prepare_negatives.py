"""
prepare_negatives.py
====================
Build a GC+length-matched negative set from Ensembl human cDNA.

Strategy (fast, 2-pass):
  Pass 1 — Stream Ensembl FASTA (via system gzip -cd); reservoir-sample
            up to SAMPLE_CAP protein_coding sequences; store only
            (header, seq, gc, length, gene_sym) — no full-sequence dict.
  Pass 2 — For each positive, search the sampled pool for a match
            (same gc ±6%, same length ±25%), ensure no gene overlap.
  Output — negatives_ensembl.fasta (one matched negative per positive).

Why not index all 221k sequences:
  Building a RAM-resident dict of 221k × 3knt strings in Python is slow
  (~4 min, 730 MB). Reservoir-sampling 30k is instant and gives ample
  choice for matching ~1200 positives.

Usage:
    python Functions/prepare_negatives.py [--sample-cap 30000]
"""

import argparse
import io
import re
import random
import subprocess
import sys
from pathlib import Path

# ─────────────────────── config ──────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_RAW      = PROJECT_ROOT / "Data" / "raw"
ENSEMBL_FASTA = DATA_RAW / "ensembl_human_cdna.fa.gz"
POS_FASTAS    = [DATA_RAW / "all_positives_dedup.fasta"]
OUT_NEG_FASTA = DATA_RAW / "negatives_ensembl.fasta"

# Tokens we should not treat as gene symbols even if they appear in headers.
_HEADER_NOISE = {
    "RPS2", "PARKERSG", "SMOOPS", "RNAPHASEP", "POS", "NEG", "HOMO", "SAPIENS",
    "HUMAN", "MOUSE", "UNDEFINED", "NATURAL", "DESIGNED", "LNCRNA", "MRNA",
    "MIRNA", "RRNA", "SNORNA", "SNRNA", "PIRNA", "SIRNA", "VIRYSRNA", "TOTALRNA",
}

SEQ_MIN, SEQ_MAX = 50, 10_000
GC_TOL           = 0.06    # ±6%
LEN_TOL          = 0.25    # ±25%
SAMPLE_CAP       = 30_000  # reservoir-sample this many Ensembl seqs

ALLOWED_BIOTYPES = {"protein_coding"}
SEED = 42

# ─────────────────────── helpers ─────────────────────────────────

def normalise(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    seq = re.sub(r"[^AUGCN]", "", seq)
    return seq

def gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in "GC") / len(seq)

def parse_fasta_stream(path: str):
    """Stream (header, seq) using system gzip -cd for .gz files."""
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

def parse_fasta_file(path: Path):
    if not path.exists():
        return
    for hdr, seq in parse_fasta_stream(str(path)):
        yield hdr, seq

# ─────────────────────── main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-cap", type=int, default=SAMPLE_CAP)
    parser.add_argument("--seed",       type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1. Load positives: gene symbols + stats
    print("Loading positive sequences ...")
    pos_genes = set()
    pos_stats = []  # list of (gc, length)

    for path in POS_FASTAS:
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        for hdr, seq in parse_fasta_file(path):
            seq = normalise(seq)
            if not SEQ_MIN <= len(seq) <= SEQ_MAX:
                continue
            # Collect every "|"-separated token that could be a gene id/symbol;
            # the merged FASTA has heterogeneous headers across source DBs.
            for tok in hdr.split("|"):
                t = tok.strip().upper()
                if not t or t in _HEADER_NOISE:
                    continue
                if t.startswith(("ENSG", "ENST", "ENSMUSG", "ENSMUST", "NM_", "NR_", "XM_", "XR_")):
                    pos_genes.add(t.split(".")[0])
                elif re.fullmatch(r"[A-Z][A-Z0-9-]{1,15}", t):
                    pos_genes.add(t)
            pos_stats.append((gc_content(seq), len(seq)))

    print(f"  Positive seqs: {len(pos_stats)}, unique gene tokens: {len(pos_genes)}")

    # 2. Reservoir-sample Ensembl protein_coding sequences
    print(f"Reservoir-sampling Ensembl cDNA (cap={args.sample_cap:,})…")
    pool = []   # (hdr, seq, gc, length, gene)
    n_seen = 0  # protein_coding sequences seen so far

    for hdr, seq in parse_fasta_stream(str(ENSEMBL_FASTA)):
        # Biotype filter (fast check on header string)
        bt_m = re.search(r"transcript_biotype:(\S+)", hdr)
        if not bt_m or bt_m.group(1) not in ALLOWED_BIOTYPES:
            continue

        # Sequence filters
        seq_norm = normalise(seq)
        L = len(seq_norm)
        if not SEQ_MIN <= L <= SEQ_MAX:
            n_seen += 1
            continue

        # Collect candidate identifiers from the Ensembl header.
        # Ensembl cDNA headers look like:
        #   >ENST... cdna chromosome:GRCh38:... gene:ENSG00000... transcript_biotype:... gene_symbol:NEAT1
        gm_sym = re.search(r"gene_symbol:(\S+)", hdr)
        gm_ens = re.search(r"gene:(\S+)", hdr)
        enst_m = re.match(r"(\S+)", hdr)
        gene_tokens = set()
        if gm_sym:
            gene_tokens.add(gm_sym.group(1).upper())
        if gm_ens:
            gene_tokens.add(gm_ens.group(1).upper().split(".")[0])
        if enst_m:
            gene_tokens.add(enst_m.group(1).upper().split(".")[0])
        gene = next(iter(gene_tokens & pos_genes), "") or (gm_sym.group(1).upper() if gm_sym else "")

        # Skip known positive genes / transcripts
        if gene_tokens & pos_genes:
            n_seen += 1
            continue

        gc = gc_content(seq_norm)
        n_seen += 1

        # Reservoir sampling (Knuth's algorithm R)
        if len(pool) < args.sample_cap:
            pool.append((hdr.split()[0], seq_norm, gc, L, gene))
        else:
            j = rng.randint(0, n_seen - 1)
            if j < args.sample_cap:
                pool[j] = (hdr.split()[0], seq_norm, gc, L, gene)

    print(f"  Protein_coding seen: {n_seen:,} → sampled {len(pool):,}")

    # 3. Match negatives to positives
    print("Matching negatives…")
    used_genes = set()
    used_enst  = set()
    negatives  = []   # (enst, seq, gene, gc, length)

    for gc_p, len_p in pos_stats:
        gc_lo, gc_hi   = gc_p - GC_TOL, gc_p + GC_TOL
        len_lo, len_hi = len_p * (1 - LEN_TOL), len_p * (1 + LEN_TOL)

        candidates = [
            e for e in pool
            if gc_lo <= e[2] <= gc_hi
            and len_lo <= e[3] <= len_hi
            and e[0] not in used_enst
            and (not e[4] or e[4] not in used_genes)
        ]

        if not candidates:
            continue  # no match found for this positive

        chosen = rng.choice(candidates)
        used_enst.add(chosen[0])
        if chosen[4]:
            used_genes.add(chosen[4])
        negatives.append(chosen)

    print(f"  Matched: {len(negatives)}/{len(pos_stats)} positives "
          f"({100*len(negatives)/max(1,len(pos_stats)):.1f}%)")

    # 4. Write output FASTA
    with open(OUT_NEG_FASTA, "w", encoding="utf-8") as f:
        for enst, seq, gene, gc, length in negatives:
            f.write(f">neg|{enst}|{gene}|len={length}|gc={gc:.3f}\n{seq}\n")

    print(f"✓ Written {len(negatives)} negatives → {OUT_NEG_FASTA}")

    # 5. Stats
    if negatives:
        lens = sorted(e[3] for e in negatives)
        gcs  = sorted(e[2] for e in negatives)
        print(f"\nNeg length: min={min(lens)} median={lens[len(lens)//2]} max={max(lens)}")
        print(f"Neg GC:     min={min(gcs):.3f} median={gcs[len(gcs)//2]:.3f} max={max(gcs):.3f}")

if __name__ == "__main__":
    main()
