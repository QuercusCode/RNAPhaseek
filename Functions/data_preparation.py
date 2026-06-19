"""
Data Preparation for RNA LLPS Predictor
=========================================
This script downloads and processes the RPS 2.0 database to build
positive and negative training sets.

Steps
-----
1. Fetch RPS 2.0 "Reviewed" RNA entries  → positive set
2. Build negative set from Ensembl transcriptome (length/GC-matched)
3. CD-HIT clustering to remove redundancy at 80% identity
4. Train/val/test split

Requirements
------------
  pip install biopython requests tqdm
  sudo apt-get install cd-hit   (or: conda install -c bioconda cd-hit)

Usage
-----
  python Functions/data_preparation.py
"""

import os
import re
import json
import hashlib
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("Data/raw")
PROCESSED_DIR = Path("Data/processed")
SPLITS_DIR    = Path("Data/splits")

for d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── RPS 2.0 API ───────────────────────────────────────────────────────────────
RPS2_BASE = "https://rps.renlab.cn"

def fetch_rps2_reviewed(out_fasta: Path, max_entries: int = None) -> int:
    """
    Fetch RPS 2.0 'Reviewed' entries and write to FASTA.
    Returns number of sequences written.

    Note: RPS 2.0 (https://rps.renlab.cn) may require manual download
    from the web interface. This function attempts their REST API;
    if unavailable, instructions for manual download are printed.
    """
    url = f"{RPS2_BASE}/api/download?evidence=Reviewed&format=fasta"
    print(f"Fetching RPS 2.0 Reviewed entries from {url} ...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(out_fasta, "wb") as f:
            f.write(r.content)
        count = sum(1 for _ in SeqIO.parse(str(out_fasta), "fasta"))
        print(f"  Downloaded {count} sequences → {out_fasta}")
        return count
    except Exception as e:
        print(f"  [!] Automatic download failed: {e}")
        print(
            "\n  Manual download instructions:"
            "\n  1. Go to https://rps.renlab.cn"
            "\n  2. Filter by: Evidence = 'Reviewed'"
            "\n  3. Download as FASTA"
            f"\n  4. Save to: {out_fasta}"
            "\n  Then re-run this script.\n"
        )
        return 0


def normalise_fasta(in_fasta: Path, out_fasta: Path) -> int:
    """
    Normalise sequences: uppercase, T→U, remove non-AUGC chars,
    filter sequences < 20 nt or > 10,000 nt.
    """
    records = []
    for rec in SeqIO.parse(str(in_fasta), "fasta"):
        seq = str(rec.seq).upper().replace("T", "U")
        seq = re.sub(r"[^AUGC]", "", seq)
        if 20 <= len(seq) <= 10_000:
            records.append(SeqRecord(Seq(seq), id=rec.id, description=rec.description))
    SeqIO.write(records, str(out_fasta), "fasta")
    print(f"Normalised {len(records)} sequences → {out_fasta}")
    return len(records)


def cluster_cdhit(in_fasta: Path, out_fasta: Path, identity: float = 0.80) -> int:
    """Run cd-hit-est to cluster at given identity threshold."""
    cmd = (
        f"cd-hit-est -i {in_fasta} -o {out_fasta} "
        f"-c {identity} -n 8 -M 4000 -T 4 -d 0 -sc 1 -sf 1"
    )
    print(f"Clustering at {identity*100:.0f}% identity...")
    ret = os.system(cmd)
    if ret != 0:
        print("  [!] cd-hit-est failed. Install with: conda install -c bioconda cd-hit")
        return 0
    count = sum(1 for _ in SeqIO.parse(str(out_fasta), "fasta"))
    print(f"  After clustering: {count} sequences")
    return count


def build_gc_length_matched_negatives(
    pos_fasta: Path,
    transcriptome_fasta: Path,
    out_fasta: Path,
    ratio: int = 2,
) -> int:
    """
    Select negative sequences from a transcriptome FASTA that are
    length- and GC-content-matched to the positives.

    transcriptome_fasta: all transcripts (e.g. Ensembl cDNA).
    Sequences known to be in RPS 2.0 (positives) are excluded.
    """
    pos_ids = {r.id for r in SeqIO.parse(str(pos_fasta), "fasta")}
    pos_seqs = [str(r.seq) for r in SeqIO.parse(str(pos_fasta), "fasta")]

    def gc(seq): return (seq.count("G") + seq.count("C")) / max(len(seq), 1)
    def len_bin(l): return l // 100
    def gc_bin(g):  return round(g * 10) / 10  # round to 0.1

    # Build positive profile
    pos_profile = [(len_bin(len(s)), gc_bin(gc(s))) for s in pos_seqs]

    # Build pool of candidate negatives
    pool = {}
    print("Building transcriptome pool for negative sampling...")
    for rec in tqdm(SeqIO.parse(str(transcriptome_fasta), "fasta")):
        if rec.id in pos_ids:
            continue
        seq = str(rec.seq).upper().replace("T", "U")
        seq = re.sub(r"[^AUGC]", "", seq)
        if len(seq) < 20:
            continue
        key = (len_bin(len(seq)), gc_bin(gc(seq)))
        pool.setdefault(key, []).append((rec.id, seq))

    # Sample negatives matching each positive's profile
    selected = []
    rng = np.random.default_rng(42)
    for lb, gb in pos_profile:
        for key in [(lb, gb), (lb, gb - 0.1), (lb, gb + 0.1),
                    (lb - 1, gb), (lb + 1, gb)]:
            if key in pool and pool[key]:
                n = min(ratio, len(pool[key]))
                idxs = rng.choice(len(pool[key]), size=n, replace=False)
                for i in idxs:
                    selected.append(pool[key][i])
                break

    records = [SeqRecord(Seq(seq), id=rid, description="negative") for rid, seq in selected]
    SeqIO.write(records, str(out_fasta), "fasta")
    print(f"Wrote {len(records)} negative sequences → {out_fasta}")
    return len(records)


def train_val_test_split(
    pos_fasta: Path,
    neg_fasta: Path,
    splits_dir: Path,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
):
    """Split into train/val/test and save separate FASTA files."""
    rng = np.random.default_rng(seed)

    def split_records(fasta):
        recs = list(SeqIO.parse(str(fasta), "fasta"))
        rng.shuffle(recs)
        n = len(recs)
        ntest = int(n * test_frac)
        nval  = int(n * val_frac)
        return recs[ntest + nval:], recs[ntest:ntest + nval], recs[:ntest]

    pos_tr, pos_va, pos_te = split_records(pos_fasta)
    neg_tr, neg_va, neg_te = split_records(neg_fasta)

    for split, p, n in [("train", pos_tr, neg_tr),
                        ("val",   pos_va, neg_va),
                        ("test",  pos_te, neg_te)]:
        SeqIO.write(p, str(splits_dir / f"{split}_pos.fasta"), "fasta")
        SeqIO.write(n, str(splits_dir / f"{split}_neg.fasta"), "fasta")
        print(f"  {split:5s}: {len(p)} pos | {len(n)} neg")

    print(f"Splits saved to {splits_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1 — Download positives from RPS 2.0
    raw_pos = RAW_DIR / "rps2_reviewed_raw.fasta"
    n = fetch_rps2_reviewed(raw_pos)

    if n > 0:
        # Step 2 — Normalise
        norm_pos = PROCESSED_DIR / "pos_normalised.fasta"
        normalise_fasta(raw_pos, norm_pos)

        # Step 3 — Cluster at 80%
        clust_pos = PROCESSED_DIR / "pos_clustered.fasta"
        cluster_cdhit(norm_pos, clust_pos, identity=0.80)

        # Step 4 — Build negatives (requires a local transcriptome FASTA)
        transcriptome = RAW_DIR / "ensembl_human_cdna.fa"
        if transcriptome.exists():
            neg_raw = PROCESSED_DIR / "neg_raw.fasta"
            build_gc_length_matched_negatives(clust_pos, transcriptome, neg_raw, ratio=2)
            clust_neg = PROCESSED_DIR / "neg_clustered.fasta"
            cluster_cdhit(neg_raw, clust_neg, identity=0.80)

            # Step 5 — Split
            print("\nSplitting into train/val/test...")
            train_val_test_split(clust_pos, clust_neg, SPLITS_DIR)
        else:
            print(
                f"\n[!] Transcriptome not found at {transcriptome}."
                "\n    Download from: https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/cdna/"
                f"\n    Save to: {transcriptome}"
            )
    else:
        print("\nDownload RPS 2.0 manually (see instructions above) and re-run.")
