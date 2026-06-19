"""
precompute_biophysical.py
==========================
Pre-compute the 26 RNA2PS + ENCORI biophysical features for every sequence
and save them as aligned .npy arrays next to the FEGS matrices.

Output files (all in Data/splits/):
  biophys_pos.npy          (482, 26) float32  — all positives
  biophys_neg.npy          (847, 26) float32  — all negatives
  biophys_norm_stats.npz   — mean/std used for z-score normalisation
  biophys_train_pos.npy    (390, 26)
  biophys_train_neg.npy    (678, 26)
  biophys_val_pos.npy       (46, 26)
  biophys_val_neg.npy       (84, 26)
  biophys_test_pos.npy      (46, 26)
  biophys_test_neg.npy      (85, 26)

The ordering of rows MUST match the corresponding encoded .npy arrays
(train_pos_encoded.npy, etc.) — both are built in FASTA record order.

Usage
-----
    python Functions/precompute_biophysical.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Functions.RNA_biophysical import RNABiophysicalExtractor, N_FEATURES, FEATURE_NAMES

SPLITS_DIR   = ROOT / "Data" / "splits"
NORM_STATS   = SPLITS_DIR / "biophys_norm_stats.npz"


def load_fasta_seqs(path: Path) -> list:
    """Return list of (header, seq) from a FASTA file using plain Python open()."""
    records = []
    hdr, chunks = None, []
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr is not None:
                    records.append((hdr, "".join(chunks)))
                hdr, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if hdr is not None:
        records.append((hdr, "".join(chunks)))
    return records


def compute_and_save(records: list, out_path: Path, extractor: RNABiophysicalExtractor,
                     desc: str = "") -> np.ndarray:
    seqs = [seq for _, seq in records]
    feats = np.stack(
        [extractor._compute_one(s) for s in tqdm(seqs, desc=desc, ncols=90)],
        axis=0,
    ).astype(np.float32)
    np.save(str(out_path), feats)
    print(f"  Saved {out_path.name}  shape={feats.shape}")
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta_pos", type=str, default="",
                   help="Single positives FASTA. With --fasta_neg, skip splits "
                        "and write biophys_{pos,neg}.npy indexed by FASTA order. "
                        "Required for hybrid-model alignment.")
    p.add_argument("--fasta_neg", type=str, default="")
    args = p.parse_args()

    print(f"RNA Biophysical Feature Precomputation")
    print(f"  N_FEATURES = {N_FEATURES}")
    print(f"  Features   : {', '.join(FEATURE_NAMES[:5])} ... (+{N_FEATURES-5} more)\n")

    extractor = RNABiophysicalExtractor(normalize=False)  # raw; normalise later

    # ── Single-FASTA mode (for hybrid model) ──────────────────────────────────
    if args.fasta_pos and args.fasta_neg:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Single-FASTA mode:\n  pos: {args.fasta_pos}\n  neg: {args.fasta_neg}")
        pos_recs = load_fasta_seqs(Path(args.fasta_pos))
        neg_recs = load_fasta_seqs(Path(args.fasta_neg))
        pos_feats = compute_and_save(pos_recs, SPLITS_DIR / "biophys_pos.npy", extractor,
                                      desc=f"pos ({len(pos_recs)})")
        neg_feats = compute_and_save(neg_recs, SPLITS_DIR / "biophys_neg.npy", extractor,
                                      desc=f"neg ({len(neg_recs)})")
        train_all = np.vstack([pos_feats, neg_feats])
        mean = train_all.mean(axis=0)
        std  = train_all.std(axis=0).clip(min=1e-8)
        np.savez(str(NORM_STATS), mean=mean, std=std)
        print(f"Normalisation stats saved -> {NORM_STATS.name}")
        print("\nDone (single-FASTA mode).")
        return

    splits  = ["train", "val", "test"]
    labels  = ["pos", "neg"]
    all_pos_raw, all_neg_raw = [], []

    per_split: dict = {}

    # ── Per-split computation ─────────────────────────────────────────────────
    for split in splits:
        for label in labels:
            fasta = SPLITS_DIR / f"{split}_{label}.fasta"
            if not fasta.exists():
                print(f"  [skip] {fasta.name} not found")
                continue
            records = load_fasta_seqs(fasta)
            if not records:
                print(f"  [skip] {fasta.name} empty")
                continue

            out_path = SPLITS_DIR / f"biophys_{split}_{label}.npy"
            feats    = compute_and_save(
                records, out_path, extractor,
                desc=f"{split}/{label} ({len(records)})"
            )
            per_split[f"{split}_{label}"] = feats

            if label == "pos":
                all_pos_raw.append(feats)
            else:
                all_neg_raw.append(feats)

    # ── Merged arrays ─────────────────────────────────────────────────────────
    pos_merged = np.vstack(all_pos_raw)
    neg_merged = np.vstack(all_neg_raw)
    np.save(str(SPLITS_DIR / "biophys_pos.npy"), pos_merged)
    np.save(str(SPLITS_DIR / "biophys_neg.npy"), neg_merged)
    print(f"\nMerged pos: {pos_merged.shape}   neg: {neg_merged.shape}")

    # ── Fit z-score normalisation on training set only ────────────────────────
    train_all = np.vstack([
        per_split.get("train_pos", np.zeros((0, N_FEATURES))),
        per_split.get("train_neg", np.zeros((0, N_FEATURES))),
    ])
    mean = train_all.mean(axis=0)
    std  = train_all.std(axis=0).clip(min=1e-8)
    np.savez(str(NORM_STATS), mean=mean, std=std)
    print(f"Normalisation stats saved → {NORM_STATS.name}")

    # ── Feature stats report ──────────────────────────────────────────────────
    all_data = np.vstack([pos_merged, neg_merged])
    print(f"\n{'Feature':<32} {'pos_mean':>10} {'neg_mean':>10} {'|Δ|':>8}")
    print("─" * 64)
    for i, name in enumerate(FEATURE_NAMES):
        pm = pos_merged[:, i].mean()
        nm = neg_merged[:, i].mean()
        print(f"  {name:<30} {pm:>10.4f} {nm:>10.4f} {abs(pm-nm):>8.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
