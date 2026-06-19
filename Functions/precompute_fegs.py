"""
RNA-FEGS Matrix Precomputation
================================
For each RNA sequence in the dataset, computes the top-k RNA-FEGS Lhat
matrices and saves them as individual .npz files.

These .npz files are the graph-bias inputs (Lhat_stack) consumed by the
RNAPhaseek DataLoader during training and inference.

What a .npz file contains
--------------------------
  M0.npy … M{k-1}.npy   —  float32 (L, L) normalised walk-distance matrices
                             one per RNA-FEGS motif group (default k=10)

Where L = min(sequence_length, SEQ_LEN) — sequences are truncated before
matrix computation to keep memory bounded.

How Lhat is computed (per motif m, sequence s)
-----------------------------------------------
  1. GRS walk  : trace a 3-D walk through s guided by the nucleotide
                 coordinates of motif m → W ∈ R^{(L+1)×3}
  2. Euclidean distances:  E = squareform(pdist(W[1:]))    (L×L)
  3. Walk-path distances:  sdist[i,j] = sum of step lengths from i to j
  4. Lhat = E / (sdist + I)   (normalised adjacency; I prevents /0)

This is the same computation as Phaseek's _ME_static, but instead of
extracting only the leading eigenvalue we keep the full matrix for use as
a per-head attention bias in the Transformer.

Usage
-----
  # Process all splits (pos + neg)
  python Functions/precompute_fegs.py

  # Only process positive sequences
  python Functions/precompute_fegs.py --label pos

  # Custom split and FASTA directory
  python Functions/precompute_fegs.py --splits_dir Data/splits --topk 10

  # Re-compute even if .npz already exists
  python Functions/precompute_fegs.py --overwrite

Performance
-----------
  Matrix computation is O(L²) per motif per sequence.
  For L=1024, 10 motifs: ~10 ms per sequence on a modern CPU.
  1800 sequences ≈ 18 seconds single-threaded; < 5 s with 4 workers.

  Use --workers N to control parallelism (default: all CPUs − 1).
"""

import argparse
import os
import sys
import time
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from Bio import SeqIO

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Functions.RNA_FEGS.RNA_FEGS_feature_extraction import (
    _build_nt_coordinates,
    _GRS_static,
    DEFAULT_MOTIF_GROUPS,
    _normalise_rna,
)

# ── Defaults ──────────────────────────────────────────────────────────────────
SPLITS_DIR  = ROOT / "Data"  / "splits"
PROC_DIR    = ROOT / "Data"  / "processed"
SEQ_LEN     = 1024          # truncation limit
TOPK        = 10            # number of motif matrices to save per sequence


# =============================================================================
# Core matrix computation
# =============================================================================

def _compute_lhat(walk: np.ndarray) -> np.ndarray:
    """
    Compute the normalised walk-distance matrix Lhat from a GRS walk array.

    Parameters
    ----------
    walk : (L+1, 3)  GRS walk including origin at row 0

    Returns
    -------
    Lhat : (L, L) float32  normalised adjacency matrix
    """
    W = walk[1:]                            # drop the origin; shape (L, 3)
    L = W.shape[0]

    if L < 2:
        return np.zeros((L, L), dtype=np.float32)

    # ── Euclidean pairwise distances ──────────────────────────────────────────
    E = squareform(pdist(W)).astype(np.float64)   # (L, L)

    # ── Walk-path (geodesic) distances along the sequential walk ─────────────
    # Vectorised: step lengths are the L-1 consecutive Euclidean distances.
    # sdist[i,j] = sum of step lengths from position i to j along the walk
    #            = cumsteps[j] - cumsteps[i]  where cumsteps is the cumsum
    #              of consecutive step distances.
    steps     = np.diag(E, k=1)                          # (L-1,) consecutive steps
    cumsteps  = np.concatenate([[0.0], np.cumsum(steps)]) # (L,)
    i_idx, j_idx = np.triu_indices(L, k=1)
    sdist = np.zeros((L, L), dtype=np.float64)
    sdist[i_idx, j_idx] = cumsteps[j_idx] - cumsteps[i_idx]
    sdist += sdist.T                        # symmetrise

    # ── Normalised adjacency ──────────────────────────────────────────────────
    denom = sdist + np.eye(L, dtype=np.float64)   # +I avoids division by zero
    Lhat  = (E / denom).astype(np.float32)
    Lhat  = np.nan_to_num(Lhat, nan=0.0, posinf=0.0, neginf=0.0)
    return Lhat


def compute_fegs_matrices(
    seq:          str,
    motif_groups: list = DEFAULT_MOTIF_GROUPS,
    topk:         int  = TOPK,
    seq_len:      int  = SEQ_LEN,
) -> list:
    """
    Compute top-k RNA-FEGS Lhat matrices for a single sequence.

    Parameters
    ----------
    seq          : RNA string (AUGC; already normalised)
    motif_groups : list of nucleotide-group strings (length ≥ topk)
    topk         : how many motif groups to process
    seq_len      : hard truncation limit

    Returns
    -------
    list of topk (L, L) float32 np.ndarray  where L = min(len(seq), seq_len)
    """
    seq  = seq[:seq_len]                    # truncate
    P, V = _build_nt_coordinates()
    walks = _GRS_static(seq, P, V, motif_groups[:topk])
    return [_compute_lhat(w) for w in walks]


# =============================================================================
# Worker function (multiprocessing-safe)
# =============================================================================

def _worker(args: tuple) -> Optional[str]:
    """
    Process one (seq_idx, seq_id, seq) triple:
      - compute FEGS matrices
      - save .npz file named by zero-padded integer index (so alphabetical
        sort = FASTA order, which guarantees alignment with .npy arrays)
      - return the output path, or None if skipped/failed

    Designed to run in a subprocess — must be top-level for pickle.
    """
    seq_idx, seq_id, seq, out_dir, topk, seq_len, overwrite, motif_groups = args

    # Use zero-padded integer index so sorted() order = FASTA order
    out_path = Path(out_dir) / f"{seq_idx:08d}.npz"

    if out_path.exists() and not overwrite:
        return str(out_path)            # already done

    try:
        matrices = compute_fegs_matrices(seq, motif_groups, topk, seq_len)
        save_dict = {f"M{i}": m for i, m in enumerate(matrices)}
        np.savez_compressed(str(out_path), **save_dict)
        return str(out_path)
    except Exception as e:
        # Log but don't crash the whole pool
        print(f"\n[ERROR] {seq_id}: {e}")
        return None


# =============================================================================
# Per-split processing
# =============================================================================

def process_fasta(
    fasta_path:   Path,
    out_dir:      Path,
    topk:         int   = TOPK,
    seq_len:      int   = SEQ_LEN,
    workers:      int   = 1,
    overwrite:    bool  = False,
    motif_groups: list  = DEFAULT_MOTIF_GROUPS,
    start_idx:    int   = 0,       # global offset for zero-padded filenames
) -> list:
    """
    Compute and save FEGS .npz files for every sequence in a FASTA file.

    Parameters
    ----------
    fasta_path   : input FASTA
    out_dir      : directory where .npz files are written
    topk         : number of motif matrices per sequence
    seq_len      : truncation limit
    workers      : number of parallel worker processes
    overwrite    : recompute even if .npz already exists
    start_idx    : global index offset (so multiple FASTA files don't collide)

    Returns
    -------
    list of (seq_id, npz_path) pairs in FASTA order
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load sequences ────────────────────────────────────────────────────────
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        print(f"  [skip] {fasta_path.name} is empty")
        return []

    # Build (seq_id, seq) pairs, preserving FASTA order
    pairs = []
    seen  = {}
    for rec in records:
        base_id = rec.id.replace("/", "_").replace(" ", "_")[:80]
        # Deduplicate label by appending a counter (for the index.tsv only)
        count   = seen.get(base_id, 0)
        seen[base_id] = count + 1
        seq_id  = base_id if count == 0 else f"{base_id}_{count}"
        seq     = _normalise_rna(str(rec.seq))
        if len(seq) >= 10:
            pairs.append((seq_id, seq))

    # .npz filenames use global zero-padded index → sorted() = FASTA order
    npz_paths = [out_dir / f"{start_idx + i:08d}.npz" for i in range(len(pairs))]
    n_skip = sum(1 for p in npz_paths if p.exists() and not overwrite)
    n_todo = len(pairs) - n_skip

    print(f"\n  {fasta_path.name}: {len(pairs):,} sequences  "
          f"(global idx {start_idx}–{start_idx+len(pairs)-1}, "
          f"{n_skip} cached, {n_todo} to compute)")

    if n_todo == 0:
        return [(sid, str(npz_paths[i])) for i, (sid, _) in enumerate(pairs)]

    # ── Build worker args ─────────────────────────────────────────────────────
    # seq_idx is the global 0-based position across all FASTA files; used as
    # the zero-padded filename so sorted() order = FASTA order.
    worker_args = [
        (start_idx + i, sid, seq, str(out_dir), topk, seq_len, overwrite, motif_groups)
        for i, (sid, seq) in enumerate(pairs)
    ]

    # ── Run (parallel or serial) ──────────────────────────────────────────────
    t0      = time.time()
    results = []

    if workers > 1:
        with Pool(processes=workers) as pool:
            for path in tqdm(
                pool.imap(_worker, worker_args, chunksize=4),
                total=len(worker_args),
                desc=f"  FEGS [{fasta_path.stem}]",
                ncols=90,
            ):
                results.append(path)
    else:
        for wargs in tqdm(worker_args,
                          desc=f"  FEGS [{fasta_path.stem}]",
                          ncols=90):
            results.append(_worker(wargs))

    elapsed  = time.time() - t0
    n_ok     = sum(1 for r in results if r is not None)
    n_fail   = len(results) - n_ok
    print(f"  Done in {elapsed:.1f}s  |  "
          f"OK={n_ok}  failed={n_fail}  "
          f"({elapsed / max(n_todo, 1) * 1000:.1f} ms/seq)")

    # results[i] is the path to {start_idx+i:08d}.npz; its position in
    # sorted() order matches position (start_idx+i) across the full label set,
    # which aligns with the .npy row order from tokenizer_training.py.
    return [(sid, results[i]) for i, (sid, _) in enumerate(pairs)]


# =============================================================================
# Index file
# =============================================================================

def write_index(pairs: list, index_path: Path):
    """
    Write a two-column TSV mapping seq_id → npz_path.
    This makes it easy to reconstruct ordered lists of .npz paths
    matching the encoded .npy arrays.
    """
    with open(index_path, "w") as f:
        f.write("seq_id\tnpz_path\n")
        for sid, npz in pairs:
            if npz:
                f.write(f"{sid}\t{npz}\n")
    print(f"  Index written → {index_path}")


# =============================================================================
# Validation
# =============================================================================

def validate_npz(npz_path: str, topk: int = TOPK) -> dict:
    """
    Load one .npz and report its contents. Useful for spot-checking.
    Returns a dict with matrix shapes and value ranges.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        info = {"path": npz_path, "keys": list(z.files), "matrices": {}}
        for k in sorted(z.files):
            M = z[k]
            info["matrices"][k] = {
                "shape": M.shape,
                "min":   float(M.min()),
                "max":   float(M.max()),
                "nan":   int(np.isnan(M).sum()),
            }
    return info


def validate_batch(out_dir: Path, n_sample: int = 5, topk: int = TOPK):
    """Print a quick sanity report for n_sample random .npz files."""
    npz_files = sorted(out_dir.glob("*.npz"))
    if not npz_files:
        print(f"  [!] No .npz files found in {out_dir}")
        return

    sample = list(np.random.choice(npz_files, size=min(n_sample, len(npz_files)),
                                    replace=False))
    print(f"\nValidating {len(sample)} random .npz files from {out_dir.name}/:")
    all_ok = True
    for f in sample:
        info = validate_npz(str(f), topk)
        n_mats   = len(info["matrices"])
        has_nan  = any(v["nan"] > 0 for v in info["matrices"].values())
        shapes   = set(str(v["shape"]) for v in info["matrices"].values())
        status   = "✓" if (n_mats == topk and not has_nan) else "✗"
        if status == "✗":
            all_ok = False
        print(f"  {status} {Path(f).name:<50}  "
              f"mats={n_mats}/{topk}  "
              f"shapes={shapes}  "
              f"{'NaN!' if has_nan else 'clean'}")
    print(f"  → {'All OK' if all_ok else 'Some files have issues'}")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Precompute RNA-FEGS Lhat matrices for all dataset sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--splits_dir",  type=str, default=str(SPLITS_DIR))
    p.add_argument("--out_dir_pos", type=str,
                   default=str(PROC_DIR / "fegs_topk_pos"),
                   help="Output directory for positive sequence .npz files")
    p.add_argument("--out_dir_neg", type=str,
                   default=str(PROC_DIR / "fegs_topk_neg"),
                   help="Output directory for negative sequence .npz files")
    p.add_argument("--topk",        type=int, default=TOPK)
    p.add_argument("--seq_len",     type=int, default=SEQ_LEN)
    p.add_argument("--workers",     type=int,
                   default=max(1, cpu_count() - 1),
                   help="Number of parallel worker processes")
    p.add_argument("--label",       type=str, choices=["pos", "neg", "both"],
                   default="both", help="Which label to process")
    p.add_argument("--split",       type=str,
                   choices=["train", "val", "test", "all"],
                   default="all", help="Which split to process")
    p.add_argument("--fasta_pos",   type=str, default="",
                   help="Single positives FASTA. When set with --fasta_neg, "
                        "skip splits_dir and index .npz files by record order "
                        "in this FASTA. Required for hybrid-model alignment.")
    p.add_argument("--fasta_neg",   type=str, default="",
                   help="Single negatives FASTA (use with --fasta_pos).")
    p.add_argument("--overwrite",   action="store_true",
                   help="Recompute even if .npz already exists")
    p.add_argument("--validate",    action="store_true",
                   help="Run validation check on completed output dirs and exit")
    p.add_argument("--validate_n",  type=int, default=5,
                   help="Number of random files to validate")
    args = p.parse_args()

    splits_dir  = Path(args.splits_dir)
    out_pos     = Path(args.out_dir_pos)
    out_neg     = Path(args.out_dir_neg)

    # ── Validation mode ───────────────────────────────────────────────────────
    if args.validate:
        for d, lbl in [(out_pos, "positive"), (out_neg, "negative")]:
            if d.exists():
                print(f"\n{lbl.capitalize()} directory: {d}")
                validate_batch(d, n_sample=args.validate_n, topk=args.topk)
            else:
                print(f"  [skip] {d} does not exist yet")
        return

    # ── Determine which splits × labels to process ────────────────────────────
    splits_to_run = (["train", "val", "test"]
                     if args.split == "all" else [args.split])
    labels_to_run = (["pos", "neg"]
                     if args.label == "both" else [args.label])

    print(f"RNA-FEGS Precomputation")
    print(f"  topk    = {args.topk}")
    print(f"  seq_len = {args.seq_len}")
    print(f"  workers = {args.workers}")
    print(f"  splits  = {splits_to_run}")
    print(f"  labels  = {labels_to_run}")

    total_t0 = time.time()

    # Single-FASTA mode: align .npz files to a unified positives/negatives FASTA
    # so that filename order (sorted) == FASTA record order. Required for the
    # hybrid model which loads `all_positives_dedup.fasta` as a single source.
    single_mode = bool(args.fasta_pos and args.fasta_neg)

    # ── Process each split × label combination ────────────────────────────────
    for label in labels_to_run:
        out_dir = out_pos if label == "pos" else out_neg
        all_pairs = []
        global_idx = 0   # cumulative offset so filenames don't collide across FASTAs

        if single_mode:
            fasta = Path(args.fasta_pos if label == "pos" else args.fasta_neg)
            if not fasta.exists():
                print(f"\n  [skip] {fasta} not found")
                continue
            pairs = process_fasta(
                fasta_path   = fasta,
                out_dir      = out_dir,
                topk         = args.topk,
                seq_len      = args.seq_len,
                workers      = args.workers,
                overwrite    = args.overwrite,
                motif_groups = DEFAULT_MOTIF_GROUPS,
                start_idx    = 0,
            )
            all_pairs.extend(pairs)
        else:
            for split in splits_to_run:
                fasta = splits_dir / f"{split}_{label}.fasta"
                if not fasta.exists():
                    print(f"\n  [skip] {fasta.name} not found")
                    continue

                pairs = process_fasta(
                    fasta_path   = fasta,
                    out_dir      = out_dir,
                    topk         = args.topk,
                    seq_len      = args.seq_len,
                    workers      = args.workers,
                    overwrite    = args.overwrite,
                    motif_groups = DEFAULT_MOTIF_GROUPS,
                    start_idx    = global_idx,
                )
                global_idx += len(pairs)
                all_pairs.extend(pairs)

        if all_pairs:
            idx_path = out_dir / "index.tsv"
            write_index(all_pairs, idx_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s")

    # Count .npz files produced
    for out_dir, lbl in [(out_pos, "positive"), (out_neg, "negative")]:
        if out_dir.exists():
            n = len(list(out_dir.glob("*.npz")))
            print(f"  {lbl:10s}: {n:,} .npz files in {out_dir}")

    print(f"\nNext step: train the tokenizer and encode sequences:")
    print(f"  python Functions/tokenizer_training.py")
    print(f"\nThen train the model:")
    print(f"  python Functions/RNAPhaseek/RNAPhaseek_train.py")


if __name__ == "__main__":
    main()
