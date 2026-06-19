"""
CLI entry point for full-sequence hybrid training.

Run:
    python -m Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_train

Or use launch_hybrid_fullseq_detached.py to run in a session that survives
harness reloads.
"""

import argparse
import os
import sys

import numpy as np
import multimolecule  # noqa: F401 -- registers RnaFmModel / RnaTokenizer

from .RNAPhaseek_utils                 import set_seed, setup_device
from .RNAPhaseek_hybrid_fullseq_trainer import fit
from .RNAPhaseek_hybrid_fullseq_data    import read_fasta
from . import RNAPhaseek_hybrid_fullseq_config as CFG


def parse_args():
    p = argparse.ArgumentParser(description="Train RNAPhaseek hybrid (full-sequence, multi-window)")
    defaults = CFG.HybridFullSeqArgs()
    p.add_argument("--backbone",         default=defaults.backbone)
    p.add_argument("--n_adapter_layers", type=int,   default=defaults.n_adapter_layers)
    p.add_argument("--n_heads",          type=int,   default=defaults.n_heads)
    p.add_argument("--window",           type=int,   default=defaults.window)
    p.add_argument("--stride",           type=int,   default=defaults.stride)
    p.add_argument("--max_windows",      type=int,   default=defaults.max_windows)
    p.add_argument("--epochs",           type=int,   default=defaults.epochs)
    p.add_argument("--batch_size",       type=int,   default=defaults.batch_size)
    p.add_argument("--lr",               type=float, default=defaults.lr)
    p.add_argument("--weight_decay",     type=float, default=defaults.weight_decay)
    p.add_argument("--warmup_frac",      type=float, default=defaults.warmup_frac)
    p.add_argument("--label_smooth",     type=float, default=defaults.label_smooth)
    p.add_argument("--patience",         type=int,   default=defaults.patience)
    p.add_argument("--num_workers",      type=int,   default=0)
    p.add_argument("--fasta_pos",        default=defaults.fasta_pos)
    p.add_argument("--fasta_neg",        default=defaults.fasta_neg)
    p.add_argument("--bio_pos",          default=defaults.bio_pos)
    p.add_argument("--bio_neg",          default=defaults.bio_neg)
    p.add_argument("--bio_norm",         default=defaults.bio_norm)
    p.add_argument("--bio_dim",          type=int,   default=defaults.bio_dim)
    p.add_argument("--no_bio",           action="store_true")
    p.add_argument("--best_ckpt",        default=defaults.best_ckpt)
    p.add_argument("--final_ckpt",       default=defaults.final_ckpt)
    p.add_argument("--init_from",        default="",
                   help="Load model weights from this .pt before training (e.g. Phase 1)")
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(42)
    device = setup_device()
    os.makedirs("model", exist_ok=True)

    # ── Load sequences (no FEGS paths needed) ────────────────────────────────
    print("Loading positive sequences ...", flush=True)
    pos_records = read_fasta(a.fasta_pos)
    pos_seqs = [s for _, s in pos_records]
    print(f"  {len(pos_seqs)} positive sequences", flush=True)

    print("Loading negative sequences ...", flush=True)
    neg_records = read_fasta(a.fasta_neg)
    neg_seqs = [s for _, s in neg_records]
    print(f"  {len(neg_seqs)} negative sequences", flush=True)

    all_seqs = pos_seqs + neg_seqs
    y = np.concatenate([
        np.ones(len(pos_seqs),  dtype=np.int64),
        np.zeros(len(neg_seqs), dtype=np.int64),
    ])
    print(f"\nTotal: {len(y)}  |  Pos: {int(y.sum())}  |  Neg: {int((y == 0).sum())}")

    # Length statistics so the user can see what's being processed
    lens = np.array([len(s) for s in all_seqs])
    print(f"Sequence length stats: min={lens.min()} median={int(np.median(lens))} "
          f"mean={lens.mean():.0f} max={lens.max()}", flush=True)

    # ── Optional biophysical features ────────────────────────────────────────
    X_bio = None
    if not a.no_bio and os.path.exists(a.bio_pos) and os.path.exists(a.bio_neg):
        bio_pos = np.load(a.bio_pos).astype(np.float32)
        bio_neg = np.load(a.bio_neg).astype(np.float32)
        if len(bio_pos) == len(pos_seqs) and len(bio_neg) == len(neg_seqs):
            X_bio = np.vstack([bio_pos, bio_neg])
            print(f"Biophysical features loaded: shape={X_bio.shape}")
        else:
            print("[WARN] Biophysical feature count does not match sequences — disabled.")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer from {a.backbone} ...", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.backbone, trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] Could not load tokenizer: {e}")
        sys.exit(1)

    # ── Build args + fit ─────────────────────────────────────────────────────
    train_args = CFG.HybridFullSeqArgs(
        backbone           = a.backbone,
        n_adapter_layers   = a.n_adapter_layers,
        n_heads            = a.n_heads,
        window             = a.window,
        stride             = a.stride,
        max_windows        = a.max_windows,
        bio_dim            = a.bio_dim if X_bio is not None else 0,
        batch_size         = a.batch_size,
        num_workers        = a.num_workers,
        epochs             = a.epochs,
        lr                 = a.lr,
        weight_decay       = a.weight_decay,
        warmup_frac        = a.warmup_frac,
        label_smooth       = a.label_smooth,
        patience           = a.patience,
        fasta_pos          = a.fasta_pos,
        fasta_neg          = a.fasta_neg,
        bio_pos            = a.bio_pos,
        bio_neg            = a.bio_neg,
        bio_norm           = a.bio_norm,
        best_ckpt          = a.best_ckpt,
        final_ckpt         = a.final_ckpt,
    )

    fit(
        seqs      = all_seqs,
        y         = y,
        args      = train_args,
        device    = device,
        tokenizer = tokenizer,
        X_bio     = X_bio,
        init_from = a.init_from,
    )


if __name__ == "__main__":
    main()
