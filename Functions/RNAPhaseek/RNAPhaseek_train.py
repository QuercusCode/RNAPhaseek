"""
RNAPhaseek — Training Entry Point
====================================
Run from the project root:
  python Functions/RNAPhaseek/RNAPhaseek_train.py [options]

Expects:
  Data/processed/fegs_topk_pos/  — .npz files (RNA-FEGS matrices, positive)
  Data/processed/fegs_topk_neg/  — .npz files (RNA-FEGS matrices, negative)
  Data/splits/train_pos.fasta, train_neg.fasta  — BPE-tokenised arrays
  Data/splits/biophys_pos.npy, biophys_neg.npy  — precomputed biophysical features (optional)
"""

import argparse
import os
import numpy as np

from .RNAPhaseek_utils   import set_seed, setup_device, list_npz_sorted
from .RNAPhaseek_trainer import fit
from . import RNAPhaseek_config as CFG


def parse_args():
    p = argparse.ArgumentParser(description="Train RNAPhaseek")
    p.add_argument("--src_pos",    type=str,   default=CFG.SRC_POS)
    p.add_argument("--src_neg",    type=str,   default=CFG.SRC_NEG)
    p.add_argument("--epochs",     type=int,   default=CFG.EPOCHS)
    p.add_argument("--seq_len",    type=int,   default=CFG.SEQ_LEN)
    p.add_argument("--topk_m",     type=int,   default=CFG.TOPK_M)
    p.add_argument("--n_layers",   type=int,   default=CFG.N_LAYERS)
    p.add_argument("--d_model",    type=int,   default=CFG.D_MODEL)
    p.add_argument("--n_heads",    type=int,   default=CFG.N_HEADS)
    p.add_argument("--batch_size", type=int,   default=CFG.BATCH_SIZE)
    p.add_argument("--num_workers",type=int,   default=CFG.NUM_WORKERS)
    p.add_argument("--lr",         type=float, default=CFG.LR)
    p.add_argument("--weight_decay",type=float,default=CFG.WEIGHT_DECAY)
    p.add_argument("--warmup_frac",type=float, default=CFG.WARMUP_FRAC)
    p.add_argument("--label_smooth",type=float,default=CFG.LABEL_SMOOTH)
    p.add_argument("--best_ckpt",  type=str,   default=CFG.BEST_CKPT)
    p.add_argument("--final_ckpt", type=str,   default=CFG.FINAL_CKPT)
    p.add_argument("--fp16_bias",  action="store_true", default=CFG.FP16_BIAS)
    p.add_argument("--bio_pos",    type=str,   default=CFG.BIO_POS)
    p.add_argument("--bio_neg",    type=str,   default=CFG.BIO_NEG)
    p.add_argument("--bio_norm",   type=str,   default=CFG.BIO_NORM)
    p.add_argument("--bio_dim",    type=int,   default=CFG.BIO_DIM)
    p.add_argument("--no_bio",     action="store_true", default=False,
                   help="Disable biophysical features even if files exist")
    return p.parse_args()


def main():
    args_ns = parse_args()
    set_seed(42)
    device = setup_device()

    pos_npz = list_npz_sorted(args_ns.src_pos)
    neg_npz = list_npz_sorted(args_ns.src_neg)

    # ── Load BPE-encoded sequences ────────────────────────────────────────────
    # Sequences should be pre-tokenised and saved as numpy arrays (.npy).
    # Each file: (N, seq_len) int array, padding token = 0.
    pos_seq_path = "Data/splits/pos_seq_encoded.npy"
    neg_seq_path = "Data/splits/neg_seq_encoded.npy"
    pos_seq = np.load(pos_seq_path)
    neg_seq = np.load(neg_seq_path)

    assert len(pos_seq) == len(pos_npz), \
        f"Mismatch: {len(pos_seq)} sequences vs {len(pos_npz)} .npz files (positive)"
    assert len(neg_seq) == len(neg_npz), \
        f"Mismatch: {len(neg_seq)} sequences vs {len(neg_npz)} .npz files (negative)"

    X_seq = np.vstack([pos_seq, neg_seq])
    paths = np.array(pos_npz + neg_npz, dtype=object)
    y     = np.concatenate([
        np.ones(len(pos_seq),  dtype=np.int64),
        np.zeros(len(neg_seq), dtype=np.int64),
    ])
    print(f"Total samples: {len(y)} | Pos: {int(y.sum())} | Neg: {int((y==0).sum())}")

    # ── Optional biophysical features (RNA2PS + ENCORI, 26 dims) ─────────────
    X_bio = None
    if not args_ns.no_bio and os.path.exists(args_ns.bio_pos) and os.path.exists(args_ns.bio_neg):
        bio_pos = np.load(args_ns.bio_pos).astype(np.float32)
        bio_neg = np.load(args_ns.bio_neg).astype(np.float32)
        assert len(bio_pos) == len(pos_seq), \
            f"Bio/seq mismatch (pos): {len(bio_pos)} vs {len(pos_seq)}"
        assert len(bio_neg) == len(neg_seq), \
            f"Bio/seq mismatch (neg): {len(bio_neg)} vs {len(neg_seq)}"
        X_bio = np.vstack([bio_pos, bio_neg])
        print(f"Biophysical features loaded: shape={X_bio.shape}")
    else:
        print("Biophysical features: not found or disabled — running without them")

    train_args = CFG.TrainArgs(
        src_pos      = args_ns.src_pos,
        src_neg      = args_ns.src_neg,
        topk_m       = args_ns.topk_m,
        seq_len      = args_ns.seq_len,
        n_layers     = args_ns.n_layers,
        d_model      = args_ns.d_model,
        n_heads      = args_ns.n_heads,
        batch_size   = args_ns.batch_size,
        num_workers  = args_ns.num_workers,
        fp16_bias    = args_ns.fp16_bias,
        epochs       = args_ns.epochs,
        lr           = args_ns.lr,
        weight_decay = args_ns.weight_decay,
        warmup_frac  = args_ns.warmup_frac,
        label_smooth = args_ns.label_smooth,
        best_ckpt    = args_ns.best_ckpt,
        final_ckpt   = args_ns.final_ckpt,
        bio_pos      = args_ns.bio_pos,
        bio_neg      = args_ns.bio_neg,
        bio_norm     = args_ns.bio_norm,
        bio_dim      = args_ns.bio_dim if X_bio is not None else 0,
    )

    fit(X_seq=X_seq, paths=paths, y=y, args=train_args, device=device, X_bio=X_bio)


if __name__ == "__main__":
    main()
