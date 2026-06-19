"""
RNAPhaseek Hybrid — Training Entry Point
==========================================
Run from the project root:
  python -m Functions.RNAPhaseek.RNAPhaseek_hybrid_train

Expects (same layout as the base model, just different inputs):
  Data/raw/all_positives_dedup.fasta   — positive LLPS sequences
  Data/raw/negatives.fasta             — negative sequences
  Data/processed/fegs_topk_pos/        — FEGS .npz files (positives)
  Data/processed/fegs_topk_neg/        — FEGS .npz files (negatives)
  Data/splits/biophys_pos.npy          — biophysical features (optional)
  Data/splits/biophys_neg.npy

The script tokenises sequences at training time using the RNA-FM tokenizer
(no pre-tokenisation step required).

FEGS .npz files must be ordered to match the FASTA file order.
If your FEGS files are named 00000000.npz … they are assumed to correspond
to the FASTA sequences in order.  Use --fegs_index if you have an index.tsv.
"""

import argparse
import os
import sys

import numpy as np
import multimolecule  # registers RnaFmModel / RnaTokenizer with the transformers AutoClasses

from .RNAPhaseek_utils          import set_seed, setup_device, list_npz_sorted
from .RNAPhaseek_hybrid_trainer import fit
from .RNAPhaseek_hybrid_data    import read_fasta
from . import RNAPhaseek_hybrid_config as CFG


def parse_args():
    p = argparse.ArgumentParser(description="Train RNAPhaseek Hybrid (RNA-FM + FEGSTrans)")

    # Backbone
    p.add_argument("--backbone",       default=CFG.RNA_FM_MODEL)
    p.add_argument("--freeze_backbone",action="store_true", default=True)
    p.add_argument("--unfreeze_last_n",type=int, default=0,
                   help="Unfreeze last N backbone layers.  0=fully frozen.  "
                        "Recommended: 0 while you have <1500 positives, then 2.")

    # Adapter
    p.add_argument("--n_adapter_layers",type=int,   default=CFG.HybridTrainArgs().n_adapter_layers)
    p.add_argument("--n_heads",         type=int,   default=CFG.HybridTrainArgs().n_heads)

    # FEGS
    p.add_argument("--topk_m",          type=int,   default=CFG.TOPK_M)

    # Training
    p.add_argument("--epochs",          type=int,   default=CFG.HybridTrainArgs().epochs)
    p.add_argument("--batch_size",      type=int,   default=CFG.HybridTrainArgs().batch_size)
    p.add_argument("--lr",              type=float, default=CFG.HybridTrainArgs().lr)
    p.add_argument("--backbone_lr",     type=float, default=CFG.HybridTrainArgs().backbone_lr)
    p.add_argument("--weight_decay",    type=float, default=CFG.HybridTrainArgs().weight_decay)
    p.add_argument("--warmup_frac",     type=float, default=CFG.HybridTrainArgs().warmup_frac)
    p.add_argument("--label_smooth",    type=float, default=CFG.HybridTrainArgs().label_smooth)
    p.add_argument("--patience",        type=int,   default=CFG.HybridTrainArgs().patience)
    p.add_argument("--num_workers",     type=int,   default=0,
                   help="Must be 0 on macOS (MPS/TCC sandbox restriction).")

    # Paths
    p.add_argument("--fasta_pos",  default=CFG.FASTA_POS)
    p.add_argument("--fasta_neg",  default=CFG.FASTA_NEG)
    p.add_argument("--src_pos",    default=CFG.SRC_POS)
    p.add_argument("--src_neg",    default=CFG.SRC_NEG)
    p.add_argument("--bio_pos",    default=CFG.BIO_POS)
    p.add_argument("--bio_neg",    default=CFG.BIO_NEG)
    p.add_argument("--bio_norm",   default=CFG.BIO_NORM)
    p.add_argument("--bio_dim",    type=int, default=CFG.BIO_DIM)
    p.add_argument("--no_bio",     action="store_true",
                   help="Disable biophysical features even if files exist.")
    p.add_argument("--best_ckpt",  default=CFG.BEST_CKPT)
    p.add_argument("--final_ckpt", default=CFG.FINAL_CKPT)
    p.add_argument("--fp16_bias",  action="store_true")
    p.add_argument("--init_from",  default="",
                   help="Path to a .pt with model weights to load BEFORE training. "
                        "Loads model state_dict only (no optimizer/scheduler). Useful "
                        "for Phase 2 fine-tuning that starts from Phase 1's best.")

    return p.parse_args()


def load_sequences_and_paths(fasta_path: str, fegs_dir: str) -> tuple[list[str], list[str]]:
    """
    Read FASTA and match each record to its FEGS .npz file.
    Matching is by position order: record i → sorted_npz[i].
    """
    records  = read_fasta(fasta_path)
    npz_list = list_npz_sorted(fegs_dir)

    if len(records) != len(npz_list):
        print(
            f"[WARN] FASTA has {len(records)} records but {len(npz_list)} .npz files "
            f"in {fegs_dir}.  Truncating to the shorter list.", flush=True
        )
        n = min(len(records), len(npz_list))
        records  = records[:n]
        npz_list = npz_list[:n]

    seqs  = [seq for _, seq in records]
    paths = npz_list
    return seqs, paths


def main():
    a = parse_args()
    set_seed(42)
    device = setup_device()

    os.makedirs("model", exist_ok=True)

    # ── Load sequences & FEGS paths ──────────────────────────────────────────
    print("Loading positive sequences …", flush=True)
    pos_seqs, pos_paths = load_sequences_and_paths(a.fasta_pos, a.src_pos)
    print(f"  {len(pos_seqs)} positive sequences", flush=True)

    print("Loading negative sequences …", flush=True)
    neg_seqs, neg_paths = load_sequences_and_paths(a.fasta_neg, a.src_neg)
    print(f"  {len(neg_seqs)} negative sequences", flush=True)

    all_seqs  = pos_seqs  + neg_seqs
    all_paths = np.array(pos_paths + neg_paths, dtype=object)
    y         = np.concatenate([
        np.ones(len(pos_seqs),  dtype=np.int64),
        np.zeros(len(neg_seqs), dtype=np.int64),
    ])
    print(f"\nTotal: {len(y)}  |  Pos: {int(y.sum())}  |  Neg: {int((y == 0).sum())}")

    # ── Optional biophysical features ────────────────────────────────────────
    X_bio = None
    if not a.no_bio and os.path.exists(a.bio_pos) and os.path.exists(a.bio_neg):
        bio_pos = np.load(a.bio_pos).astype(np.float32)
        bio_neg = np.load(a.bio_neg).astype(np.float32)
        if len(bio_pos) == len(pos_seqs) and len(bio_neg) == len(neg_seqs):
            X_bio = np.vstack([bio_pos, bio_neg])
            print(f"Biophysical features loaded: shape={X_bio.shape}")
        else:
            print("[WARN] Biophysical feature count does not match sequences — skipping.")
    else:
        print("Biophysical features: not found or disabled — training without them.")

    # ── Load RNA-FM tokenizer ────────────────────────────────────────────────
    print(f"\nLoading tokenizer from {a.backbone} …", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.backbone, trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] Could not load tokenizer: {e}")
        print("Make sure the backbone model is accessible:")
        print(f"  pip install transformers  # if not installed")
        print(f"  huggingface-cli login     # if model is gated")
        sys.exit(1)

    # ── Build HybridTrainArgs ─────────────────────────────────────────────────
    train_args = CFG.HybridTrainArgs(
        backbone           = a.backbone,
        freeze_backbone    = a.freeze_backbone,
        unfreeze_last_n    = a.unfreeze_last_n,
        n_adapter_layers   = a.n_adapter_layers,
        n_heads            = a.n_heads,
        topk_m             = a.topk_m,
        bio_dim            = a.bio_dim if X_bio is not None else 0,
        batch_size         = a.batch_size,
        num_workers        = a.num_workers,
        fp16_bias          = a.fp16_bias,
        epochs             = a.epochs,
        lr                 = a.lr,
        backbone_lr        = a.backbone_lr,
        weight_decay       = a.weight_decay,
        warmup_frac        = a.warmup_frac,
        label_smooth       = a.label_smooth,
        patience           = a.patience,
        src_pos            = a.src_pos,
        src_neg            = a.src_neg,
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
        paths     = all_paths,
        y         = y,
        args      = train_args,
        device    = device,
        tokenizer = tokenizer,
        X_bio     = X_bio,
        init_from = a.init_from,
    )


if __name__ == "__main__":
    main()
