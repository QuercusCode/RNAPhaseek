"""
RNAPhaseek Hybrid — Test-Set Evaluation
==========================================
Loads the best hybrid checkpoint and evaluates on a held-out FASTA + FEGS set.

Run:
  python -m Functions.RNAPhaseek.RNAPhaseek_hybrid_evaluate

Reports: loss, accuracy, AUROC, PR-AUC, F1, MCC, confusion matrix,
sensitivity, specificity, precision, and learned β per adapter block.
"""

import argparse
import os

import numpy as np
import torch
import multimolecule  # noqa: F401 — registers RnaFmModel / RnaTokenizer with transformers
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, confusion_matrix,
)

from .RNAPhaseek_utils          import setup_device, list_npz_sorted
from .RNAPhaseek_hybrid         import RNAFMHybridClassifier
from .RNAPhaseek_hybrid_config  import HybridTrainArgs
from .RNAPhaseek_hybrid_data    import read_fasta, HybridRNADataset, make_collate_fn
from torch.utils.data           import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RNAPhaseek Hybrid on test set")
    p.add_argument("--fasta_pos",   default="Data/splits/test_pos.fasta")
    p.add_argument("--fasta_neg",   default="Data/splits/test_neg.fasta")
    p.add_argument("--src_pos",     default="Data/processed/fegs_topk_pos")
    p.add_argument("--src_neg",     default="Data/processed/fegs_topk_neg")
    p.add_argument("--bio_pos",     default="Data/splits/biophys_test_pos.npy")
    p.add_argument("--bio_neg",     default="Data/splits/biophys_test_neg.npy")
    p.add_argument("--bio_norm",    default="Data/splits/biophys_norm_stats.npz")
    p.add_argument("--bio_dim",     type=int, default=26)
    p.add_argument("--ckpt",        default="model/hybrid_best.pt")
    p.add_argument("--thresh",      type=float, default=None,
                   help="Decision threshold.  If omitted, uses saved threshold (F1-optimal).")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--topk_m",      type=int, default=10)
    p.add_argument("--backbone",    default="multimolecule/rnafm")
    p.add_argument("--n_adapter_layers", type=int, default=2)
    p.add_argument("--n_heads",     type=int, default=8)
    p.add_argument("--no_bio",      action="store_true")
    return p.parse_args()


def load_data(fasta_pos, fasta_neg, src_pos, src_neg,
              bio_pos, bio_neg, bio_norm, bio_dim, no_bio):
    def match(fasta, fegs_dir):
        records  = read_fasta(fasta)
        npz_list = list_npz_sorted(fegs_dir)
        n = min(len(records), len(npz_list))
        return [s for _, s in records[:n]], npz_list[:n]

    pos_seqs, pos_paths = match(fasta_pos, src_pos)
    neg_seqs, neg_paths = match(fasta_neg, src_neg)
    seqs  = pos_seqs + neg_seqs
    paths = pos_paths + neg_paths
    y     = np.concatenate([np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))]).astype(np.int64)

    X_bio = None
    if not no_bio and os.path.exists(bio_pos) and os.path.exists(bio_neg):
        bp = np.load(bio_pos).astype(np.float32)
        bn = np.load(bio_neg).astype(np.float32)
        if len(bp) == len(pos_seqs) and len(bn) == len(neg_seqs):
            bio = np.vstack([bp, bn])
            if os.path.exists(bio_norm):
                z  = np.load(bio_norm)
                m, s = z["mean"], z["std"]
                bio = (bio - m) / s.clip(min=1e-8)
            X_bio = bio

    return seqs, paths, y, X_bio


def main():
    a      = parse_args()
    device = setup_device()

    # ── Threshold ─────────────────────────────────────────────────────────────
    thresh = a.thresh
    if thresh is None:
        thresh_path = a.ckpt.replace(".pt", "_thresh.npy")
        if os.path.exists(thresh_path):
            d      = np.load(thresh_path, allow_pickle=True).item()
            thresh = float(d.get("f1", 0.5))
            print(f"Using saved F1-optimal threshold: {thresh:.2f}")
        else:
            thresh = 0.5
            print("Using default threshold: 0.50")

    # ── Load data ─────────────────────────────────────────────────────────────
    seqs, paths, y, X_bio = load_data(
        a.fasta_pos, a.fasta_neg,
        a.src_pos,   a.src_neg,
        a.bio_pos,   a.bio_neg, a.bio_norm, a.bio_dim, a.no_bio,
    )
    print(f"Test set: {len(y)}  |  Pos: {int(y.sum())}  |  Neg: {int((y==0).sum())}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.backbone, trust_remote_code=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    args = HybridTrainArgs(
        backbone          = a.backbone,
        n_adapter_layers  = a.n_adapter_layers,
        n_heads           = a.n_heads,
        topk_m            = a.topk_m,
        bio_dim           = a.bio_dim if X_bio is not None else 0,
    )
    model = RNAFMHybridClassifier(args).to(device)
    state = torch.load(a.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {a.ckpt}")

    # ── DataLoader ─────────────────────────────────────────────────────────────
    ds      = HybridRNADataset(seqs, paths, y, bio_array=X_bio)
    collate = make_collate_fn(tokenizer, topk_m=a.topk_m)
    loader  = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate)

    # ── Inference ─────────────────────────────────────────────────────────────
    tot_loss, tot_n = 0.0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for token_ids, att_mask, Lhat, bio, yb in loader:
            token_ids = token_ids.to(device)
            att_mask  = att_mask.to(device)
            Lhat      = Lhat.to(device)
            yb        = yb.to(device)
            bio       = bio.to(device) if bio is not None else None

            logits, loss = model(token_ids, att_mask,
                                 labels=yb, Lhat_stack=Lhat, bio_features=bio)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            tot_loss += float(loss.item()) * token_ids.size(0)
            tot_n    += token_ids.size(0)
            all_probs.append(probs.cpu())
            all_labels.append(yb.cpu())

    probs_np  = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    preds_np  = (probs_np >= thresh).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auroc = roc_auc_score(labels_np, probs_np)
    prauc = average_precision_score(labels_np, probs_np)
    f1    = f1_score(labels_np, preds_np, zero_division=0)
    mcc   = matthews_corrcoef(labels_np, preds_np)
    cm    = confusion_matrix(labels_np, preds_np)
    tn, fp, fn, tp = cm.ravel()
    sens  = tp / max(1, tp + fn)
    spec  = tn / max(1, tn + fp)
    prec  = tp / max(1, tp + fp)

    print(f"\n{'='*50}")
    print(f"Loss          : {tot_loss / max(1, tot_n):.4f}")
    print(f"Accuracy      : {(preds_np == labels_np).mean():.4f}")
    print(f"AUROC         : {auroc:.4f}")
    print(f"PR-AUC        : {prauc:.4f}")
    print(f"F1 (LLPS+)    : {f1:.4f}   @ threshold {thresh:.2f}")
    print(f"MCC           : {mcc:.4f}")
    print(f"\nConfusion matrix (threshold={thresh:.2f}):")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"\nSensitivity : {sens:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Precision   : {prec:.4f}")

    print("\nLearned β per adapter block:")
    for i, blk in enumerate(model.adapter):
        print(f"  adapter block {i}: β={blk.attn.beta.detach().cpu().item():.4f}")


if __name__ == "__main__":
    main()
