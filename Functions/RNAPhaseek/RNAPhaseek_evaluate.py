"""
RNAPhaseek Test-Set Evaluation
================================
Loads the best checkpoint and evaluates on the held-out test split.

Run from the project root:
    python Functions/RNAPhaseek/RNAPhaseek_evaluate.py [options]

Expects (all relative to the project root):
    model/rna_phaseek_best.pt
    Data/splits/test_pos.fasta  / test_neg.fasta
    Data/splits/test_pos_encoded.npy / test_neg_encoded.npy
    Data/splits/biophys_test_pos.npy / biophys_test_neg.npy
    Data/splits/biophys_norm_stats.npz
    Data/splits/pos_seq_encoded.npy  / neg_seq_encoded.npy  (to infer vocab size)
    Data/processed/fegs_topk_pos/index.tsv
    Data/processed/fegs_topk_neg/index.tsv
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, confusion_matrix,
)

# Ensure project root is on sys.path for package imports
_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Functions.RNAPhaseek.RNAPhaseek      import RNAPhaseekClassifier, Config
from Functions.RNAPhaseek.RNAPhaseek_data import (
    RNASeqTopkDataset, collate_fn,
)
import Functions.RNAPhaseek.RNAPhaseek_data   as DataMod
import Functions.RNAPhaseek.RNAPhaseek_config as CFG


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_fasta_headers(fasta_path: str) -> list:
    """Return sequence IDs (header text after '>') in FASTA order."""
    headers = []
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                headers.append(line[1:].strip())
    return headers


def load_fegs_index(index_path: str) -> dict:
    """Load index.tsv → {seq_id: npz_path}."""
    idx = {}
    with open(index_path) as fh:
        next(fh)  # skip header row
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                idx[parts[0]] = parts[1]
    return idx


def get_npz_paths(headers: list, fegs_index: dict) -> np.ndarray:
    """Map FASTA sequence headers to FEGS .npz paths via index.

    Some index entries were created from FASTA files that contained literal tab
    characters in the header line (e.g. '...lncRNA|le\\tn=3436'), which caused
    the header to be truncated at the tab when the TSV was written.  To handle
    this, fall back to a prefix match if an exact match is not found.
    """
    # Pre-build a sorted list of (key, path) for prefix matching
    sorted_keys = sorted(fegs_index.keys(), key=len, reverse=True)

    paths = []
    missing = []
    for h in headers:
        if h in fegs_index:
            paths.append(fegs_index[h])
        else:
            # Prefix fallback: index key is a prefix of the FASTA header
            match = None
            for k in sorted_keys:
                if h.startswith(k):
                    match = fegs_index[k]
                    break
            if match is not None:
                paths.append(match)
            else:
                missing.append(h)
    if missing:
        raise KeyError(
            f"{len(missing)} sequence(s) not found in FEGS index.\n"
            f"First missing: {missing[0]}"
        )
    return np.array(paths)


def build_model(vocab_size: int, device: str) -> RNAPhaseekClassifier:
    model_cfg = Config(
        vocab_size  = vocab_size,
        block_size  = CFG.SEQ_LEN,
        n_layer     = CFG.N_LAYERS,
        n_head      = CFG.N_HEADS,
        n_embd      = CFG.D_MODEL,
        embd_pdrop  = 0.1,
        resid_pdrop = 0.1,
        attn_pdrop  = 0.1,
        causal         = False,
        use_graph_bias = True,
    )
    return RNAPhaseekClassifier(
        model_cfg,
        topk_m       = CFG.TOPK_M,
        label_smooth = CFG.LABEL_SMOOTH,
        weight_decay = CFG.WEIGHT_DECAY,
        bio_dim      = CFG.BIO_DIM,
    ).to(device)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device: str) -> dict:
    model.eval()
    tot_loss, tot_n, n_correct = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []

    for batch in loader:
        xb, yb, Lhat, bio = batch
        xb   = xb.to(device, non_blocking=True)
        yb   = yb.to(device, non_blocking=True)
        Lhat = Lhat.to(device, non_blocking=True)
        bio  = bio.to(device, non_blocking=True) if bio is not None else None

        logits, loss = model(xb, yb, Lhat_stack=Lhat, bio_features=bio)
        probs  = torch.softmax(logits, dim=-1)[:, 1]
        preds  = logits.argmax(dim=-1)

        tot_loss  += float(loss.item()) * xb.size(0)
        tot_n     += xb.size(0)
        n_correct += int((preds == yb).sum())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())

    probs_np  = torch.cat(all_probs).numpy()
    preds_np  = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    has_pos = labels_np.sum() > 0
    return {
        "loss":   tot_loss / max(1, tot_n),
        "acc":    n_correct / max(1, tot_n),
        "auroc":  roc_auc_score(labels_np, probs_np)          if has_pos else float("nan"),
        "prauc":  average_precision_score(labels_np, probs_np) if has_pos else float("nan"),
        "f1":     f1_score(labels_np, preds_np, zero_division=0),
        "mcc":    matthews_corrcoef(labels_np, preds_np),
        "cm":     confusion_matrix(labels_np, preds_np),
        "probs":  probs_np,
        "labels": labels_np,
    }


# ── Main evaluation routine ───────────────────────────────────────────────────

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ── Load encoded sequences ────────────────────────────────────────────────
    print("\nLoading test arrays…")
    test_pos_enc = np.load(args.test_pos_enc)           # (46, 1024)
    test_neg_enc = np.load(args.test_neg_enc)           # (85, 1024)
    bio_pos_raw  = np.load(args.bio_test_pos).astype(np.float32)   # (46, 26)
    bio_neg_raw  = np.load(args.bio_test_neg).astype(np.float32)   # (85, 26)

    # Z-score normalise with the statistics computed on the training split
    ns   = np.load(args.bio_norm)
    mu   = ns["mean"].astype(np.float32)
    sig  = ns["std"].astype(np.float32).clip(min=1e-8)
    bio_pos = (bio_pos_raw - mu) / sig
    bio_neg = (bio_neg_raw - mu) / sig

    # ── Resolve FEGS .npz paths ───────────────────────────────────────────────
    pos_headers  = parse_fasta_headers(args.test_pos_fasta)
    neg_headers  = parse_fasta_headers(args.test_neg_fasta)
    pos_fegs_idx = load_fegs_index(args.pos_fegs_index)
    neg_fegs_idx = load_fegs_index(args.neg_fegs_index)
    pos_paths    = get_npz_paths(pos_headers, pos_fegs_idx)
    neg_paths    = get_npz_paths(neg_headers, neg_fegs_idx)

    # Sanity-check alignment between FASTA order and encoded array
    n_pos, n_neg = len(test_pos_enc), len(test_neg_enc)
    assert len(pos_paths) == n_pos == len(bio_pos), (
        f"Positive count mismatch: enc={n_pos}, paths={len(pos_paths)}, bio={len(bio_pos)}"
    )
    assert len(neg_paths) == n_neg == len(bio_neg), (
        f"Negative count mismatch: enc={n_neg}, paths={len(neg_paths)}, bio={len(bio_neg)}"
    )

    # Combine pos and neg
    X_seq = np.vstack([test_pos_enc, test_neg_enc])
    y     = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    paths = np.concatenate([pos_paths, neg_paths])
    bio   = np.vstack([bio_pos, bio_neg])

    print(f"  Positive : {n_pos}")
    print(f"  Negative : {n_neg}")
    print(f"  Total    : {len(X_seq)}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    DataMod.TOPK_M    = CFG.TOPK_M
    DataMod.SEQ_LEN   = CFG.SEQ_LEN
    DataMod.FP16_BIAS = CFG.FP16_BIAS

    test_ds = RNASeqTopkDataset(X_seq, y, paths, bio_array=bio)
    test_loader = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 0,          # macOS TCC sandbox requires 0
        pin_memory  = (device == "cuda"),
        collate_fn  = collate_fn,
    )

    # ── Build model and load checkpoint ──────────────────────────────────────
    # Infer vocab_size from the complete encoded arrays (same logic as trainer)
    full_pos = np.load(args.full_pos_enc)
    full_neg = np.load(args.full_neg_enc)
    vocab_size = int(max(full_pos.max(initial=0), full_neg.max(initial=0))) + 1
    print(f"\nVocab size : {vocab_size}")

    model = build_model(vocab_size, device)
    print(f"Parameters : {model.num_parameters:,}")

    ckpt_path = args.checkpoint
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"Checkpoint : {ckpt_path}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\nRunning inference…")
    m = run_inference(model, test_loader, device)

    # ── Pretty-print results ──────────────────────────────────────────────────
    divider = "=" * 56
    print(f"\n{divider}")
    print("  RNAPhaseek  —  Test-Set Evaluation")
    print(divider)
    print(f"  Checkpoint  :  {ckpt_path}")
    print(f"  Sequences   :  {len(X_seq)}  (pos={n_pos}, neg={n_neg})")
    print(divider)
    print(f"  Loss        :  {m['loss']:.4f}")
    print(f"  Accuracy    :  {m['acc']:.4f}  ({m['acc']*100:.1f} %)")
    print(f"  AUROC       :  {m['auroc']:.4f}")
    print(f"  PR-AUC      :  {m['prauc']:.4f}")
    print(f"  F1 (LLPS+)  :  {m['f1']:.4f}")
    print(f"  MCC         :  {m['mcc']:.4f}")
    cm = m["cm"]
    print(f"\n  Confusion matrix  (rows = true, cols = predicted)")
    print(f"                  Pred 0    Pred 1")
    print(f"    True 0 (neg): {cm[0,0]:>6}    {cm[0,1]:>6}")
    print(f"    True 1 (pos): {cm[1,0]:>6}    {cm[1,1]:>6}")
    # Per-class breakdown
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    print(f"\n  Sensitivity (recall)  : {sens:.4f}")
    print(f"  Specificity           : {spec:.4f}")
    print(f"  Precision             : {prec:.4f}")
    print(divider)

    # ── Learned β values ─────────────────────────────────────────────────────
    print("\n  Learned β (graph-bias weight) per Transformer block:")
    for i, blk in enumerate(model.transformer.h):
        print(f"    block {i}: β = {blk.attn.beta.detach().cpu().item():.4f}")

    # ── Optional save ─────────────────────────────────────────────────────────
    if args.save_results:
        result_dict = {
            "loss": float(m["loss"]),
            "acc":  float(m["acc"]),
            "auroc": float(m["auroc"]),
            "prauc": float(m["prauc"]),
            "f1":   float(m["f1"]),
            "mcc":  float(m["mcc"]),
            "cm":   m["cm"].tolist(),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "precision":   float(prec),
        }
        np.save(args.save_results, result_dict)
        print(f"\nResults saved → {args.save_results}")

    return m


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RNAPhaseek on the held-out test split"
    )
    parser.add_argument(
        "--checkpoint",
        default="model/rna_phaseek_best.pt",
        help="Path to best model checkpoint (.pt)",
    )
    parser.add_argument("--test_pos_fasta", default="Data/splits/test_pos.fasta")
    parser.add_argument("--test_neg_fasta", default="Data/splits/test_neg.fasta")
    parser.add_argument("--test_pos_enc",   default="Data/splits/test_pos_encoded.npy")
    parser.add_argument("--test_neg_enc",   default="Data/splits/test_neg_encoded.npy")
    parser.add_argument("--bio_test_pos",   default="Data/splits/biophys_test_pos.npy")
    parser.add_argument("--bio_test_neg",   default="Data/splits/biophys_test_neg.npy")
    parser.add_argument("--bio_norm",       default="Data/splits/biophys_norm_stats.npz")
    parser.add_argument(
        "--pos_fegs_index",
        default="Data/processed/fegs_topk_pos/index.tsv",
        help="index.tsv mapping seq_id → FEGS .npz path (positive set)",
    )
    parser.add_argument(
        "--neg_fegs_index",
        default="Data/processed/fegs_topk_neg/index.tsv",
        help="index.tsv mapping seq_id → FEGS .npz path (negative set)",
    )
    parser.add_argument(
        "--full_pos_enc",
        default="Data/splits/pos_seq_encoded.npy",
        help="Full positive encoded array (all splits) — used to infer vocab size",
    )
    parser.add_argument(
        "--full_neg_enc",
        default="Data/splits/neg_seq_encoded.npy",
        help="Full negative encoded array (all splits) — used to infer vocab size",
    )
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument(
        "--save_results",
        default="",
        help="If set, save result dict as .npy file at this path",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
