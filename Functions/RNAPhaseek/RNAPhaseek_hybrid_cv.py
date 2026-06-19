"""
RNAPhaseek Hybrid — 5-fold Cross-Validation on the strict RNA-self-LLPS pool.

Each fold:
  - new 33-dim biophysical features (with absolute repeat/periodicity)
  - unfreeze last N RNA-FM layers (default 2)
  - warm-start from Phase-1
  - early stopping on the held-out fold (AUROC)

Aggregates:
  - per-fold best AUROC -> mean ± std
  - pooled out-of-fold predictions -> overall AUROC + PR-AUC
  - easy/hard diagnostic on pooled predictions (pos vs matched vs hard negs)

Resumable: completed folds are read from fold_results.json and skipped.

Run:
  python -m Functions.RNAPhaseek.RNAPhaseek_hybrid_cv
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import multimolecule  # noqa: F401
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from Functions.RNAPhaseek.RNAPhaseek_hybrid         import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config  import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data    import read_fasta, make_dataloaders
from Functions.RNAPhaseek.RNAPhaseek_hybrid_trainer import evaluate, make_scheduler
from Functions.RNAPhaseek.RNAPhaseek_utils          import list_npz_sorted, setup_device, set_seed

FASTA_POS = "Data/raw/multispecies/strict_pool_positives.fasta"
FASTA_NEG = "Data/raw/multispecies/strict_pool_negatives_all.fasta"
SRC_POS   = "Data/processed/fegs_topk_strict_pos"
SRC_NEG   = "Data/processed/fegs_topk_strict_neg"
BIO_POS   = "Data/splits/biophys_strict_pos.npy"
BIO_NEG   = "Data/splits/biophys_strict_neg.npy"
INIT_FROM = "model/phase1/hybrid_best.pt"
OUT_DIR   = "model/strict_cv"
RESULTS   = os.path.join(OUT_DIR, "fold_results.json")


def neg_subtype(hdr: str) -> str:
    h = hdr.lower()
    if "hardneg_terra" in h: return "hard:TERRA"
    if "hardneg_shuffle" in h: return "hard:shuffle"
    if "hardneg_" in h: return "hard:subthreshold"
    return "matched"


def train_one_fold(fold, tr_idx, va_idx, all_seqs, all_paths, all_hdrs, y, bio,
                   tokenizer, args, device):
    seqs_tr = [all_seqs[i] for i in tr_idx]; seqs_va = [all_seqs[i] for i in va_idx]
    ptr     = [all_paths[i] for i in tr_idx]; pva    = [all_paths[i] for i in va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    bio_tr, bio_va = bio[tr_idx], bio[va_idx]

    # z-score from train fold only (no leakage)
    m = bio_tr.mean(0); sd = bio_tr.std(0).clip(min=1e-8)
    bio_tr_n = (bio_tr - m) / sd
    bio_va_n = (bio_va - m) / sd

    train_loader, val_loader = make_dataloaders(
        seqs_tr, seqs_va, ptr, pva, ytr, yva, tokenizer, args,
        bio_tr=bio_tr_n, bio_va=bio_va_n,
    )

    model = RNAFMHybridClassifier(args).to(device)
    if os.path.exists(INIT_FROM):
        state = torch.load(INIT_FROM, map_location=device, weights_only=True)
        sd_in = state["model"] if isinstance(state, dict) and "model" in state else state
        # Shape-filter: strict=False ignores missing/unexpected keys but NOT size
        # mismatches. bio_proj changed 26->33, so drop mismatched keys and let
        # them reinitialize; everything else (backbone, adapter, head) transfers.
        msd = model.state_dict()
        keep = {k: v for k, v in sd_in.items() if k in msd and v.shape == msd[k].shape}
        dropped = [k for k in sd_in if k not in keep]
        miss, unexp = model.load_state_dict(keep, strict=False)
        print(f"  [fold {fold}] init_from: loaded={len(keep)} dropped(shape)={len(dropped)} "
              f"reinit={len(miss)} ({[d.split('.')[0] for d in dropped][:3]})", flush=True)

    optimizer = model.configure_optimizers(args)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_frac)

    best_auroc, best_state, pat = -1.0, None, 0
    best_probs, best_labels = None, None
    for epoch in range(args.epochs):
        model.train()
        for token_ids, att, Lhat, biob, yb in train_loader:
            token_ids = token_ids.to(device); att = att.to(device)
            Lhat = Lhat.to(device); yb = yb.to(device)
            biob = biob.to(device) if biob is not None else None
            _, loss = model(token_ids, att, labels=yb, Lhat_stack=Lhat, bio_features=biob)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        mval = evaluate(model, val_loader, device)
        improved = mval["auroc"] > best_auroc + 1e-4
        print(f"  [fold {fold}] epoch {epoch+1}/{args.epochs} "
              f"AUROC={mval['auroc']:.4f} PR-AUC={mval['prauc']:.4f}"
              f"{'  *' if improved else f'  (best={max(best_auroc,0):.4f} pat {pat+1}/{args.patience})'}",
              flush=True)
        if improved:
            best_auroc = mval["auroc"]; pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_probs, best_labels = mval["probs"].copy(), mval["labels"].copy()
        else:
            pat += 1
            if pat >= args.patience:
                print(f"  [fold {fold}] early stop @ epoch {epoch+1}", flush=True)
                break

    # Save best checkpoint for this fold
    if best_state is not None:
        torch.save(best_state, os.path.join(OUT_DIR, f"fold{fold}_best.pt"))

    # out-of-fold predictions WITH headers (rebuild in val order; evaluate keeps order)
    va_hdrs = [all_hdrs[i] for i in va_idx]
    return {
        "fold": fold,
        "best_auroc": float(best_auroc),
        "val_headers": va_hdrs,
        "val_probs": best_probs.tolist() if best_probs is not None else [],
        "val_labels": best_labels.tolist() if best_labels is not None else [],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--unfreeze_last_n", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.03)
    a = p.parse_args()

    set_seed(42)
    device = setup_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load data (FASTA order) ──
    pos = read_fasta(FASTA_POS); neg = read_fasta(FASTA_NEG)
    pp = list_npz_sorted(SRC_POS); np_ = list_npz_sorted(SRC_NEG)
    n_pos = min(len(pos), len(pp)); n_neg = min(len(neg), len(np_))
    pos, pp = pos[:n_pos], pp[:n_pos]; neg, np_ = neg[:n_neg], np_[:n_neg]
    all_seqs = [s for _, s in pos] + [s for _, s in neg]
    all_hdrs = [h for h, _ in pos] + [h for h, _ in neg]
    all_paths = list(pp) + list(np_)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
    bio = np.vstack([np.load(BIO_POS), np.load(BIO_NEG)]).astype(np.float32)
    print(f"Loaded {len(y)}  pos={n_pos} neg={n_neg}  bio_dim={bio.shape[1]}", flush=True)

    args = HybridTrainArgs(
        bio_dim=bio.shape[1], use_species_embed=False,
        unfreeze_last_n=a.unfreeze_last_n, freeze_backbone=(a.unfreeze_last_n == 0),
        epochs=a.epochs, patience=a.patience, lr=a.lr,
        backbone_lr=a.backbone_lr, weight_decay=a.weight_decay,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

    # ── Resume: load completed folds ──
    completed = {}
    if os.path.exists(RESULTS):
        completed = {r["fold"]: r for r in json.load(open(RESULTS))}
        print(f"Resuming — {len(completed)} folds already done: {sorted(completed)}", flush=True)

    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=42)
    results = list(completed.values())
    for fold, (tr_idx, va_idx) in enumerate(skf.split(all_seqs, y)):
        if fold in completed:
            continue
        print(f"\n===== FOLD {fold}/{a.folds-1}  (train={len(tr_idx)} val={len(va_idx)}) =====", flush=True)
        r = train_one_fold(fold, tr_idx, va_idx, all_seqs, all_paths, all_hdrs,
                           y, bio, tokenizer, args, device)
        results.append(r)
        json.dump(results, open(RESULTS, "w"))   # incremental save
        print(f"  [fold {fold}] best AUROC = {r['best_auroc']:.4f}", flush=True)

    # ── Aggregate ──
    results = sorted(results, key=lambda r: r["fold"])
    aurocs = [r["best_auroc"] for r in results]
    print("\n" + "=" * 60)
    print("5-FOLD CV SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  fold {r['fold']}: AUROC = {r['best_auroc']:.4f}")
    print(f"\n  MEAN AUROC = {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")

    # Pooled out-of-fold
    P = np.concatenate([np.array(r["val_probs"]) for r in results])
    L = np.concatenate([np.array(r["val_labels"]) for r in results])
    H = sum([r["val_headers"] for r in results], [])
    print(f"\n  Pooled out-of-fold AUROC = {roc_auc_score(L, P):.4f}  "
          f"PR-AUC = {average_precision_score(L, P):.4f}  (n={len(L)})")

    # Easy/hard diagnostic on pooled predictions
    sub = np.array([neg_subtype(h) if l == 0 else "positive" for h, l in zip(H, L)])
    easy = (sub == "positive") | (sub == "matched")
    if (L[easy] == 1).sum() and (L[easy] == 0).sum():
        print(f"  EASY (pos vs matched)        AUROC = {roc_auc_score(L[easy], P[easy]):.4f}")
    print("\n  Separability (pooled OOF) — positives vs each negative type:")
    pos_mask = L == 1
    for c in ["matched", "hard:subthreshold", "hard:shuffle", "hard:TERRA"]:
        msk = sub == c
        if msk.sum() < 3:
            print(f"    pos vs {c:<18} n={int(msk.sum())} (too few)"); continue
        yy = np.concatenate([np.ones(int(pos_mask.sum())), np.zeros(int(msk.sum()))])
        pp_ = np.concatenate([P[pos_mask], P[msk]])
        print(f"    pos vs {c:<18} n={int(msk.sum()):<4} AUROC = {roc_auc_score(yy, pp_):.4f}")

    json.dump({
        "mean_auroc": float(np.mean(aurocs)), "std_auroc": float(np.std(aurocs)),
        "fold_aurocs": aurocs,
        "pooled_oof_auroc": float(roc_auc_score(L, P)),
        "pooled_oof_prauc": float(average_precision_score(L, P)),
    }, open(os.path.join(OUT_DIR, "cv_summary.json"), "w"), indent=2)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'cv_summary.json')}")


if __name__ == "__main__":
    main()
