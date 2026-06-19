"""
RNAPhaseek Hybrid — Training Loop
====================================
Mirrors RNAPhaseek_trainer.py but drives the RNA-FM + FEGSTrans adapter model.

Key differences:
  - Tokenizer is passed to make_dataloaders (RNA-FM nucleotide tokenizer).
  - Optimizer uses two param groups: adapter/head (lr) + backbone (backbone_lr).
  - AMP is disabled on MPS; only enabled on CUDA.
  - Resume checkpoint stores tokenizer path alongside model state.
"""

import math
import os

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from .RNAPhaseek_hybrid         import RNAFMHybridClassifier
from .RNAPhaseek_hybrid_config  import HybridTrainArgs
from .RNAPhaseek_hybrid_data    import make_dataloaders


# ── LR scheduler (cosine with warm-up) ───────────────────────────────────────

def make_scheduler(optimizer, total_steps: int, warmup_frac: float):
    warmup = max(10, int(warmup_frac * total_steps))
    def lr_lambda(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: RNAFMHybridClassifier, loader, device: str) -> dict:
    model.eval()
    tot_loss, tot_n, correct = 0.0, 0, 0
    all_probs, all_labels    = [], []

    n_sanitised = 0
    for token_ids, att_mask, Lhat, bio, yb in loader:
        # IMPORTANT: clone labels on CPU BEFORE moving to MPS.
        # PyTorch MPS has a known bug where `yb.cpu()` after `.to("mps")` can
        # return stale/garbage device memory (pointer-like values), so we keep
        # a pristine CPU copy to use for metrics / accuracy.
        yb_cpu = yb.detach().clone().long()
        yb     = yb.to(device, non_blocking=True)

        token_ids = token_ids.to(device, non_blocking=True)
        att_mask  = att_mask.to(device, non_blocking=True)
        Lhat      = Lhat.to(device, non_blocking=True)
        bio       = bio.to(device, non_blocking=True) if bio is not None else None

        logits, loss = model(
            token_ids, att_mask,
            labels=yb, Lhat_stack=Lhat, bio_features=bio,
        )

        # MPS attention can occasionally emit NaN for specific input patterns.
        # Replace non-finite logits with zeros (uniform 0.5/0.5 prediction) so
        # those examples don't poison loss / softmax / sklearn metrics.
        non_finite = ~torch.isfinite(logits).all(dim=-1)
        if non_finite.any():
            n_sanitised += int(non_finite.sum())
            logits = torch.where(non_finite[:, None], torch.zeros_like(logits), logits)
            loss = torch.nn.functional.cross_entropy(
                logits, yb, label_smoothing=model.label_smooth
            )

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds_cpu = logits.argmax(dim=-1).cpu()

        tot_loss += float(loss.item()) * token_ids.size(0)
        tot_n    += token_ids.size(0)
        correct  += int((preds_cpu == yb_cpu).sum())
        all_probs.append(probs.detach().cpu())
        all_labels.append(yb_cpu)
    if n_sanitised:
        print(f"  [val-sanitise] replaced non-finite logits in {n_sanitised} samples "
              f"with uniform 0.5/0.5 predictions.", flush=True)

    probs_np  = torch.cat(all_probs).numpy().astype(np.float32).flatten()
    labels_np = torch.cat(all_labels).numpy().astype(np.int64).flatten()

    model.train()

    # Diagnostics — print EVERYTHING relevant so future failures are debuggable.
    unique_lbl, lbl_counts = np.unique(labels_np, return_counts=True)
    n_pos       = int((labels_np == 1).sum())
    n_neg       = int((labels_np == 0).sum())
    n_other     = int(((labels_np != 0) & (labels_np != 1)).sum())
    n_nan_probs = int(np.sum(~np.isfinite(probs_np)))
    p_min, p_max, p_mean = float(probs_np.min()), float(probs_np.max()), float(probs_np.mean())
    print(f"  [val-diag] n={len(labels_np)} pos={n_pos} neg={n_neg} other={n_other} "
          f"probs[min={p_min:.4f} max={p_max:.4f} mean={p_mean:.4f}] nan_probs={n_nan_probs}",
          flush=True)
    if n_other:
        print(f"  [val-diag] WARNING: {n_other} labels are not 0 or 1. "
              f"unique={dict(zip(unique_lbl.tolist(), lbl_counts.tolist()))}. "
              f"Clamping to {{0,1}}.", flush=True)
        labels_np = np.clip(labels_np, 0, 1)
        n_pos = int((labels_np == 1).sum())
        n_neg = int((labels_np == 0).sum())
    if n_nan_probs:
        print(f"  [WARN] {n_nan_probs}/{len(probs_np)} val probs are non-finite; "
              f"replacing with 0.5.", flush=True)
        probs_np = np.where(np.isfinite(probs_np), probs_np, 0.5)

    # Compute AUROC/PR-AUC defensively — never crash training on a metric.
    try:
        auroc = float(roc_auc_score(labels_np, probs_np)) if n_pos > 0 and n_neg > 0 else float("nan")
    except Exception as e:
        print(f"  [val-diag] roc_auc_score failed: {type(e).__name__}: {e}", flush=True)
        auroc = float("nan")
    try:
        prauc = float(average_precision_score(labels_np, probs_np)) if n_pos > 0 and n_neg > 0 else float("nan")
    except Exception as e:
        print(f"  [val-diag] average_precision_score failed: {type(e).__name__}: {e}", flush=True)
        prauc = float("nan")

    return {
        "loss":   tot_loss / max(1, tot_n),
        "acc":    correct  / max(1, tot_n),
        "auroc":  auroc,
        "prauc":  prauc,
        "probs":  probs_np,
        "labels": labels_np,
    }


# ── Main fit function ─────────────────────────────────────────────────────────

def fit(
    seqs:   list[str],
    paths:  np.ndarray,
    y:      np.ndarray,
    args:   HybridTrainArgs,
    device: str,
    tokenizer,
    X_bio:  np.ndarray = None,
    init_from: str = "",
):
    # ── Train / val split ─────────────────────────────────────────────────────
    split_inputs = [seqs, paths.tolist(), y]
    if X_bio is not None:
        split_inputs.append(X_bio)
        out = train_test_split(*split_inputs, test_size=0.15, random_state=42, stratify=y)
        seqs_tr, seqs_va, ptr, pva, ytr, yva, bio_tr, bio_va = out
        ptr, pva = np.array(ptr), np.array(pva)
        # Z-score normalise biophysical features using train stats
        if os.path.exists(args.bio_norm):
            z = np.load(args.bio_norm)
            m, s = z["mean"], z["std"]
        else:
            m = bio_tr.mean(axis=0)
            s = bio_tr.std(axis=0).clip(min=1e-8)
        bio_tr = (bio_tr - m) / s
        bio_va = (bio_va - m) / s
        print(f"Biophysical features: {bio_tr.shape[1]} dims (RNA2PS + ENCORI)")
    else:
        out = train_test_split(*split_inputs, test_size=0.15, random_state=42, stratify=y)
        seqs_tr, seqs_va, ptr, pva, ytr, yva = out
        ptr, pva = np.array(ptr), np.array(pva)
        bio_tr = bio_va = None

    print(f"Train: {len(seqs_tr)}  |  Val: {len(seqs_va)}", flush=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        seqs_tr, seqs_va, ptr.tolist(), pva.tolist(),
        ytr, yva, tokenizer, args,
        bio_tr=bio_tr, bio_va=bio_va,
    )

    # ── Model / optimizer / scheduler ────────────────────────────────────────
    model = RNAFMHybridClassifier(args).to(device)

    # Phase 2 fine-tune: load Phase 1 weights BEFORE building optimizer, so the
    # optimizer is created against the (possibly newly-unfrozen) trainable set.
    if init_from and os.path.exists(init_from):
        print(f"Loading Phase 1 weights from {init_from} ...", flush=True)
        state = torch.load(init_from, map_location=device, weights_only=True)
        # Phase 1 saved as a bare state_dict; tolerate either format.
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  init_from: missing={len(missing)} unexpected={len(unexpected)} keys", flush=True)

    optimizer = model.configure_optimizers(args)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler   = make_scheduler(optimizer, total_steps, args.warmup_frac)

    # AMP only on CUDA; disabled on MPS and CPU
    amp_on = (device == "cuda")
    scaler = GradScaler("cuda", enabled=amp_on)

    # ── Early stopping (AUROC-based) ──────────────────────────────────────────
    best_auroc = 0.0
    patience, pat = args.patience, 0
    resume_ckpt   = args.best_ckpt.replace(".pt", "_resume.pt")

    # ── Resume from checkpoint if available ──────────────────────────────────
    start_epoch = 0
    if os.path.exists(resume_ckpt):
        print(f"Resuming from {resume_ckpt} …", flush=True)
        state = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if amp_on and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_auroc  = state["best_auroc"]
        pat         = state["patience"]
        print(f"  Resumed at epoch {start_epoch + 1}, "
              f"best_auroc={best_auroc:.4f}, patience={pat}/{patience}", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    from tqdm import tqdm
    MID_EPOCH_CKPT_EVERY = 250   # save resume state every N batches within an epoch
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=120)
        for step, (token_ids, att_mask, Lhat, bio, yb) in enumerate(pbar):
            token_ids = token_ids.to(device, non_blocking=True)
            att_mask  = att_mask.to(device, non_blocking=True)
            Lhat      = Lhat.to(device, non_blocking=True)
            yb        = yb.to(device, non_blocking=True)
            bio       = bio.to(device, non_blocking=True) if bio is not None else None

            with autocast(device_type="cuda", enabled=amp_on):
                _, loss = model(
                    token_ids, att_mask,
                    labels=yb, Lhat_stack=Lhat, bio_features=bio,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            # Mid-epoch resume checkpoint (atomic via rename) — survives crashes
            # mid-training, so we don't lose 20+ minutes if epoch 1 dies near the end.
            if (step + 1) % MID_EPOCH_CKPT_EVERY == 0:
                tmp = resume_ckpt + ".tmp"
                torch.save({
                    "epoch":      epoch - 1,  # treat as "previous epoch fully done at restart"
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "scaler":     scaler.state_dict(),
                    "best_auroc": best_auroc,
                    "patience":   pat,
                }, tmp)
                os.replace(tmp, resume_ckpt)

        # ── Epoch evaluation ──────────────────────────────────────────────────
        m = evaluate(model, val_loader, device)
        improved = m["auroc"] > best_auroc + 1e-4
        flag = " ✓" if improved else f"  (best={best_auroc:.4f}, pat {pat + 1}/{patience})"
        print(
            f"  → val_loss={m['loss']:.4f} | acc={m['acc']:.4f} "
            f"| AUROC={m['auroc']:.4f} | PR-AUC={m['prauc']:.4f}{flag}",
            flush=True,
        )

        if improved:
            best_auroc = m["auroc"]
            pat = 0
            torch.save(model.state_dict(), args.best_ckpt)
            print(f"  Checkpoint saved → {args.best_ckpt}", flush=True)
        else:
            pat += 1
            if pat >= patience:
                print("Early stopping.", flush=True)
                break

        # Save full resume state after every epoch
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "scaler":     scaler.state_dict(),
            "best_auroc": best_auroc,
            "patience":   pat,
        }, resume_ckpt)
        print(f"  Resume state saved → {resume_ckpt}", flush=True)

    # ── Final evaluation on val set (best checkpoint) ─────────────────────────
    if os.path.exists(args.best_ckpt):
        model.load_state_dict(torch.load(args.best_ckpt, map_location=device, weights_only=True))
    final = evaluate(model, val_loader, device)
    print("\n=== Final (best checkpoint) ===")
    print(f"val_loss={final['loss']:.4f} | acc={final['acc']:.4f} "
          f"| AUROC={final['auroc']:.4f} | PR-AUC={final['prauc']:.4f}")

    # ── Optimal threshold sweep ───────────────────────────────────────────────
    from sklearn.metrics import f1_score as _f1, matthews_corrcoef as _mcc
    best_f1_t, best_f1_v   = 0.5, 0.0
    best_mcc_t, best_mcc_v = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.02):
        preds = (final["probs"] >= t).astype(int)
        f1    = _f1(final["labels"],  preds, zero_division=0)
        mcc   = _mcc(final["labels"], preds)
        if f1  > best_f1_v:  best_f1_v  = f1;  best_f1_t  = t
        if mcc > best_mcc_v: best_mcc_v = mcc; best_mcc_t = t
    print(f"\nOptimal threshold (val F1={best_f1_v:.4f})  → {best_f1_t:.2f}")
    print(f"Optimal threshold (val MCC={best_mcc_v:.4f}) → {best_mcc_t:.2f}")

    # ── Learned β per adapter block ───────────────────────────────────────────
    print("\nLearned β (graph-bias weight) per adapter block:")
    for i, blk in enumerate(model.adapter):
        print(f"  adapter block {i}: β={blk.attn.beta.detach().cpu().item():.4f}")

    torch.save(model.state_dict(), args.final_ckpt)
    print(f"\nFinal weights saved → {args.final_ckpt}")

    thresh_path = args.best_ckpt.replace(".pt", "_thresh.npy")
    np.save(thresh_path, {"f1": float(best_f1_t), "mcc": float(best_mcc_t)})
    print(f"Thresholds saved  → {thresh_path}")
    return model
