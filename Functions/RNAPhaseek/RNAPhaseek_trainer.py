"""
RNAPhaseek Training Loop
=========================
Ported from Phaseek_v3_trainer.py with RNA-specific logging and
cross-organism evaluation hooks.
"""

import math
import os
import numpy as np
import torch
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from .RNAPhaseek     import RNAPhaseekClassifier, Config
from .RNAPhaseek_data    import make_dataloaders
from . import RNAPhaseek_config as CFG


def ensure_fegs_local(paths: np.ndarray) -> None:
    """
    Force iCloud Drive to download every FEGS .npz file before training starts.

    macOS iCloud Drive evicts files from local storage after first access.
    On the second training run those files time out ([Errno 60]) because the
    OS has to re-download them.  Reading each file once here ensures they are
    all present on local SSD for the duration of training.
    """
    import time
    unique = sorted(set(paths.tolist()))
    missing, ok = [], 0
    print(f"Preflight: checking {len(unique)} FEGS files are locally available …")
    t0 = time.time()
    for p in unique:
        if not os.path.exists(p):
            missing.append(p); continue
        try:
            with open(p, "rb") as fh:
                fh.read(64)          # read 64 bytes — enough to trigger iCloud download
            ok += 1
        except OSError as e:
            missing.append(p)
            print(f"  [WARN] Cannot prefetch {p}: {e}")
    elapsed = time.time() - t0
    print(f"  ✓ {ok}/{len(unique)} files confirmed local  ({elapsed:.1f}s)")
    if missing:
        print(f"  ✗ {len(missing)} files unavailable — they will use zero-bias fallback")


def build_model(vocab_size: int, args: CFG.TrainArgs, device: str) -> RNAPhaseekClassifier:
    model_cfg = Config(
        vocab_size  = vocab_size,
        block_size  = args.seq_len,
        n_layer     = args.n_layers,
        n_head      = args.n_heads,
        n_embd      = args.d_model,
        embd_pdrop  = 0.1,
        resid_pdrop = 0.1,
        attn_pdrop  = 0.1,
        causal         = False,
        use_graph_bias = True,
    )
    model = RNAPhaseekClassifier(
        model_cfg, topk_m=args.topk_m,
        label_smooth=args.label_smooth,
        weight_decay=args.weight_decay,
        bio_dim=getattr(args, 'bio_dim', 0),
    ).to(device)
    print(f"Model parameters: {model.num_parameters:,}")
    return model


def make_scheduler(optimizer, total_steps: int, warmup_frac: float):
    warmup = max(10, int(warmup_frac * total_steps))
    def lr_lambda(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = float(step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    model.eval()
    tot_loss, tot_n, correct = 0.0, 0, 0
    all_probs, all_labels    = [], []

    n_sanitised = 0
    for batch in loader:
        xb, yb, Lhat, bio = batch
        # Pristine CPU copy of labels BEFORE moving to MPS — avoids the known
        # MPS bug where yb.cpu() can return stale device memory (garbage values).
        yb_cpu = yb.detach().clone().long()
        xb   = xb.to(device, non_blocking=True)
        yb   = yb.to(device, non_blocking=True)
        Lhat = Lhat.to(device, non_blocking=True)
        bio  = bio.to(device, non_blocking=True) if bio is not None else None
        logits, loss = model(xb, yb, Lhat_stack=Lhat, bio_features=bio)

        # Sanitise non-finite logits from intermittent MPS attention NaN issue.
        non_finite = ~torch.isfinite(logits).all(dim=-1)
        if non_finite.any():
            n_sanitised += int(non_finite.sum())
            logits = torch.where(non_finite[:, None], torch.zeros_like(logits), logits)
            loss = torch.nn.functional.cross_entropy(
                logits, yb, label_smoothing=getattr(model, "label_smooth", 0.0)
            )

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds_cpu = logits.argmax(dim=-1).cpu()

        tot_loss += float(loss.item()) * xb.size(0)
        tot_n    += xb.size(0)
        correct  += int((preds_cpu == yb_cpu).sum())
        all_probs.append(probs.detach().cpu())
        all_labels.append(yb_cpu)
    if n_sanitised:
        print(f"  [val-sanitise] replaced non-finite logits in {n_sanitised} samples "
              f"with uniform 0.5/0.5 predictions.", flush=True)

    probs_np  = torch.cat(all_probs).numpy().astype(np.float32).flatten()
    labels_np = torch.cat(all_labels).numpy().astype(np.int64).flatten()

    # Defensive diagnostics — survives any sklearn detection of multiclass etc.
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
        probs_np = np.where(np.isfinite(probs_np), probs_np, 0.5)

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

    metrics = {
        "loss":   tot_loss / max(1, tot_n),
        "acc":    correct  / max(1, tot_n),
        "auroc":  auroc,
        "prauc":  prauc,
        "probs":  probs_np,
        "labels": labels_np,
    }
    model.train()
    return metrics


def fit(
    X_seq:  np.ndarray,
    paths:  np.ndarray,
    y:      np.ndarray,
    args:   CFG.TrainArgs,
    device: str,
    X_bio:  np.ndarray = None,   # (N, 26) biophysical features, optional
):
    # ── Split ─────────────────────────────────────────────────────────────────
    split_inputs = [X_seq, paths, y]
    if X_bio is not None:
        split_inputs.append(X_bio)
        Xtr, Xva, ptr, pva, ytr, yva, bio_tr, bio_va = train_test_split(
            *split_inputs, test_size=0.15, random_state=42, stratify=y
        )
        # Z-score normalise bio features using training stats
        if hasattr(args, 'bio_norm') and os.path.exists(args.bio_norm):
            z = np.load(args.bio_norm)
            m, s = z['mean'], z['std']
        else:
            m  = bio_tr.mean(axis=0)
            s  = bio_tr.std(axis=0).clip(min=1e-8)
        bio_tr = (bio_tr - m) / s
        bio_va = (bio_va - m) / s
        print(f"Biophysical features: {bio_tr.shape[1]} dims (RNA2PS + ENCORI)")
    else:
        Xtr, Xva, ptr, pva, ytr, yva = train_test_split(
            *split_inputs, test_size=0.15, random_state=42, stratify=y
        )
        bio_tr = bio_va = None

    vocab_size = int(max(Xtr.max(initial=0), Xva.max(initial=0))) + 1

    # ── Prefetch all FEGS files from iCloud Drive (prevents Errno 60 timeouts) ─
    ensure_fegs_local(np.concatenate([ptr, pva]))

    # sync data-module globals
    from . import RNAPhaseek_data as DataMod
    DataMod.TOPK_M    = args.topk_m
    DataMod.SEQ_LEN   = args.seq_len
    DataMod.FP16_BIAS = args.fp16_bias

    train_loader, val_loader = make_dataloaders(
        Xtr, Xva, ptr, pva, ytr, yva,
        args.batch_size, args.num_workers, args.prefetch, device,
        bio_tr=bio_tr, bio_va=bio_va,
    )

    # ── Model / optimiser / scheduler ────────────────────────────────────────
    model     = build_model(vocab_size, args, device)
    optimiser = model.configure_optimizers(lr=args.lr, betas=(0.9, 0.95))
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler   = make_scheduler(optimiser, total_steps, args.warmup_frac)

    # AMP: CUDA only.  MPS and CPU run in full precision.
    amp_on = (device == "cuda")
    scaler = GradScaler("cuda", enabled=amp_on)

    # ── Early stopping monitors AUROC (higher = better), not loss.
    # Val-loss is too noisy on small validation sets (~200 samples);
    # AUROC is a rank-based metric that is more stable epoch-to-epoch.
    best_auroc = 0.0
    patience, pat = 10, 0          # was 6 — give more room on noisy val
    resume_ckpt = args.best_ckpt.replace(".pt", "_resume.pt")

    # ── Resume from checkpoint if available ──────────────────────────────────
    start_epoch = 0
    if os.path.exists(resume_ckpt):
        print(f"Resuming from {resume_ckpt} ...")
        state = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if amp_on:
            scaler.load_state_dict(state["scaler"])
        start_epoch  = state["epoch"] + 1
        best_auroc   = state["best_auroc"]
        pat          = state["patience"]
        print(f"  Resumed at epoch {start_epoch+1}, "
              f"best_auroc={best_auroc:.4f}, patience={pat}/{patience}")

    for epoch in range(start_epoch, args.epochs):
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=110)
        for xb, yb, Lhat, bio in pbar:
            xb   = xb.to(device, non_blocking=True)
            yb   = yb.to(device, non_blocking=True)
            Lhat = Lhat.to(device, non_blocking=True)
            bio  = bio.to(device, non_blocking=True) if bio is not None else None

            with autocast(device_type="cuda", enabled=amp_on):
                _, loss = model(xb, yb, Lhat_stack=Lhat, bio_features=bio)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser); scaler.update(); scheduler.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        m = evaluate(model, val_loader, device)
        improved = m["auroc"] > best_auroc + 1e-4
        flag = " ✓" if improved else f"  (best={best_auroc:.4f}, pat {pat+1}/{patience})"
        print(f"  → val_loss={m['loss']:.4f} | acc={m['acc']:.4f} "
              f"| AUROC={m['auroc']:.4f} | PR-AUC={m['prauc']:.4f}{flag}")

        if improved:
            best_auroc = m["auroc"]; pat = 0
            torch.save(model.state_dict(), args.best_ckpt)
            print(f"  Checkpoint saved → {args.best_ckpt}")
        else:
            pat += 1
            if pat >= patience:
                print("Early stopping.")
                break

        # ── Save full resume state after every epoch ──────────────────────
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimiser.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "scaler":     scaler.state_dict(),
            "best_auroc": best_auroc,
            "patience":   pat,
        }, resume_ckpt)
        print(f"  Resume state saved → {resume_ckpt}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    if os.path.exists(args.best_ckpt):
        model.load_state_dict(torch.load(args.best_ckpt, map_location=device))
    final = evaluate(model, val_loader, device)
    print("\n=== Final (best checkpoint) ===")
    print(f"val_loss={final['loss']:.4f} | acc={final['acc']:.4f} "
          f"| AUROC={final['auroc']:.4f} | PR-AUC={final['prauc']:.4f}")

    # ── Optimal threshold sweep on val set ───────────────────────────────────
    from sklearn.metrics import f1_score as _f1, matthews_corrcoef as _mcc
    best_f1_t, best_f1_v   = 0.5, 0.0
    best_mcc_t, best_mcc_v = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.02):
        preds = (final["probs"] >= t).astype(int) if "probs" in final else None
        if preds is None:
            break
        f1  = _f1(final["labels"],  preds, zero_division=0)
        mcc = _mcc(final["labels"], preds)
        if f1  > best_f1_v:  best_f1_v=f1;   best_f1_t=t
        if mcc > best_mcc_v: best_mcc_v=mcc; best_mcc_t=t
    print(f"\nOptimal threshold (val F1={best_f1_v:.4f})  → {best_f1_t:.2f}")
    print(f"Optimal threshold (val MCC={best_mcc_v:.4f}) → {best_mcc_t:.2f}")

    # Diagnostics: learned beta per block
    print("\nLearned β (graph-bias weight) per Transformer block:")
    for i, blk in enumerate(model.transformer.h):
        print(f"  block {i}: β={blk.attn.beta.detach().cpu().item():.4f}")

    torch.save(model.state_dict(), args.final_ckpt)
    print(f"\nFinal weights saved → {args.final_ckpt}")

    # Save threshold alongside the best checkpoint
    thresh_path = args.best_ckpt.replace(".pt", "_thresh.npy")
    np.save(thresh_path, {"f1": float(best_f1_t), "mcc": float(best_mcc_t)})
    print(f"Thresholds saved  → {thresh_path}")
    return model
