"""
RNAPhaseek Hybrid — Rigorous evaluation protocol on the strict pool.

Clean 3-way separation (no leakage):
  1. LOCK a 15% stratified TEST set up front  -> never used in training/stopping.
  2. Leakage-free 5-fold CV on the remaining 85% (dev):
       each fold carves an INNER validation from its training portion for
       early-stopping, and scores its held-out outer fold ONCE at the
       best-inner-val state  (the outer fold never drives stopping).
  3. Train ONE final model on all dev (internal val for stopping), evaluate
       ONCE on the locked test -> headline number.
  4. Easy/hard separability diagnostic on the locked test.

Config: 33-dim features (absolute repeat), unfreeze last-2 RNA-FM layers,
warm-start from Phase-1.

Run:
  python -m Functions.RNAPhaseek.RNAPhaseek_hybrid_eval
"""
import argparse, json, os, sys
import numpy as np
import torch
import multimolecule  # noqa
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from Functions.RNAPhaseek.RNAPhaseek_hybrid         import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config  import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data    import (read_fasta, make_dataloaders,
                                                            HybridRNADataset, make_collate_fn)
from Functions.RNAPhaseek.RNAPhaseek_hybrid_trainer import evaluate, make_scheduler
from Functions.RNAPhaseek.RNAPhaseek_utils          import list_npz_sorted, setup_device, set_seed
from torch.utils.data import DataLoader

FASTA_POS = "Data/raw/multispecies/strict_pool_positives.fasta"
FASTA_NEG = "Data/raw/multispecies/strict_pool_negatives_all.fasta"
SRC_POS   = "Data/processed/fegs_topk_strict_pos"
SRC_NEG   = "Data/processed/fegs_topk_strict_neg"
BIO_POS   = "Data/splits/biophys_strict_pos.npy"
BIO_NEG   = "Data/splits/biophys_strict_neg.npy"
INIT_FROM = "model/phase1/hybrid_best.pt"
OUT_DIR   = "model/strict_eval"
TEST_LOCK = "Data/splits/strict_locked_test.json"

G = {}  # global data holder


def neg_subtype(h):
    hl = h.lower()
    if "hardneg_struct" in hl: return "hard:struct"      # v4 self-complementarity-destroyed
    if "hardneg_terra" in hl: return "hard:TERRA"
    if "hardneg_shuffle" in hl: return "hard:shuffle"
    if "hardneg_" in hl: return "hard:subthreshold"
    return "matched"


def init_model(args, device):
    model = RNAFMHybridClassifier(args).to(device)
    if os.path.exists(INIT_FROM):
        st = torch.load(INIT_FROM, map_location=device, weights_only=True)
        sd = st["model"] if isinstance(st, dict) and "model" in st else st
        msd = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(keep, strict=False)
    return model


def train_model(tr_idx, va_idx, args, device, tokenizer, tag=""):
    """Train with early-stopping on va_idx. Returns (model with best state, m, sd).
    If G['aug'] is set, the synthetic augmentation is appended to the TRAIN side
    only (never to va_idx / test / outer folds)."""
    seqs_tr = [G["seqs"][i] for i in tr_idx]; seqs_va = [G["seqs"][i] for i in va_idx]
    ptr = [G["paths"][i] for i in tr_idx];   pva = [G["paths"][i] for i in va_idx]
    ytr, yva = G["y"][tr_idx], G["y"][va_idx]
    bio_tr_raw = G["bio"][tr_idx]
    if G.get("aug"):
        a = G["aug"]
        seqs_tr = seqs_tr + a["seqs"]
        ptr     = ptr + a["paths"]
        ytr     = np.concatenate([ytr, a["y"]])
        bio_tr_raw = np.vstack([bio_tr_raw, a["bio"]])
    m = bio_tr_raw.mean(0); sd = bio_tr_raw.std(0).clip(min=1e-8)
    bio_tr = (bio_tr_raw - m) / sd; bio_va = (G["bio"][va_idx] - m) / sd

    train_loader, val_loader = make_dataloaders(seqs_tr, seqs_va, ptr, pva, ytr, yva,
                                                tokenizer, args, bio_tr=bio_tr, bio_va=bio_va)
    model = init_model(args, device)
    opt = model.configure_optimizers(args)
    sched = make_scheduler(opt, args.epochs * max(1, len(train_loader)), args.warmup_frac)

    best_au, best_state, pat = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for token_ids, att, Lhat, biob, yb in train_loader:
            token_ids = token_ids.to(device); att = att.to(device); Lhat = Lhat.to(device)
            yb = yb.to(device); biob = biob.to(device) if biob is not None else None
            _, loss = model(token_ids, att, labels=yb, Lhat_stack=Lhat, bio_features=biob)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        mv = evaluate(model, val_loader, device)
        if mv["auroc"] > best_au + 1e-4:
            best_au = mv["auroc"]; pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        print(f"    [{tag}] ep{ep+1}/{args.epochs} inner_val_AUROC={mv['auroc']:.4f} "
              f"(best={max(best_au,0):.4f} pat {pat}/{args.patience})", flush=True)
        if pat >= args.patience:
            print(f"    [{tag}] early stop @ ep{ep+1}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, m, sd, best_au


@torch.no_grad()
def score_with(model, idx, m, sd, args, device, tokenizer):
    """Inference on idx using normalisation (m, sd). Returns probs, labels, headers."""
    model.eval()
    seqs = [G["seqs"][i] for i in idx]; paths = [G["paths"][i] for i in idx]
    ds = HybridRNADataset(seqs, paths, G["y"][idx], (G["bio"][idx] - m) / sd, args.max_nucleotides)
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=make_collate_fn(tokenizer, topk_m=args.topk_m))
    probs = []
    for token_ids, att, Lhat, biob, _ in loader:
        token_ids = token_ids.to(device); att = att.to(device); Lhat = Lhat.to(device)
        biob = biob.to(device) if biob is not None else None
        logits, _ = model(token_ids, att, labels=None, Lhat_stack=Lhat, bio_features=biob)
        fin = torch.isfinite(logits).all(-1, keepdim=True)
        logits = torch.where(fin, logits, torch.zeros_like(logits))
        probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs), G["y"][idx], [G["hdrs"][i] for i in idx]


def diagnostic(probs, labels, hdrs, title):
    sub = np.array([neg_subtype(h) if l == 0 else "positive" for h, l in zip(hdrs, labels)])
    print(f"\n  --- {title} diagnostic ---")
    easy = (sub == "positive") | (sub == "matched")
    if (labels[easy] == 1).sum() and (labels[easy] == 0).sum():
        print(f"    EASY (pos vs matched)  AUROC = {roc_auc_score(labels[easy], probs[easy]):.4f}")
    posm = labels == 1
    for c in ["matched", "hard:struct", "hard:subthreshold", "hard:shuffle", "hard:TERRA"]:
        msk = sub == c
        if msk.sum() < 3:
            print(f"    pos vs {c:<18} n={int(msk.sum())} (too few)"); continue
        yy = np.concatenate([np.ones(int(posm.sum())), np.zeros(int(msk.sum()))])
        pp = np.concatenate([probs[posm], probs[msk]])
        print(f"    pos vs {c:<18} n={int(msk.sum()):<4} AUROC = {roc_auc_score(yy, pp):.4f}")


def main():
    global FASTA_POS, FASTA_NEG, SRC_POS, SRC_NEG, BIO_POS, BIO_NEG, TEST_LOCK, OUT_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--unfreeze_last_n", type=int, default=2)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--inner_val_frac", type=float, default=0.15)
    p.add_argument("--augment", action="store_true",
                   help="Append synthetic repeat augmentation to every training fold.")
    p.add_argument("--out_suffix", type=str, default="",
                   help="Suffix for the output dir (e.g. '_aug') to avoid overwriting.")
    p.add_argument("--fasta_pos", default=FASTA_POS)
    p.add_argument("--fasta_neg", default=FASTA_NEG)
    p.add_argument("--src_pos",   default=SRC_POS)
    p.add_argument("--src_neg",   default=SRC_NEG)
    p.add_argument("--bio_pos",   default=BIO_POS)
    p.add_argument("--bio_neg",   default=BIO_NEG)
    p.add_argument("--test_lock", default=TEST_LOCK)
    a = p.parse_args()

    FASTA_POS, FASTA_NEG = a.fasta_pos, a.fasta_neg
    SRC_POS, SRC_NEG = a.src_pos, a.src_neg
    BIO_POS, BIO_NEG = a.bio_pos, a.bio_neg
    TEST_LOCK = a.test_lock
    OUT_DIR = OUT_DIR + a.out_suffix
    set_seed(42); device = setup_device(); os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load ──
    pos = read_fasta(FASTA_POS); neg = read_fasta(FASTA_NEG)
    pp = list_npz_sorted(SRC_POS); npn = list_npz_sorted(SRC_NEG)
    npos = min(len(pos), len(pp)); nneg = min(len(neg), len(npn))
    G["seqs"] = [s for _, s in pos[:npos]] + [s for _, s in neg[:nneg]]
    G["hdrs"] = [h for h, _ in pos[:npos]] + [h for h, _ in neg[:nneg]]
    G["paths"] = list(pp[:npos]) + list(npn[:nneg])
    G["y"] = np.concatenate([np.ones(npos), np.zeros(nneg)]).astype(int)
    G["bio"] = np.vstack([np.load(BIO_POS), np.load(BIO_NEG)]).astype(np.float32)
    N = len(G["y"])
    print(f"Loaded {N}  pos={npos} neg={nneg}  bio_dim={G['bio'].shape[1]}", flush=True)

    args = HybridTrainArgs(bio_dim=G["bio"].shape[1], use_species_embed=False,
                           unfreeze_last_n=a.unfreeze_last_n,
                           freeze_backbone=(a.unfreeze_last_n == 0),
                           epochs=a.epochs, patience=a.patience,
                           lr=1e-4, backbone_lr=5e-6, weight_decay=0.03)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

    # ── Optional synthetic augmentation (TRAIN-only) ──
    if a.augment:
        meta = json.load(open("Data/splits/synthetic_train_meta.json"))
        aug_recs = read_fasta(meta["fasta"])
        aug_paths = list_npz_sorted(meta["fegs_dir"])
        n_aug = min(len(aug_recs), len(aug_paths))
        G["aug"] = {
            "seqs":  [s for _, s in aug_recs[:n_aug]],
            "paths": list(aug_paths[:n_aug]),
            "y":     np.array(meta["labels"][:n_aug], dtype=int),
            "bio":   np.load(meta["bio"]).astype(np.float32)[:n_aug],
        }
        print(f"AUGMENTATION ON: +{n_aug} synthetic repeats per training fold "
              f"(pos={int(G['aug']['y'].sum())} neg={int((G['aug']['y']==0).sum())})", flush=True)

    # ── 1. LOCK a stratified test set (random_state=999, matching project convention) ──
    all_idx = np.arange(N)
    dev_idx, test_idx = train_test_split(all_idx, test_size=a.test_frac,
                                         random_state=999, stratify=G["y"])
    json.dump({"test_headers": [G["hdrs"][i] for i in test_idx.tolist()],
               "n_test": len(test_idx), "n_test_pos": int(G["y"][test_idx].sum()),
               "n_test_neg": int((G["y"][test_idx] == 0).sum()),
               "random_state": 999, "test_frac": a.test_frac},
              open(TEST_LOCK, "w"), indent=2)
    print(f"\nLOCKED TEST: n={len(test_idx)} (pos={int(G['y'][test_idx].sum())} "
          f"neg={int((G['y'][test_idx]==0).sum())})  ->  {TEST_LOCK}")
    print(f"DEV (for CV + final): n={len(dev_idx)}", flush=True)

    # ── 2. Leakage-free 5-fold CV on dev ──
    print("\n===== LEAKAGE-FREE 5-FOLD CV (on dev only) =====", flush=True)
    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=42)
    fold_scores, oof_probs, oof_labels, oof_hdrs = [], [], [], []
    for fold, (otr_rel, ote_rel) in enumerate(skf.split(dev_idx, G["y"][dev_idx])):
        otr = dev_idx[otr_rel]; ote = dev_idx[ote_rel]
        # inner val from outer-train (for early stopping)
        itr, iva = train_test_split(otr, test_size=a.inner_val_frac, random_state=42,
                                    stratify=G["y"][otr])
        print(f"\n  FOLD {fold}: inner_train={len(itr)} inner_val={len(iva)} outer_test={len(ote)}", flush=True)
        model, m, sd, _ = train_model(itr, iva, args, device, tokenizer, tag=f"f{fold}")
        pr, lb, hd = score_with(model, ote, m, sd, args, device, tokenizer)
        au = roc_auc_score(lb, pr)
        fold_scores.append(au); oof_probs.append(pr); oof_labels.append(lb); oof_hdrs += hd
        print(f"  FOLD {fold} clean outer-test AUROC = {au:.4f}", flush=True)
        json.dump({"fold_scores": fold_scores}, open(f"{OUT_DIR}/cv_progress.json", "w"))
        del model
        if device == "mps":
            torch.mps.empty_cache()

    cv_mean, cv_std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    P = np.concatenate(oof_probs); L = np.concatenate(oof_labels)
    print("\n" + "=" * 56)
    print(f"CV (leakage-free) AUROC = {cv_mean:.4f} ± {cv_std:.4f}   folds={[f'{s:.3f}' for s in fold_scores]}")
    print(f"Pooled out-of-fold AUROC = {roc_auc_score(L, P):.4f}  PR-AUC = {average_precision_score(L, P):.4f}")
    diagnostic(P, L, oof_hdrs, "Pooled OOF")

    # ── 3. Final model on ALL dev, evaluate ONCE on locked test ──
    print("\n===== FINAL MODEL (train on all dev, test on locked test) =====", flush=True)
    f_tr, f_va = train_test_split(dev_idx, test_size=a.inner_val_frac, random_state=7,
                                  stratify=G["y"][dev_idx])
    fmodel, fm, fsd, _ = train_model(f_tr, f_va, args, device, tokenizer, tag="final")
    torch.save(fmodel.state_dict(), f"{OUT_DIR}/final_model.pt")
    tpr, tlb, thd = score_with(fmodel, test_idx, fm, fsd, args, device, tokenizer)
    test_au = roc_auc_score(tlb, tpr); test_pr = average_precision_score(tlb, tpr)
    print(f"\n  *** LOCKED-TEST AUROC = {test_au:.4f}  PR-AUC = {test_pr:.4f}  (n={len(tlb)}) ***")
    diagnostic(tpr, tlb, thd, "Locked test")

    json.dump({
        "cv_mean_auroc": cv_mean, "cv_std_auroc": cv_std, "cv_fold_scores": fold_scores,
        "pooled_oof_auroc": float(roc_auc_score(L, P)),
        "locked_test_auroc": float(test_au), "locked_test_prauc": float(test_pr),
        "n_test": int(len(tlb)), "n_dev": int(len(dev_idx)),
        "config": {"unfreeze_last_n": a.unfreeze_last_n, "bio_dim": int(G["bio"].shape[1]),
                   "epochs": a.epochs, "patience": a.patience},
    }, open(f"{OUT_DIR}/eval_summary.json", "w"), indent=2)
    print(f"\nSaved -> {OUT_DIR}/eval_summary.json")


if __name__ == "__main__":
    main()
