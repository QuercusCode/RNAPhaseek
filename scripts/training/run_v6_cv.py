"""
v6 organism-balanced training as a full 5-fold cluster-grouped CV (same pool, same
cluster groups, same folds/seeds as v5) so the non-yeast OOF AUROC has n=268 power —
the robust confirmation the n=29 locked test couldn't give.

Each fold trains with the sampler P(neg)=.5 / P(yeast-pos)=.25 / P(non-yeast-pos)=.25.
Reports pooled-OOF overall / yeast / NON-yeast / struct AUROC vs v5's 0.883 / 0.919 / 0.763.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python run_v6_cv.py
"""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # project path bootstrap (see paths.py)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed
from run_v5_final import build_pool_v5, subset_auroc

OUT = "model/strict_eval_v6_cv"
SP = "Data/splits"
TARGET = {0: 0.50, 1: 0.25, 2: 0.25}  # neg / yeast-pos / nonyeast-pos


def train_orgbalanced(tr_idx, va_idx, aug, yeast, args, device, tok, tag):
    y = E.G["y"]
    seqs_tr = [E.G["seqs"][i] for i in tr_idx] + aug["seqs"]
    ptr = [E.G["paths"][i] for i in tr_idx] + aug["paths"]
    ytr = np.concatenate([y[tr_idx], aug["y"]])
    bio_tr_raw = np.vstack([E.G["bio"][tr_idx], aug["bio"]])
    m = bio_tr_raw.mean(0); sd = bio_tr_raw.std(0).clip(min=1e-8)
    bio_tr = (bio_tr_raw - m) / sd
    grp = np.empty(len(ytr), dtype=int); yk = yeast[tr_idx]
    for i in range(len(tr_idx)):
        grp[i] = 0 if y[tr_idx][i] == 0 else (1 if yk[i] else 2)
    for j in range(len(aug["y"])):
        grp[len(tr_idx) + j] = 0 if aug["y"][j] == 0 else 2
    cnt = {g: max(int((grp == g).sum()), 1) for g in TARGET}
    w = np.array([TARGET[g] / cnt[g] for g in grp], dtype=float)

    collate = make_collate_fn(tok, topk_m=args.topk_m, fp16_bias=args.fp16_bias)
    train_ds = HybridRNADataset(seqs_tr, ptr, ytr, bio_tr, args.max_nucleotides)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    seqs_va = [E.G["seqs"][i] for i in va_idx]; pva = [E.G["paths"][i] for i in va_idx]
    bio_va = (E.G["bio"][va_idx] - m) / sd
    val_loader = DataLoader(HybridRNADataset(seqs_va, pva, y[va_idx], bio_va, args.max_nucleotides),
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = E.init_model(args, device); opt = model.configure_optimizers(args)
    sched = E.make_scheduler(opt, args.epochs * max(1, len(train_loader)), args.warmup_frac)
    best_au, best_state, pat = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for tk, at, Lh, bi, yb in train_loader:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device)
            yb = yb.to(device); bi = bi.to(device) if bi is not None else None
            _, loss = model(tk, at, labels=yb, Lhat_stack=Lh, bio_features=bi)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        mv = E.evaluate(model, val_loader, device)
        if mv["auroc"] > best_au + 1e-4:
            best_au = mv["auroc"]; pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        print(f"    [{tag}] ep{ep+1}/{args.epochs} val={mv['auroc']:.4f} (best={max(best_au,0):.4f} pat {pat}/{args.patience})", flush=True)
        if pat >= args.patience:
            print(f"    [{tag}] early stop @ ep{ep+1}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, m, sd


def main():
    set_seed(42); device = setup_device(); os.makedirs(OUT, exist_ok=True)
    is_struct = build_pool_v5()
    groups = np.load(f"{SP}/cluster_groups_v5.npy"); yeast = np.load(f"{SP}/is_yeast_v5.npy")
    y = E.G["y"]; N = len(y); all_idx = np.arange(N)
    args = HybridTrainArgs(bio_dim=38, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False,
                           epochs=30, patience=6, lr=1e-4, backbone_lr=5e-6, weight_decay=0.03)
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    meta = json.load(open(f"{SP}/synthetic_train_meta.json"))
    ar = read_fasta(meta["fasta"]); apa = list_npz_sorted(meta["fegs_dir"]); na = min(len(ar), len(apa))
    aug = {"seqs": [s for _, s in ar[:na]], "paths": list(apa[:na]),
           "y": np.array(meta["labels"][:na], dtype=int), "bio": np.load(f"{SP}/biophys_v4_synth.npy").astype(np.float32)[:na]}

    dev_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=999).split(all_idx, y, groups))
    prog_path = f"{OUT}/cv_progress.json"
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"folds": {}}
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    oof = np.full(N, np.nan)
    for k, (otr, ova) in enumerate(sgkf.split(dev_idx, y[dev_idx], groups[dev_idx])):
        tr, va = dev_idx[otr], dev_idx[ova]
        if str(k) in prog["folds"]:
            oof[va] = np.array(prog["folds"][str(k)]["probs"]); print(f"  [fold {k}] cached", flush=True); continue
        assert set(groups[tr]).isdisjoint(set(groups[va])), f"fold {k} leak"
        model, m, sd = train_orgbalanced(tr, va, aug, yeast, args, device, tok, f"v6cv{k}")
        pr, lb, _ = E.score_with(model, va, m, sd, args, device, tok)
        oof[va] = pr; prog["folds"][str(k)] = {"auroc": float(roc_auc_score(lb, pr)), "probs": pr.tolist(), "va_idx": va.tolist()}
        json.dump(prog, open(prog_path, "w"))
        print(f"  [fold {k}] val AUROC={prog['folds'][str(k)]['auroc']:.4f} (n={len(lb)})", flush=True)
        del model
        if hasattr(torch, "mps"): torch.mps.empty_cache()

    dm = ~np.isnan(oof) & np.isin(all_idx, dev_idx)
    oau = roc_auc_score(y[dm], oof[dm])
    sau, nsn = subset_auroc(oof[dm], y[dm], is_struct[dm] | True, mask_neg=is_struct[dm])
    yau, nyp = subset_auroc(oof[dm], y[dm], yeast[dm])
    nyau, nnyp = subset_auroc(oof[dm], y[dm], ~yeast[dm])
    cv = [prog["folds"][str(k)]["auroc"] for k in range(5) if str(k) in prog["folds"]]
    v5 = json.load(open("model/strict_eval_v5/eval_summary.json"))
    print(f"\n*** v6 ORGANISM-BALANCED CV (confirmation, n_nonyeast={nnyp}) ***", flush=True)
    print(f"  overall OOF AUROC   = {oau:.4f}   (v5 = {v5['cv_oof_auroc']:.4f})")
    print(f"  pos-vs-STRUCT       = {sau:.4f}   (v5 = {v5['cv_struct_auroc']:.4f})")
    print(f"  YEAST-pos AUROC     = {yau:.4f}   (v5 = {v5['cv_yeast_auroc']:.4f})   n={nyp}")
    print(f"  NON-YEAST AUROC     = {nyau:.4f}   (v5 = {v5['cv_nonyeast_auroc']:.4f})   n={nnyp}  <- the robust verdict")
    json.dump({"cv_oof_auroc": float(oau), "cv_struct_auroc": sau, "cv_yeast_auroc": yau,
               "cv_nonyeast_auroc": nyau, "cv_nonyeast_n": nnyp, "cv_fold_scores": cv,
               "v5_cv_oof": v5["cv_oof_auroc"], "v5_cv_yeast": v5["cv_yeast_auroc"], "v5_cv_nonyeast": v5["cv_nonyeast_auroc"],
               "sampler": "P(neg)=.5 P(yeast-pos)=.25 P(nonyeast-pos)=.25"},
              open(f"{OUT}/eval_summary.json", "w"), indent=2)
    print(f"Saved -> {OUT}/eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
