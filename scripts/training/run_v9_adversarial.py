"""v9 — domain-adversarial organism invariance (item 1). Same protocol as run_v6_cv
(5-fold cluster-grouped, organism-balanced sampler, locked-test held out, resumable) but adds
a gradient-reversal organism head: the shared representation is pushed to be yeast/non-yeast
INVARIANT, forcing reliance on the LLPS mechanism. Goal: lift NON-YEAST OOF AUROC above v6's
0.798 without hurting overall (v6 0.884). GRL strength ramps 0->1 over the first epochs.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/training/run_v9_adversarial.py
"""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # noqa
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed
from run_v5_final import build_pool_v5, subset_auroc

OUT = "model/strict_eval_v9_adv"
SP = "Data/splits"
TARGET = {0: 0.50, 1: 0.25, 2: 0.25}
RAMP_EPOCHS = 5


class AdvDataset(HybridRNADataset):
    """Adds a per-sample organism label (yeast=1 / non-yeast=0) to the standard 4-tuple."""
    def __init__(self, seqs, paths, labels, bio, organism, max_nt):
        super().__init__(seqs, paths, labels, bio, max_nt)
        self.organism = np.asarray(organism, dtype=np.int64)

    def __getitem__(self, i):
        return (*super().__getitem__(i), int(self.organism[i]))


def make_adv_collate(tok, topk_m):
    """Wrap the standard collate: strip organism, run base collate, re-attach organism."""
    base = make_collate_fn(tok, topk_m=topk_m)
    def collate(batch):
        org = torch.tensor([b[-1] for b in batch], dtype=torch.long)
        tk, at, Lh, bi, yb = base([b[:4] for b in batch])
        return tk, at, Lh, bi, yb, org
    return collate


def train_adv(tr_idx, va_idx, aug, yeast, args, device, tok, tag):
    y = E.G["y"]
    seqs_tr = [E.G["seqs"][i] for i in tr_idx] + aug["seqs"]
    ptr = [E.G["paths"][i] for i in tr_idx] + aug["paths"]
    ytr = np.concatenate([y[tr_idx], aug["y"]])
    org_tr = np.concatenate([yeast[tr_idx].astype(int), np.zeros(len(aug["y"]), int)])  # yeast=1 / non-yeast=0
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

    collate = make_adv_collate(tok, args.topk_m)
    train_ds = AdvDataset(seqs_tr, ptr, ytr, bio_tr, org_tr, args.max_nucleotides)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    seqs_va = [E.G["seqs"][i] for i in va_idx]; pva = [E.G["paths"][i] for i in va_idx]
    bio_va = (E.G["bio"][va_idx] - m) / sd
    val_loader = DataLoader(HybridRNADataset(seqs_va, pva, y[va_idx], bio_va, args.max_nucleotides),
                            batch_size=args.batch_size, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=args.topk_m))
    model = E.init_model(args, device); opt = model.configure_optimizers(args)
    sched = E.make_scheduler(opt, args.epochs * max(1, len(train_loader)), args.warmup_frac)
    best_au, best_state, pat = -1.0, None, 0
    for ep in range(args.epochs):
        model.adv_lambda = min(1.0, (ep + 1) / RAMP_EPOCHS) * args.adv_lambda   # ramp GRL strength
        model.train()
        for tk, at, Lh, bi, yb, org in train_loader:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device); yb = yb.to(device)
            bi = bi.to(device) if bi is not None else None; org = org.to(device)
            _, loss = model(tk, at, labels=yb, Lhat_stack=Lh, bio_features=bi, organism_labels=org)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        mv = E.evaluate(model, val_loader, device)
        if mv["auroc"] > best_au + 1e-4:
            best_au = mv["auroc"]; pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        print(f"    [{tag}] ep{ep+1}/{args.epochs} val={mv['auroc']:.4f} "
              f"(best={max(best_au,0):.4f} pat {pat}/{args.patience} lam={model.adv_lambda:.2f})", flush=True)
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
                           epochs=30, patience=6, lr=1e-4, backbone_lr=5e-6, weight_decay=0.03,
                           adv_organism=True, adv_lambda=1.0, n_organisms=2)
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
        model, m, sd = train_adv(tr, va, aug, yeast, args, device, tok, f"v9adv{k}")
        pr, lb, _ = E.score_with(model, va, m, sd, args, device, tok)
        oof[va] = pr; prog["folds"][str(k)] = {"auroc": float(roc_auc_score(lb, pr)), "probs": pr.tolist(), "va_idx": va.tolist()}
        json.dump(prog, open(prog_path, "w"))
        print(f"  [fold {k}] val AUROC={prog['folds'][str(k)]['auroc']:.4f} (n={len(lb)})  (v6 folds 0.880-0.892)", flush=True)
        del model
        if hasattr(torch, "mps"): torch.mps.empty_cache()

    dm = ~np.isnan(oof) & np.isin(all_idx, dev_idx)
    oau = roc_auc_score(y[dm], oof[dm])
    sau, _ = subset_auroc(oof[dm], y[dm], is_struct[dm] | True, mask_neg=is_struct[dm])
    yau, nyp = subset_auroc(oof[dm], y[dm], yeast[dm])
    nyau, nnyp = subset_auroc(oof[dm], y[dm], ~yeast[dm])
    cv = [prog["folds"][str(k)]["auroc"] for k in range(5) if str(k) in prog["folds"]]
    v6 = json.load(open("model/strict_eval_v6_cv/eval_summary.json"))
    print(f"\n*** v9 DOMAIN-ADVERSARIAL CV (organism-invariance) ***", flush=True)
    print(f"  overall OOF = {oau:.4f}   (v6 = {v6['cv_oof_auroc']:.4f})")
    print(f"  STRUCT-spec = {sau:.4f}   (v6 = {v6['cv_struct_auroc']:.4f})")
    print(f"  YEAST       = {yau:.4f}   (v6 = {v6['cv_yeast_auroc']:.4f})   n={nyp}")
    print(f"  NON-YEAST   = {nyau:.4f}   (v6 = {v6['cv_nonyeast_auroc']:.4f})   n={nnyp}   <- the target")
    json.dump({"cv_oof_auroc": float(oau), "cv_struct_auroc": sau, "cv_yeast_auroc": yau,
               "cv_nonyeast_auroc": nyau, "cv_nonyeast_n": nnyp, "cv_fold_scores": cv,
               "v6_cv_nonyeast": v6["cv_nonyeast_auroc"], "v6_cv_oof": v6["cv_oof_auroc"],
               "method": "DANN organism-invariance, GRL lambda ramp 0->1"},
              open(f"{OUT}/eval_summary.json", "w"), indent=2)
    print(f"Saved -> {OUT}/eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
