"""
FINAL ACCEPTED MODEL (v6 production re-fit).

Trains the accepted v6 recipe on 100% of the v5 data — no held-out locked test (it has
served its purpose; performance is already characterized by the leakage-free v6 CV:
overall 0.884, non-yeast 0.798, struct-specificity 0.898). A small cluster-grouped 10%
inner-validation slice is kept only for early-stopping, so the shipped model trains on
~90% of all 2,177 sequences (vs the dev-checkpoint's ~72%).

Recipe = RNA-FM + FEGS adapter (unfreeze last 2) + 38-dim biophysics (incl. self-
complementarity) + 184 structural hard negatives + synthetic aug, CD-HIT cluster grouping,
organism-balanced sampler P(neg)=.5 / P(yeast-pos)=.25 / P(non-yeast-pos)=.25.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python run_v6_production.py
"""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # project path bootstrap (see paths.py)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed
from run_v5_final import build_pool_v5

OUT = "model/strict_eval_v6_production"
SP = "Data/splits"
TARGET = {0: 0.50, 1: 0.25, 2: 0.25}


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

    # 90/10 cluster-grouped split of ALL data — val only for early-stopping
    tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=0.10, random_state=7).split(all_idx, y, groups))
    assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx])), "train/val group leak"
    print(f"production: train={len(tr_idx)} (~90% of {N}) val={len(va_idx)} | "
          f"yeast-pos in train={int((yeast[tr_idx]&(y[tr_idx]==1)).sum())} nonyeast-pos={int((~yeast[tr_idx]&(y[tr_idx]==1)).sum())}",
          flush=True)

    # organism-balanced train loader (train + synth aug)
    seqs_tr = [E.G["seqs"][i] for i in tr_idx] + aug["seqs"]
    ptr = [E.G["paths"][i] for i in tr_idx] + aug["paths"]
    ytr = np.concatenate([y[tr_idx], aug["y"]])
    bio_tr_raw = np.vstack([E.G["bio"][tr_idx], aug["bio"]])
    m = bio_tr_raw.mean(0); sd = bio_tr_raw.std(0).clip(min=1e-8)
    bio_tr = (bio_tr_raw - m) / sd
    grp = np.empty(len(ytr), int); yk = yeast[tr_idx]
    for i in range(len(tr_idx)): grp[i] = 0 if y[tr_idx][i] == 0 else (1 if yk[i] else 2)
    for j in range(len(aug["y"])): grp[len(tr_idx)+j] = 0 if aug["y"][j] == 0 else 2
    cnt = {g: max(int((grp == g).sum()), 1) for g in TARGET}
    w = np.array([TARGET[g]/cnt[g] for g in grp], float)

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
        print(f"    [v6-prod] ep{ep+1}/{args.epochs} inner_val={mv['auroc']:.4f} (best={max(best_au,0):.4f} pat {pat}/{args.patience})", flush=True)
        if pat >= args.patience:
            print(f"    [v6-prod] early stop @ ep{ep+1}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUT}/final_model.pt")
    np.savez(f"{OUT}/norm_stats.npz", mean=m, std=sd)
    json.dump({"model": "v6 FINAL ACCEPTED (organism-balanced, 100% v5 data re-fit)",
               "recipe": "RNA-FM+FEGS adapter (unfreeze 2) + 38-dim biophysics + 184 struct-neg + synth aug, "
                         "CD-HIT cluster grouping, organism-balanced sampler 0.5/0.25/0.25",
               "n_train": int(len(tr_idx)), "n_val_earlystop": int(len(va_idx)), "n_total": int(N),
               "best_inner_val_auroc": float(best_au), "bio_dim": 38,
               "validated_performance_cv": {"overall_auroc": 0.8837, "nonyeast_auroc": 0.7982,
                                            "yeast_auroc": 0.9094, "struct_specificity": 0.8976,
                                            "source": "model/strict_eval_v6_cv (leakage-free 5-fold cluster-grouped CV)"}},
              open(f"{OUT}/model_card.json", "w"), indent=2)
    print(f"\n*** FINAL ACCEPTED MODEL saved -> {OUT}/final_model.pt (best inner-val {best_au:.4f}) ***", flush=True)
    print(f"    Honest performance (v6 CV): overall 0.884 | non-yeast 0.798 | specificity 0.898", flush=True)
    print(f"    Model card -> {OUT}/model_card.json", flush=True)


if __name__ == "__main__":
    main()
