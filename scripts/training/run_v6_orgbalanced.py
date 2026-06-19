"""
v6 experiment: organism-balanced training to attack the yeast-generalization gap
(v5: yeast-pos 0.92 vs non-yeast 0.71 on the locked test).

Identical to v5's final-model recipe (same v5 pool, same cluster-grouped dev/test split,
seeds 999/7, same aug) EXCEPT the training sampler is reweighted so each batch draws
  P(negative) = 0.50,  P(yeast-positive) = 0.25,  P(non-yeast-positive) = 0.25
i.e. the ~268 non-yeast positives are seen as often as the ~1055 yeast positives, so the
model can't coast on the yeast majority. Trains ONE final model on the v5 dev split and
scores the SAME locked test -> directly compares non-yeast AUROC vs v5's 0.714 (overall 0.860).

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python run_v6_orgbalanced.py
"""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # project path bootstrap (see paths.py)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed
from run_v5_final import build_pool_v5, subset_auroc

OUT = "model/strict_eval_v6_orgbalanced"
SP = "Data/splits"


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

    # SAME splits as v5
    dev_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=999).split(all_idx, y, groups))
    f_tr, f_va = next(GroupShuffleSplit(1, test_size=0.15, random_state=7).split(dev_idx, y[dev_idx], groups[dev_idx]))
    f_tr, f_va = dev_idx[f_tr], dev_idx[f_va]

    # ---- build organism-balanced train loader (real f_tr + synth aug) ----
    seqs_tr = [E.G["seqs"][i] for i in f_tr] + aug["seqs"]
    ptr = [E.G["paths"][i] for i in f_tr] + aug["paths"]
    ytr = np.concatenate([y[f_tr], aug["y"]])
    bio_tr_raw = np.vstack([E.G["bio"][f_tr], aug["bio"]])
    m = bio_tr_raw.mean(0); sd = bio_tr_raw.std(0).clip(min=1e-8)
    bio_tr = (bio_tr_raw - m) / sd
    # group id per augmented-train row: 0=neg, 1=yeast-pos, 2=nonyeast-pos (synth pos -> nonyeast)
    grp = np.empty(len(ytr), dtype=int)
    yk = yeast[f_tr]
    for i in range(len(f_tr)):
        grp[i] = 0 if y[f_tr][i] == 0 else (1 if yk[i] else 2)
    for j in range(len(aug["y"])):
        grp[len(f_tr) + j] = 0 if aug["y"][j] == 0 else 2
    target = {0: 0.50, 1: 0.25, 2: 0.25}
    cnt = {g: int((grp == g).sum()) for g in target}
    w = np.array([target[g] / max(cnt[g], 1) for g in grp], dtype=float)
    print(f"train groups: neg={cnt[0]} yeast-pos={cnt[1]} nonyeast-pos={cnt[2]} "
          f"(sampler -> 0.50/0.25/0.25)", flush=True)

    train_ds = HybridRNADataset(seqs_tr, ptr, ytr, bio_tr, args.max_nucleotides)
    collate = make_collate_fn(tok, topk_m=args.topk_m, fp16_bias=args.fp16_bias)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    seqs_va = [E.G["seqs"][i] for i in f_va]; pva = [E.G["paths"][i] for i in f_va]
    bio_va = (E.G["bio"][f_va] - m) / sd
    val_ds = HybridRNADataset(seqs_va, pva, y[f_va], bio_va, args.max_nucleotides)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # ---- train (mirrors train_model loop) ----
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
        print(f"    [v6] ep{ep+1}/{args.epochs} inner_val_AUROC={mv['auroc']:.4f} (best={max(best_au,0):.4f} pat {pat}/{args.patience})", flush=True)
        if pat >= args.patience:
            print(f"    [v6] early stop @ ep{ep+1}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUT}/final_model.pt")
    np.savez(f"{OUT}/norm_stats.npz", mean=m, std=sd)

    # ---- score locked test, compare to v5 ----
    tpr, tlb, thd = E.score_with(model, test_idx, m, sd, args, device, tok)
    tau = roc_auc_score(tlb, tpr); tacc = accuracy_score(tlb, (tpr >= 0.5).astype(int))
    tny, tnyn = subset_auroc(tpr, tlb, ~yeast[test_idx])
    tyy, tyn = subset_auroc(tpr, tlb, yeast[test_idx])
    v5 = json.load(open("model/strict_eval_v5/eval_summary.json"))
    print(f"\n*** v6 ORGANISM-BALANCED LOCKED TEST ***", flush=True)
    print(f"  overall AUROC      = {tau:.4f}   (v5 = {v5['locked_test_auroc']:.4f})")
    print(f"  yeast-pos AUROC    = {tyy}   (n={tyn})")
    print(f"  NON-YEAST AUROC    = {tny}   (n={tnyn})   <- v5 = {v5['locked_test_nonyeast_auroc']:.4f}")
    print(f"  acc@0.5            = {tacc*100:.1f}%")
    json.dump({"locked_test_auroc": float(tau), "locked_test_acc": float(tacc),
               "locked_test_yeast_auroc": tyy, "locked_test_nonyeast_auroc": tny,
               "v5_locked_auroc": v5["locked_test_auroc"], "v5_locked_nonyeast": v5["locked_test_nonyeast_auroc"],
               "sampler": "P(neg)=.5 P(yeast-pos)=.25 P(nonyeast-pos)=.25"},
              open(f"{OUT}/eval_summary.json", "w"), indent=2)
    print(f"Saved -> {OUT}/eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
