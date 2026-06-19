"""v13 production — v6 recipe (RNA-FM + FEGS adapter, unfreeze 2, 38-dim biophysics, organism-
balanced sampler) trained on the v13 pool = v5 + de-leaked matched TRAINING pairs. Tests the
data-problem fix: does teaching the free-vs-sequestered discrimination move the HELD-OUT
structure-specificity benchmark (matched-pair acc vs v6's 0.67)? Backbone = RNA-FM (the 3 negatives
showed a fancy backbone isn't the lever). Saves a self-contained checkpoint for the benchmark.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/training/run_v13_production.py
"""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # noqa
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed

OUT = "model/strict_eval_v13_production"
SP = "Data/splits"
TARGET = {0: 0.50, 1: 0.25, 2: 0.25}


def build_pool_v13():
    pos = read_fasta("Data/raw/multispecies/strict_pool_v13_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v13_negatives_all.fasta")
    sneg = read_fasta("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
    pp = list_npz_sorted("Data/processed/fegs_v13_pos"); npn = list_npz_sorted("Data/processed/fegs_v13_neg")
    spn = list_npz_sorted("Data/processed/fegs_struct_neg_v4")
    npos, nneg, nsn = len(pos), len(neg), len(sneg)
    assert len(pp) == npos and len(npn) == nneg and len(spn) == nsn, \
        f"FEGS misalign: {len(pp)}/{npos} {len(npn)}/{nneg} {len(spn)}/{nsn}"
    E.G["seqs"] = [s for _, s in pos] + [s for _, s in neg] + [s for _, s in sneg]
    E.G["hdrs"] = [h for h, _ in pos] + [h for h, _ in neg] + [h for h, _ in sneg]
    E.G["paths"] = list(pp) + list(npn) + list(spn)
    E.G["y"] = np.concatenate([np.ones(npos), np.zeros(nneg), np.zeros(nsn)]).astype(int)
    E.G["bio"] = np.vstack([np.load(f"{SP}/biophys_v13_pos.npy"), np.load(f"{SP}/biophys_v13_neg.npy"),
                            np.load(f"{SP}/biophys_v4_structneg.npy")]).astype(np.float32)
    print(f"v13 pool: {npos} pos / {nneg} real-neg / {nsn} struct-neg = {len(E.G['y'])}", flush=True)


def main():
    set_seed(42); device = setup_device(); os.makedirs(OUT, exist_ok=True)
    build_pool_v13()
    groups = np.load(f"{SP}/cluster_groups_v13.npy"); yeast = np.load(f"{SP}/is_yeast_v13.npy")
    y = E.G["y"]; N = len(y); all_idx = np.arange(N)
    assert len(groups) == N == len(yeast), "split arrays misaligned"
    args = HybridTrainArgs(bio_dim=38, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False,
                           epochs=30, patience=6, lr=1e-4, backbone_lr=5e-6, weight_decay=0.03)
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    meta = json.load(open(f"{SP}/synthetic_train_meta.json"))
    ar = read_fasta(meta["fasta"]); apa = list_npz_sorted(meta["fegs_dir"]); na = min(len(ar), len(apa))
    aug = {"seqs": [s for _, s in ar[:na]], "paths": list(apa[:na]),
           "y": np.array(meta["labels"][:na], dtype=int), "bio": np.load(f"{SP}/biophys_v4_synth.npy").astype(np.float32)[:na]}

    tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=0.10, random_state=7).split(all_idx, y, groups))
    assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx])), "train/val group leak"
    print(f"v13 production: train={len(tr_idx)} (~90% of {N}) val={len(va_idx)}", flush=True)

    seqs_tr = [E.G["seqs"][i] for i in tr_idx] + aug["seqs"]; ptr = [E.G["paths"][i] for i in tr_idx] + aug["paths"]
    ytr = np.concatenate([y[tr_idx], aug["y"]])
    bio_tr_raw = np.vstack([E.G["bio"][tr_idx], aug["bio"]]); m = bio_tr_raw.mean(0); sd = bio_tr_raw.std(0).clip(min=1e-8)
    bio_tr = (bio_tr_raw - m) / sd
    grp = np.empty(len(ytr), int); yk = yeast[tr_idx]
    for i in range(len(tr_idx)): grp[i] = 0 if y[tr_idx][i] == 0 else (1 if yk[i] else 2)
    for j in range(len(aug["y"])): grp[len(tr_idx)+j] = 0 if aug["y"][j] == 0 else 2
    cnt = {g: max(int((grp == g).sum()), 1) for g in TARGET}; w = np.array([TARGET[g]/cnt[g] for g in grp], float)
    collate = make_collate_fn(tok, topk_m=args.topk_m, fp16_bias=args.fp16_bias)
    train_loader = DataLoader(HybridRNADataset(seqs_tr, ptr, ytr, bio_tr, args.max_nucleotides),
                              batch_size=args.batch_size, sampler=WeightedRandomSampler(w, len(w), replacement=True),
                              num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    seqs_va = [E.G["seqs"][i] for i in va_idx]; pva = [E.G["paths"][i] for i in va_idx]; bio_va = (E.G["bio"][va_idx] - m) / sd
    val_loader = DataLoader(HybridRNADataset(seqs_va, pva, y[va_idx], bio_va, args.max_nucleotides),
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = E.init_model(args, device); opt = model.configure_optimizers(args)
    sched = E.make_scheduler(opt, args.epochs * max(1, len(train_loader)), args.warmup_frac)
    best_au, best_state, pat = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for tk, at, Lh, bi, yb in train_loader:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device); yb = yb.to(device); bi = bi.to(device)
            _, loss = model(tk, at, labels=yb, Lhat_stack=Lh, bio_features=bi)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        mv = E.evaluate(model, val_loader, device)
        if mv["auroc"] > best_au + 1e-4:
            best_au = mv["auroc"]; pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        print(f"    [v13-prod] ep{ep+1}/{args.epochs} inner_val={mv['auroc']:.4f} (best={max(best_au,0):.4f} pat {pat}/{args.patience})", flush=True)
        if pat >= args.patience:
            print(f"    [v13-prod] early stop @ ep{ep+1}", flush=True); break
    if best_state is not None: model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUT}/final_model.pt")
    np.savez(f"{OUT}/norm_stats.npz", mean=m, std=sd)
    json.dump({"model": "v13 (v6 recipe + de-leaked matched training pairs)", "backbone": args.backbone,
               "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_total": int(N),
               "best_inner_val_auroc": float(best_au), "bio_dim": 38,
               "purpose": "test if matched training pairs move the held-out structure-specificity benchmark (v6 baseline 0.67 matched-pair acc)"},
              open(f"{OUT}/model_card.json", "w"), indent=2)
    print(f"\n*** v13 model saved -> {OUT}/final_model.pt (best inner-val {best_au:.4f}) ***", flush=True)


if __name__ == "__main__":
    main()
