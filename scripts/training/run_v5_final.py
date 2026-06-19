"""
RNAPhaseek v5 training on the expanded pool (1352 positives, 3.16x).

Bakes in every lesson:
  - CD-HIT CLUSTER grouping (paralog leak fix) + struct-neg->parent (cluster_groups_v5.npy)
  - leakage-free dev/test + 5-fold CV (seeds 999/42/7)
  - YEAST-HELD-OUT DIAGNOSTIC: reports non-yeast-positive AUROC separately from yeast — the acid
    test for "did it learn RNA-LLPS or learn the yeast transcriptome?" (pool is 67% yeast)
  - struct-negative specificity AUROC carried over from v4
Standard 1022-truncation architecture (tiling/MIL is the deliberate next lever, not here).

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python run_v5_final.py [--skip_cv]
"""
import os, sys, json, argparse
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed

OUT = "model/strict_eval_v5"
SP = "Data/splits"


def build_pool_v5():
    pos = read_fasta("Data/raw/multispecies/strict_pool_v5_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v5_negatives_all.fasta")
    sneg = read_fasta("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
    pp = list_npz_sorted("Data/processed/fegs_v5_pos")
    npn = list_npz_sorted("Data/processed/fegs_v5_neg")
    spn = list_npz_sorted("Data/processed/fegs_struct_neg_v4")
    npos, nneg, nsn = len(pos), len(neg), len(sneg)
    assert (npos, nneg, nsn) == (len(pp), len(npn), len(spn)), \
        f"FEGS mismatch pos {npos}/{len(pp)} neg {nneg}/{len(npn)} sneg {nsn}/{len(spn)}"
    E.G["seqs"]  = [s for _, s in pos] + [s for _, s in neg] + [s for _, s in sneg]
    E.G["hdrs"]  = [h for h, _ in pos] + [h for h, _ in neg] + [h for h, _ in sneg]
    E.G["paths"] = list(pp) + list(npn) + list(spn)
    E.G["y"]     = np.concatenate([np.ones(npos), np.zeros(nneg), np.zeros(nsn)]).astype(int)
    bio = np.vstack([np.load(f"{SP}/biophys_v5_pos.npy"),
                     np.load(f"{SP}/biophys_v5_neg.npy"),
                     np.load(f"{SP}/biophys_v4_structneg.npy")]).astype(np.float32)
    assert bio.shape == (npos + nneg + nsn, 38), bio.shape
    E.G["bio"] = bio
    is_struct = np.array(["hardneg_struct" in h for h in E.G["hdrs"]])
    print(f"v5 pool: {npos} pos / {nneg} real-neg / {nsn} struct-neg = {len(E.G['y'])}", flush=True)
    return is_struct


def subset_auroc(probs, labels, mask_pos, mask_neg=None):
    """AUROC of a positive subset vs negatives."""
    pos = (labels == 1) & mask_pos
    neg = (labels == 0) if mask_neg is None else ((labels == 0) & mask_neg)
    if pos.sum() < 3 or neg.sum() < 3:
        return None, int(pos.sum())
    yy = np.concatenate([np.ones(int(pos.sum())), np.zeros(int(neg.sum()))])
    pp = np.concatenate([probs[pos], probs[neg]])
    return float(roc_auc_score(yy, pp)), int(pos.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip_cv", action="store_true")
    a = ap.parse_args()
    set_seed(42); device = setup_device(); os.makedirs(OUT, exist_ok=True)
    is_struct = build_pool_v5()
    groups = np.load(f"{SP}/cluster_groups_v5.npy")
    yeast = np.load(f"{SP}/is_yeast_v5.npy")
    y = E.G["y"]; N = len(y); all_idx = np.arange(N)
    assert len(groups) == N and len(yeast) == N

    args = HybridTrainArgs(bio_dim=38, use_species_embed=False, unfreeze_last_n=2,
                           freeze_backbone=False, epochs=30, patience=6,
                           lr=1e-4, backbone_lr=5e-6, weight_decay=0.03)
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    meta = json.load(open(f"{SP}/synthetic_train_meta.json"))
    ar = read_fasta(meta["fasta"]); apa = list_npz_sorted(meta["fegs_dir"]); na = min(len(ar), len(apa))
    E.G["aug"] = {"seqs": [s for _, s in ar[:na]], "paths": list(apa[:na]),
                  "y": np.array(meta["labels"][:na], dtype=int),
                  "bio": np.load(f"{SP}/biophys_v4_synth.npy").astype(np.float32)[:na]}

    dev_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=999).split(all_idx, y, groups))
    assert set(groups[dev_idx]).isdisjoint(set(groups[test_idx])), "dev/test leak"
    print(f"N={N} dev={len(dev_idx)} test={len(test_idx)} | groups={len(set(groups.tolist()))} "
          f"| yeast-pos={int((yeast&(y==1)).sum())} nonyeast-pos={int((~yeast&(y==1)).sum())}", flush=True)

    if not a.skip_cv:
        prog_path = f"{OUT}/cv_progress.json"
        prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"folds": {}}
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        oof = np.full(N, np.nan)
        for k, (otr, ova) in enumerate(sgkf.split(dev_idx, y[dev_idx], groups[dev_idx])):
            tr, va = dev_idx[otr], dev_idx[ova]
            if str(k) in prog["folds"]:
                oof[va] = np.array(prog["folds"][str(k)]["probs"]); print(f"  [fold {k}] cached", flush=True); continue
            assert set(groups[tr]).isdisjoint(set(groups[va])), f"fold {k} leak"
            model, m, sd_, _ = E.train_model(tr, va, args, device, tok, tag=f"v5cv{k}")
            pr, lb, _ = E.score_with(model, va, m, sd_, args, device, tok)
            oof[va] = pr; prog["folds"][str(k)] = {"auroc": float(roc_auc_score(lb, pr)),
                                                    "probs": pr.tolist(), "va_idx": va.tolist()}
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
        print(f"\n*** v5 CLUSTER-GROUPED CV ***", flush=True)
        print(f"  overall OOF AUROC        = {oau:.4f}   fold mean {np.mean(cv):.4f}±{np.std(cv):.4f}")
        print(f"  pos-vs-STRUCT AUROC      = {sau}   (specificity, n={nsn})")
        print(f"  YEAST-pos vs neg AUROC   = {yau}   (n_pos={nyp})")
        print(f"  NON-YEAST-pos vs neg     = {nyau}   (n_pos={nnyp})  <- organism generalization acid-test")
    else:
        oau = sau = yau = nyau = None; cv = []; nsn = nyp = nnyp = 0

    f_tr, f_va = next(GroupShuffleSplit(1, test_size=0.15, random_state=7).split(dev_idx, y[dev_idx], groups[dev_idx]))
    f_tr, f_va = dev_idx[f_tr], dev_idx[f_va]
    model, fm, fsd, _ = E.train_model(f_tr, f_va, args, device, tok, tag="v5final")
    torch.save(model.state_dict(), f"{OUT}/final_model.pt")
    np.savez(f"{OUT}/norm_stats.npz", mean=fm, std=fsd)
    tpr, tlb, thd = E.score_with(model, test_idx, fm, fsd, args, device, tok)
    tau = roc_auc_score(tlb, tpr); tacc = accuracy_score(tlb, (tpr >= 0.5).astype(int))
    tny, tnyn = subset_auroc(tpr, tlb, ~yeast[test_idx])
    print(f"\n*** v5 LOCKED-TEST AUROC={tau:.4f} acc={tacc*100:.1f}% (n={len(tlb)}) | non-yeast-pos AUROC={tny} (n={tnyn}) ***", flush=True)
    E.diagnostic(tpr, tlb, thd, "v5 locked test")
    json.dump({"cv_oof_auroc": oau, "cv_struct_auroc": sau, "cv_yeast_auroc": yau,
               "cv_nonyeast_auroc": nyau, "cv_fold_scores": cv, "cv_nonyeast_n": nnyp,
               "locked_test_auroc": float(tau), "locked_test_acc": float(tacc),
               "locked_test_nonyeast_auroc": tny, "n_test": int(len(tlb)), "n_dev": int(len(dev_idx)),
               "pool": "v5_expanded_3.16x", "n_pos": int((y == 1).sum()), "n_groups": int(len(set(groups.tolist())))},
              open(f"{OUT}/eval_summary.json", "w"), indent=2)
    print(f"Saved -> {OUT}/eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
