"""
Honest cluster-grouped re-baseline. Identical to run_v4_final's CV (same seeds 999/42,
same pool, same augmentation, same args) EXCEPT the group key is swapped from per-positive
to CD-HIT-cluster groups (Data/splits/cluster_groups_v4.npy) — so near-duplicate positive
paralogs/fragments can no longer straddle folds. This isolates the paralog-leakage effect
on the headline 0.8474 CV AUROC. CV only (the honest stability estimate); checkpointed.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python run_v4_clustercv.py
"""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # project path bootstrap (see paths.py)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed
from run_v4_final import build_pool, struct_auroc

OUT = "model/strict_eval_v4_clustercv"
SP = "Data/splits"


def main():
    set_seed(42); device = setup_device(); os.makedirs(OUT, exist_ok=True)
    _per_pos_groups, is_struct = build_pool()          # fills E.G; ignore the per-positive groups
    groups = np.load(f"{SP}/cluster_groups_v4.npy")    # cluster groups instead
    y = E.G["y"]; N = len(y)
    assert len(groups) == N, f"{len(groups)} != {N}"

    args = HybridTrainArgs(bio_dim=38, use_species_embed=False, unfreeze_last_n=2,
                           freeze_backbone=False, epochs=30, patience=6,
                           lr=1e-4, backbone_lr=5e-6, weight_decay=0.03)
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    meta = json.load(open(f"{SP}/synthetic_train_meta.json"))
    ar = read_fasta(meta["fasta"]); apa = list_npz_sorted(meta["fegs_dir"]); na = min(len(ar), len(apa))
    E.G["aug"] = {"seqs": [s for _, s in ar[:na]], "paths": list(apa[:na]),
                  "y": np.array(meta["labels"][:na], dtype=int),
                  "bio": np.load(f"{SP}/biophys_v4_synth.npy").astype(np.float32)[:na]}

    # SAME structure/seeds as run_v4_final, only the group key differs
    all_idx = np.arange(N)
    dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15,
                                               random_state=999).split(all_idx, y, groups))
    assert set(groups[dev_idx]).isdisjoint(set(groups[test_idx])), "dev/test group leak!"
    print(f"N={N} dev={len(dev_idx)} test={len(test_idx)} | "
          f"groups: {len(set(groups.tolist()))} (vs 1063 per-positive)", flush=True)

    prog_path = f"{OUT}/cv_progress.json"
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"folds": {}}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p = np.full(N, np.nan)
    for k, (otr, ova) in enumerate(sgkf.split(dev_idx, y[dev_idx], groups[dev_idx])):
        tr_idx, va_idx = dev_idx[otr], dev_idx[ova]
        if str(k) in prog["folds"]:
            oof_p[va_idx] = np.array(prog["folds"][str(k)]["probs"])
            print(f"  [fold {k}] cached AUROC={prog['folds'][str(k)]['auroc']:.4f}", flush=True)
            continue
        assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx])), f"fold {k} group leak!"
        model, m, sd_, _ = E.train_model(tr_idx, va_idx, args, device, tok, tag=f"ccv{k}")
        pr, lb, _ = E.score_with(model, va_idx, m, sd_, args, device, tok)
        fau = roc_auc_score(lb, pr); oof_p[va_idx] = pr
        prog["folds"][str(k)] = {"auroc": float(fau), "probs": pr.tolist(), "va_idx": va_idx.tolist()}
        json.dump(prog, open(prog_path, "w"))
        print(f"  [fold {k}] val AUROC = {fau:.4f}  (n={len(lb)})", flush=True)
        del model
        if hasattr(torch, "mps"): torch.mps.empty_cache()

    dm = ~np.isnan(oof_p) & np.isin(all_idx, dev_idx)
    oau = roc_auc_score(y[dm], oof_p[dm])
    sau, nsn = struct_auroc(oof_p[dm], y[dm], is_struct[dm])
    cv = [prog["folds"][str(k)]["auroc"] for k in range(5) if str(k) in prog["folds"]]
    print(f"\n*** HONEST CLUSTER-GROUPED CV ***")
    print(f"  pooled-OOF AUROC      = {oau:.4f}   (v4 per-positive-grouped = 0.8474)")
    print(f"  pos-vs-STRUCT AUROC   = {sau:.4f}   (v4 = 0.8388, n_struct={nsn})")
    print(f"  fold mean             = {np.mean(cv):.4f} ± {np.std(cv):.4f}   (v4 = 0.8501 ± 0.0249)")
    json.dump({"cv_oof_auroc": float(oau), "cv_oof_struct_auroc": sau, "cv_struct_n": nsn,
               "cv_fold_scores": cv, "n_groups": int(len(set(groups.tolist()))),
               "v4_per_positive_oof": 0.8474, "note": "cluster-grouped (CD-HIT 0.90) honest baseline"},
              open(f"{OUT}/eval_summary.json", "w"), indent=2)
    print(f"Saved -> {OUT}/eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
