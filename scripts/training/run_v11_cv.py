"""v11 CV — v6's organism-balanced 5-fold cluster-grouped CV, repointed at the v11 pool
(PRODUCTION v5/v6 pool + 18 adversarially-verified protein-free additions: RNA-G4 LLPS
positives + matched G4/mixed-sequence negatives + the 2CZ aptamer). PURELY ADDITIVE — no
removal/relabel (unlike v10, whose cleaning was net-negative). Same protocol/seeds as v6, so
v11-vs-v6 isolates the effect of the additions. Reports overall / struct-specificity / yeast /
non-yeast OOF AUROC vs v6 (0.884 / 0.898 / 0.909 / 0.798). Resumable.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/training/run_v11_cv.py
"""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # noqa
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed
from run_v5_final import subset_auroc
from run_v6_cv import train_orgbalanced   # reuse the exact organism-balanced fold trainer

OUT = "model/strict_eval_v11_cv"
SP = "Data/splits"


def build_pool_v11():
    pos = read_fasta("Data/raw/multispecies/strict_pool_v11_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v11_negatives_all.fasta")
    sneg = read_fasta("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
    pp = list_npz_sorted("Data/processed/fegs_v11_pos")
    npn = list_npz_sorted("Data/processed/fegs_v11_neg")
    spn = list_npz_sorted("Data/processed/fegs_struct_neg_v4")
    npos, nneg, nsn = len(pos), len(neg), len(sneg)
    assert len(pp) == npos and len(npn) == nneg and len(spn) == nsn, \
        f"FEGS misalign: pos {len(pp)}/{npos}  neg {len(npn)}/{nneg}  sneg {len(spn)}/{nsn}"
    E.G["seqs"] = [s for _, s in pos] + [s for _, s in neg] + [s for _, s in sneg]
    E.G["hdrs"] = [h for h, _ in pos] + [h for h, _ in neg] + [h for h, _ in sneg]
    E.G["paths"] = list(pp) + list(npn) + list(spn)
    E.G["y"] = np.concatenate([np.ones(npos), np.zeros(nneg), np.zeros(nsn)]).astype(int)
    E.G["bio"] = np.vstack([np.load(f"{SP}/biophys_v11_pos.npy"),
                            np.load(f"{SP}/biophys_v11_neg.npy"),
                            np.load(f"{SP}/biophys_v4_structneg.npy")]).astype(np.float32)
    is_struct = np.array(["hardneg_struct" in h for h in E.G["hdrs"]])
    print(f"v11 pool: {npos} pos / {nneg} real-neg / {nsn} struct-neg = {len(E.G['y'])}", flush=True)
    return is_struct


def main():
    set_seed(42); device = setup_device(); os.makedirs(OUT, exist_ok=True)
    is_struct = build_pool_v11()
    groups = np.load(f"{SP}/cluster_groups_v11.npy"); yeast = np.load(f"{SP}/is_yeast_v11.npy")
    assert len(groups) == len(E.G["y"]) == len(yeast), "split arrays misaligned with v11 pool"
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
        model, m, sd = train_orgbalanced(tr, va, aug, yeast, args, device, tok, f"v11cv{k}")
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
    print(f"\n*** v11 CV (ADDITIVE corpus: v5/v6 pool + 18 verified protein-free additions) ***", flush=True)
    print(f"  overall OOF = {oau:.4f}   (v6 = {v6['cv_oof_auroc']:.4f})")
    print(f"  STRUCT-spec = {sau:.4f}   (v6 = {v6['cv_struct_auroc']:.4f})")
    print(f"  YEAST       = {yau:.4f}   (v6 = {v6['cv_yeast_auroc']:.4f})   n={nyp}")
    print(f"  NON-YEAST   = {nyau:.4f}   (v6 = {v6['cv_nonyeast_auroc']:.4f})   n={nnyp}   <- the additions target this")
    json.dump({"cv_oof_auroc": float(oau), "cv_struct_auroc": sau, "cv_yeast_auroc": yau, "cv_nonyeast_auroc": nyau,
               "cv_nonyeast_n": nnyp, "cv_fold_scores": cv, "v6_cv_oof": v6["cv_oof_auroc"],
               "v6_cv_struct": v6["cv_struct_auroc"], "v6_cv_nonyeast": v6["cv_nonyeast_auroc"],
               "pool": "v11 additive (v5/v6 pool + 18 verified protein-free additions: RNA-G4 LLPS + matched negs + 2CZ aptamer)"},
              open(f"{OUT}/eval_summary.json", "w"), indent=2)
    print(f"Saved -> {OUT}/eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
