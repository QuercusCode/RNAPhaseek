"""v8 CV — identical protocol to run_v6_cv (5-fold cluster-grouped, organism-balanced,
locked-test held out, resumable) but bio_dim=45: adds Block 6 (partition-function ensemble)
+ Block 7 (intermolecular multivalency) to the 38-dim biophysics. Tests whether richer /
intermolecular structure improves overall OOF AUROC and especially STRUCT-specificity
(pos vs dinucleotide-scramble) over v6 (overall 0.884 / struct 0.898).

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/training/run_v8_cv.py
"""
import os, sys, json, multiprocessing as mp
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
from run_v5_final import build_pool_v5, subset_auroc
from run_v6_cv import train_orgbalanced  # reuse the exact organism-balanced fold trainer

OUT = "model/strict_eval_v8_cv"
SP = "Data/splits"

_EXT = None
def _init():
    global _EXT
    from Functions.RNA_biophysical import RNABiophysicalExtractor
    _EXT = RNABiophysicalExtractor(normalize=False, extended=True)
def _ext7(s):
    return _EXT._compute_one(s)[38:]   # the 7 new features (Block 6 + 7)
def ext7_parallel(seqs, cache):
    if os.path.exists(cache):
        a = np.load(cache)
        if len(a) == len(seqs):
            print(f"  [ext7] cached {cache} ({a.shape})", flush=True)
            return a
    with mp.Pool(min(24, mp.cpu_count() - 2), initializer=_init) as p:
        a = np.stack(p.map(_ext7, seqs, chunksize=4)).astype(np.float32)
    np.save(cache, a)
    print(f"  [ext7] computed {cache} ({a.shape})", flush=True)
    return a


def main():
    set_seed(42); device = setup_device(); os.makedirs(OUT, exist_ok=True)
    is_struct = build_pool_v5()
    # ── extend bio to 45-dim (aligned to E.G['seqs'] by construction) ──
    new7 = ext7_parallel(E.G["seqs"], f"{SP}/biophys_v5_ext7.npy")
    assert len(new7) == len(E.G["seqs"])
    E.G["bio"] = np.hstack([E.G["bio"], new7]).astype(np.float32)
    print(f"  pool bio extended -> {E.G['bio'].shape}", flush=True)

    groups = np.load(f"{SP}/cluster_groups_v5.npy"); yeast = np.load(f"{SP}/is_yeast_v5.npy")
    y = E.G["y"]; N = len(y); all_idx = np.arange(N)
    args = HybridTrainArgs(bio_dim=45, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False,
                           epochs=30, patience=6, lr=1e-4, backbone_lr=5e-6, weight_decay=0.03)
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    meta = json.load(open(f"{SP}/synthetic_train_meta.json"))
    ar = read_fasta(meta["fasta"]); apa = list_npz_sorted(meta["fegs_dir"]); na = min(len(ar), len(apa))
    aug_seqs = [s for _, s in ar[:na]]
    new7s = ext7_parallel(aug_seqs, f"{SP}/biophys_v4_synth_ext7.npy")
    aug = {"seqs": aug_seqs, "paths": list(apa[:na]), "y": np.array(meta["labels"][:na], dtype=int),
           "bio": np.hstack([np.load(f"{SP}/biophys_v4_synth.npy").astype(np.float32)[:na], new7s]).astype(np.float32)}

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
        model, m, sd = train_orgbalanced(tr, va, aug, yeast, args, device, tok, f"v8cv{k}")
        pr, lb, _ = E.score_with(model, va, m, sd, args, device, tok)
        oof[va] = pr; prog["folds"][str(k)] = {"auroc": float(roc_auc_score(lb, pr)), "probs": pr.tolist(), "va_idx": va.tolist()}
        json.dump(prog, open(prog_path, "w"))
        print(f"  [fold {k}] val AUROC={prog['folds'][str(k)]['auroc']:.4f} (n={len(lb)})  "
              f"(v6 folds ~0.88-0.89)", flush=True)
        del model
        if hasattr(torch, "mps"): torch.mps.empty_cache()

    dm = ~np.isnan(oof) & np.isin(all_idx, dev_idx)
    oau = roc_auc_score(y[dm], oof[dm])
    sau, _ = subset_auroc(oof[dm], y[dm], is_struct[dm] | True, mask_neg=is_struct[dm])
    yau, nyp = subset_auroc(oof[dm], y[dm], yeast[dm])
    nyau, nnyp = subset_auroc(oof[dm], y[dm], ~yeast[dm])
    cv = [prog["folds"][str(k)]["auroc"] for k in range(5) if str(k) in prog["folds"]]
    v6 = json.load(open("model/strict_eval_v6_cv/eval_summary.json"))
    print(f"\n*** v8 CV (bio_dim=45: +partition-function +intermolecular) ***", flush=True)
    print(f"  overall OOF  = {oau:.4f}   (v6 = {v6['cv_oof_auroc']:.4f})")
    print(f"  STRUCT-spec  = {sau:.4f}   (v6 = {v6['cv_struct_auroc']:.4f})   <- does item 8 reject scrambles better?")
    print(f"  YEAST-pos    = {yau:.4f}   (v6 = {v6['cv_yeast_auroc']:.4f})   n={nyp}")
    print(f"  NON-YEAST    = {nyau:.4f}   (v6 = {v6['cv_nonyeast_auroc']:.4f})   n={nnyp}")
    json.dump({"cv_oof_auroc": float(oau), "cv_struct_auroc": sau, "cv_yeast_auroc": yau,
               "cv_nonyeast_auroc": nyau, "cv_nonyeast_n": nnyp, "cv_fold_scores": cv,
               "v6_cv_oof": v6["cv_oof_auroc"], "v6_cv_struct": v6["cv_struct_auroc"],
               "v6_cv_nonyeast": v6["cv_nonyeast_auroc"], "bio_dim": 45,
               "added": "Block6 partition-function (4) + Block7 intermolecular (3)"},
              open(f"{OUT}/eval_summary.json", "w"), indent=2)
    print(f"Saved -> {OUT}/eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
