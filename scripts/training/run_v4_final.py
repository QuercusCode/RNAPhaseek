"""
RNAPhaseek v4 evaluation — closes the specificity gap.

Pool = v3 positives (427) + v3 real negatives (636) + NEW structural hard
negatives (184, composition-matched but self-complementarity destroyed), all with
WIDTH-38 biophysics (Block-5 self-complementarity features added).

LEAKAGE CONTROL (red-team must-fix #1): parent-grouped splitting. A structural
negative derived from positive p shares group id p with that positive, so the two
(which are ~composition-identical) can never land on opposite sides of any split.
Real negatives are singleton groups. Seeds kept: 999 (test), 7 (final), 42 (CV).

PRIMARY metric (red-team): INTERNAL pos-vs-structural-negative AUROC, pooled over
the grouped 5-fold CV out-of-fold predictions (high power, ~all dev struct negs),
plus a locked-test confirmation. The 2-point external A-bar/B-bar test is corroborating.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python run_v4_final.py [--skip_cv]
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

OUT = "model/strict_eval_v4"
SP = "Data/splits"


def build_pool():
    pos = read_fasta("Data/raw/multispecies/strict_pool_v3_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v3_negatives_all.fasta")
    sneg = read_fasta("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
    pp = list_npz_sorted("Data/processed/fegs_topk_strict_v3_pos")
    npn = list_npz_sorted("Data/processed/fegs_topk_strict_v3_neg")
    spn = list_npz_sorted("Data/processed/fegs_struct_neg_v4")
    npos, nneg, nsn = min(len(pos), len(pp)), min(len(neg), len(npn)), min(len(sneg), len(spn))
    assert (npos, nneg, nsn) == (len(pos), len(neg), len(sneg)), \
        f"FEGS count mismatch: pos {npos}/{len(pos)} neg {nneg}/{len(neg)} sneg {nsn}/{len(sneg)}"

    E.G["seqs"]  = [s for _, s in pos] + [s for _, s in neg] + [s for _, s in sneg]
    E.G["hdrs"]  = [h for h, _ in pos] + [h for h, _ in neg] + [h for h, _ in sneg]
    E.G["paths"] = list(pp) + list(npn) + list(spn)
    E.G["y"]     = np.concatenate([np.ones(npos), np.zeros(nneg), np.zeros(nsn)]).astype(int)
    bio = np.vstack([np.load(f"{SP}/biophys_v4_pos.npy"),
                     np.load(f"{SP}/biophys_v4_neg.npy"),
                     np.load(f"{SP}/biophys_v4_structneg.npy")]).astype(np.float32)
    assert bio.shape == (npos + nneg + nsn, 38), f"bio shape {bio.shape}"
    E.G["bio"] = bio

    # parent-grouped groups: positive p -> group p; struct neg(parent=p) -> group p;
    # real neg -> unique singleton group (offset beyond positive ids).
    groups = list(range(npos))                                   # positives 0..npos-1
    groups += [10_000 + j for j in range(nneg)]                  # real negs: singletons
    for h, _ in sneg:                                            # struct negs follow parent
        groups.append(int(h.split("parent=")[1].split("|")[0]))
    groups = np.array(groups)
    assert len(groups) == len(E.G["y"])

    is_struct = np.array(["hardneg_struct" in h for h in E.G["hdrs"]])
    print(f"pool: {npos} pos / {nneg} real-neg / {nsn} struct-neg = {len(E.G['y'])}  "
          f"| {len(set(groups))} groups", flush=True)
    return groups, is_struct


def struct_auroc(probs, labels, is_struct_sub):
    """pos vs structural-neg AUROC over a given index subset."""
    pos = labels == 1
    sn = (labels == 0) & is_struct_sub
    if pos.sum() < 3 or sn.sum() < 3:
        return None, int(sn.sum())
    yy = np.concatenate([np.ones(int(pos.sum())), np.zeros(int(sn.sum()))])
    pp = np.concatenate([probs[pos], probs[sn]])
    return float(roc_auc_score(yy, pp)), int(sn.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_cv", action="store_true")
    a = ap.parse_args()
    set_seed(42); device = setup_device(); os.makedirs(OUT, exist_ok=True)

    groups, is_struct = build_pool()
    y = E.G["y"]; N = len(y)

    args = HybridTrainArgs(bio_dim=38, use_species_embed=False, unfreeze_last_n=2,
                           freeze_backbone=False, epochs=30, patience=6,
                           lr=1e-4, backbone_lr=5e-6, weight_decay=0.03)
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

    # Augmentation (width-38 synth biophys), train-side only — same mechanism as v3
    meta = json.load(open(f"{SP}/synthetic_train_meta.json"))
    ar = read_fasta(meta["fasta"]); apa = list_npz_sorted(meta["fegs_dir"]); na = min(len(ar), len(apa))
    E.G["aug"] = {"seqs": [s for _, s in ar[:na]], "paths": list(apa[:na]),
                  "y": np.array(meta["labels"][:na], dtype=int),
                  "bio": np.load(f"{SP}/biophys_v4_synth.npy").astype(np.float32)[:na]}

    # ── Grouped locked test (seed 999) ──
    all_idx = np.arange(N)
    dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15,
                                               random_state=999).split(all_idx, y, groups))
    # sanity: no group spans dev and test
    assert set(groups[dev_idx]).isdisjoint(set(groups[test_idx])), "group leak dev/test!"
    print(f"N={N} dev={len(dev_idx)} test={len(test_idx)} "
          f"| test: {int((y[test_idx]==1).sum())} pos / {int((y[test_idx]==0).sum())} neg "
          f"/ {int(is_struct[test_idx].sum())} struct", flush=True)

    # ── Grouped 5-fold CV on dev → pooled OOF (recoverable) ──
    if not a.skip_cv:
        prog_path = f"{OUT}/cv_progress.json"
        prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"folds": {}}
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof_p = np.full(N, np.nan); oof_done = np.zeros(N, bool)
        for k, (otr, ova) in enumerate(sgkf.split(dev_idx, y[dev_idx], groups[dev_idx])):
            tr_idx, va_idx = dev_idx[otr], dev_idx[ova]
            if str(k) in prog["folds"]:
                pr = np.array(prog["folds"][str(k)]["probs"])
                oof_p[va_idx] = pr; oof_done[va_idx] = True
                print(f"  [fold {k}] cached AUROC={prog['folds'][str(k)]['auroc']:.4f}", flush=True)
                continue
            assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx])), f"fold {k} group leak!"
            m, sd_, ba = None, None, None
            model, m, sd_, ba = E.train_model(tr_idx, va_idx, args, device, tok, tag=f"cv{k}")
            pr, lb, _ = E.score_with(model, va_idx, m, sd_, args, device, tok)
            fau = roc_auc_score(lb, pr)
            oof_p[va_idx] = pr; oof_done[va_idx] = True
            prog["folds"][str(k)] = {"auroc": float(fau), "probs": pr.tolist(),
                                     "va_idx": va_idx.tolist()}
            json.dump(prog, open(prog_path, "w"))
            print(f"  [fold {k}] val AUROC = {fau:.4f}  (n={len(lb)})", flush=True)
            del model; torch.mps.empty_cache() if hasattr(torch, "mps") else None
        # pooled OOF metrics over dev
        dm = oof_done & np.isin(np.arange(N), dev_idx)
        oau = roc_auc_score(y[dm], oof_p[dm])
        sau, nsn = struct_auroc(oof_p[dm], y[dm], is_struct[dm])
        cv_scores = [prog["folds"][str(k)]["auroc"] for k in range(5) if str(k) in prog["folds"]]
        print(f"\n*** CV pooled-OOF AUROC = {oau:.4f} | pos-vs-STRUCT AUROC = "
              f"{sau:.4f} (n_struct={nsn}) | fold mean {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f} ***",
              flush=True)
    else:
        oau = sau = nsn = None; cv_scores = []

    # ── Final model on all dev (grouped inner-val seed 7) ──
    f_tr, f_va = next(GroupShuffleSplit(n_splits=1, test_size=0.15,
                                        random_state=7).split(dev_idx, y[dev_idx], groups[dev_idx]))
    f_tr, f_va = dev_idx[f_tr], dev_idx[f_va]
    assert set(groups[f_tr]).isdisjoint(set(groups[f_va])), "final group leak!"
    model, fm, fsd, _ = E.train_model(f_tr, f_va, args, device, tok, tag="final")
    torch.save(model.state_dict(), f"{OUT}/final_model.pt")
    np.savez(f"{OUT}/norm_stats.npz", mean=fm, std=fsd)  # for external re-test

    # ── Score locked test ──
    tpr, tlb, thd = E.score_with(model, test_idx, fm, fsd, args, device, tok)
    tau = roc_auc_score(tlb, tpr); tpa = average_precision_score(tlb, tpr)
    tacc = accuracy_score(tlb, (tpr >= 0.5).astype(int))
    tsau, tnsn = struct_auroc(tpr, tlb, is_struct[test_idx])
    print(f"\n*** LOCKED-TEST AUROC = {tau:.4f}  PR-AUC = {tpa:.4f}  acc@0.5 = {tacc*100:.1f}%  (n={len(tlb)})",
          flush=True)
    print(f"    locked-test pos-vs-STRUCT AUROC = {tsau} (n_struct={tnsn})", flush=True)
    E.diagnostic(tpr, tlb, thd, "Locked test v4")

    json.dump({"cv_oof_auroc": oau, "cv_oof_struct_auroc": sau, "cv_struct_n": nsn,
               "cv_fold_scores": cv_scores,
               "locked_test_auroc": float(tau), "locked_test_prauc": float(tpa),
               "locked_test_acc": float(tacc), "locked_test_struct_auroc": tsau,
               "locked_test_struct_n": tnsn, "n_test": int(len(tlb)), "n_dev": int(len(dev_idx)),
               "test_pos": int((tlb == 1).sum()), "test_neg": int((tlb == 0).sum()),
               "per_seq": [{"hdr": thd[i], "label": int(tlb[i]), "prob": float(tpr[i]),
                            "is_struct": bool(is_struct[test_idx][i])} for i in range(len(tlb))],
               "config": {"bio_dim": 38, "pool": "v4_structneg", "grouped_split": True}},
              open(f"{OUT}/eval_summary.json", "w"), indent=2)
    print(f"Saved -> {OUT}/eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
