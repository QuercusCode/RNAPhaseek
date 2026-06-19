"""Structure-specificity benchmark — a STANDING probe for the RNAPhaseek blind spot.

Scores any hybrid checkpoint on the 26 adversarially-verified v11 G-quadruplex / aptamer
sequences (10 hard POS, 8 hard NEG, 8 SOFT) and reports how well the model makes the
MATCHED-PAIR discrimination: within pairs that share a G4 core but differ by one
flank/spacer/G-tract change that flips LLPS, does the model rank positive > negative?

This is the exact failure mode of the kissing-loop A_bar/B_bar OOD limit and the v8
"structure-specificity gap unlearnable from sequence" finding. The v6 production model
scores G-rich matched NEGATIVES as positives, so matched-pair accuracy ~chance. Use this
to measure whether a FUTURE architecture (domain-adversarial, 2nd-foundation ensemble,
explicit structural inputs) actually closes the gap.

  # default = v6 production on the v11 probe set
  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/analysis/benchmark_structure_specificity.py
  # any other model:
  python scripts/analysis/benchmark_structure_specificity.py --ckpt model/<m>/final_model.pt --norm model/<m>/norm_stats.npz --tag <m>
"""
import os, sys, json, argparse, tempfile
from pathlib import Path
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils         import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                     import RNABiophysicalExtractor
from Functions.precompute_fegs                     import process_fasta

# Mechanistically MATCHED pairs (same G4 core; one controlled change flips/modulates LLPS).
# The model SHOULD score pos > neg in each. (pos_name, neg_name, mechanism)
PAIRS = [
    ("5pA9",        "5pC9",   "5'-flank A9 condenses vs C9 (C9 base-pairs the G-core -> no LLPS)"),
    ("5pmixed",     "5pC9",   "5'-flank ACAGU5 condenses vs C9"),
    ("3pA9",        "3pC9",   "3'-flank A9 condenses vs C9 (no aggregates)"),
    ("3pA9",        "3pU9",   "3'-flank A9 condenses vs U9 (no aggregates)"),
    ("3pmixed",     "3pC9",   "3'-flank ACAGU5 vs C9"),
    ("5pA9",        "dual_U9","single A9 flank vs dual-U9 flank (no LLPS)"),
    ("G3A2_4",      "G2A2_4", "G-tract: 4xG3 (LLPS) vs 4xG2 (soluble at physiol. spermine)"),
    ("G3A2_4",      "G3A5_4", "spacer: A2 (LLPS) vs A5 (aggregation abolished)"),
    ("A3_G3A2_4_A", "G3A5_4", "poly-A-flanked G3 core (LLPS) vs A5-spacer soluble"),
]
FASTA_DEFAULT = "Data/raw/multispecies/staging/v11_additions.fasta"


def score(model, device, tok, args, paths, seqs, bio_norm):
    ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), bio_norm, args.max_nucleotides)
    ld = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
    probs = []
    with torch.no_grad():
        for tk, at, Lh, bi, _ in ld:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device)
            bi = bi.to(device) if bi is not None else None
            lg, _ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/strict_eval_v6_production/final_model.pt")
    ap.add_argument("--norm", default="model/strict_eval_v6_production/norm_stats.npz")
    ap.add_argument("--backbone", default="multimolecule/rnafm", help="backbone of --ckpt (auto dim)")
    ap.add_argument("--fasta", default=FASTA_DEFAULT)
    ap.add_argument("--tag", default="v6_production")
    ap.add_argument("--out", default="outputs/structure_specificity_benchmark.json")
    # optional 2nd model -> soft-vote ensemble + ens_std abstention signal
    ap.add_argument("--ckpt2", default="", help="2nd checkpoint for ensemble (avg probs)")
    ap.add_argument("--norm2", default="")
    ap.add_argument("--backbone2", default="multimolecule/ernierna")
    a = ap.parse_args()
    set_seed(42); device = setup_device()

    recs = read_fasta(a.fasta)
    label = [h.split("|")[0] for h, _ in recs]          # POS / NEG / SOFT
    name = [h.split("|")[2] if h.count("|") >= 2 else h[:16] for h, _ in recs]
    seqs = [s for _, s in recs]
    by_name = {name[i]: i for i in range(len(name))}
    n_pos = label.count("POS"); n_neg = label.count("NEG"); n_soft = label.count("SOFT")
    print(f"probe set: {len(seqs)} seqs ({n_pos} POS / {n_neg} NEG / {n_soft} SOFT) from {a.fasta}")

    # features (FEGS temp dir + 38-dim biophys), same path as score_external_v4
    ext = RNABiophysicalExtractor(normalize=False)
    d = Path(tempfile.mkdtemp(prefix="fegs_ssbench_"))
    process_fasta(Path(a.fasta), d, topk=10, seq_len=1024, overwrite=True, workers=2)
    paths = list_npz_sorted(str(d))
    assert len(paths) == len(seqs), f"FEGS {len(paths)} != seqs {len(seqs)} (a seq <10nt got dropped?)"
    bio = np.stack([ext._compute_one(s) for s in seqs]).astype(np.float32)
    assert bio.shape[1] == 38, bio.shape

    def score_model(ckpt, norm, backbone):
        nz = np.load(norm); m, sd = nz["mean"].astype(np.float32), nz["std"].astype(np.float32)
        margs = HybridTrainArgs(backbone=backbone, bio_dim=38, use_species_embed=False,
                                unfreeze_last_n=2, freeze_backbone=False)   # dim auto-detects from backbone
        model = RNAFMHybridClassifier(margs).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        tok = AutoTokenizer.from_pretrained(margs.backbone, trust_remote_code=True)
        return score(model, device, tok, margs, paths, seqs, (bio - m) / sd)

    p1 = score_model(a.ckpt, a.norm, a.backbone)
    ens_std = None
    if a.ckpt2:
        p2 = score_model(a.ckpt2, a.norm2 or a.norm, a.backbone2)
        p = (p1 + p2) / 2.0                          # soft-vote ensemble
        ens_std = np.abs(p1 - p2)                    # per-seq disagreement (abstention signal)
        print(f"[ensemble] avg of {a.backbone} + {a.backbone2}; mean |Δprob| = {ens_std.mean():.3f}")
    else:
        p = p1
    prob = {name[i]: float(p[i]) for i in range(len(name))}

    # ── hard 18: POS vs NEG ──
    hard = [i for i in range(len(label)) if label[i] in ("POS", "NEG")]
    y = np.array([1 if label[i] == "POS" else 0 for i in hard]); ph = p[list(hard)]
    hard_auroc = float(roc_auc_score(y, ph)) if len(set(y)) == 2 else None
    pos_p = ph[y == 1]; neg_p = ph[y == 0]

    # ── MATCHED-PAIR discrimination (the headline) ──
    pair_rows, margins = [], []
    for pn, nn, mech in PAIRS:
        if pn not in prob or nn not in prob:
            print(f"  [skip pair] {pn} vs {nn} (missing)"); continue
        margin = prob[pn] - prob[nn]; margins.append(margin)
        pair_rows.append({"pos": pn, "neg": nn, "prob_pos": prob[pn], "prob_neg": prob[nn],
                          "margin": margin, "correct": bool(margin > 0), "mechanism": mech})
    margins = np.array(margins)
    mp_acc = float((margins > 0).mean()) if len(margins) else None
    mp_mean = float(margins.mean()) if len(margins) else None

    # ── report ──
    print(f"\n=== STRUCTURE-SPECIFICITY BENCHMARK  [{a.tag}] ===")
    print(f"hard-18 AUROC (POS vs NEG)      : {hard_auroc:.3f}" if hard_auroc is not None else "hard AUROC n/a")
    print(f"  POS recall@0.5 : {int((pos_p>=0.5).sum())}/{len(pos_p)}   (mean {pos_p.mean():.3f})")
    print(f"  NEG  FP  @0.5  : {int((neg_p>=0.5).sum())}/{len(neg_p)}   (mean {neg_p.mean():.3f})  <- matched G-rich negatives")
    print(f"\nMATCHED-PAIR discrimination (pos should outscore neg):")
    print(f"  accuracy (concordance): {mp_acc:.2f}  ({int((margins>0).sum())}/{len(margins)} pairs)   [chance 0.50]")
    print(f"  mean margin prob(pos)-prob(neg): {mp_mean:+.3f}   (>0 = discriminating)")
    print(f"  {'pos':<13}{'neg':<10}{'p_pos':>7}{'p_neg':>7}{'margin':>8}  mechanism")
    for r in sorted(pair_rows, key=lambda x: x["margin"]):
        flag = "ok " if r["correct"] else "FAIL"
        print(f"  {r['pos']:<13}{r['neg']:<10}{r['prob_pos']:>7.3f}{r['prob_neg']:>7.3f}{r['margin']:>+8.3f} {flag} {r['mechanism'][:42]}")

    print(f"\nper-sequence scores by class:")
    for cls in ("POS", "NEG", "SOFT"):
        items = sorted([(name[i], float(p[i])) for i in range(len(label)) if label[i] == cls], key=lambda x: -x[1])
        print(f"  {cls}: " + "  ".join(f"{n}={v:.2f}" for n, v in items))

    ens_block = None
    if ens_std is not None:
        neg_i = [i for i in hard if label[i] == "NEG"]
        abst = float((ens_std[neg_i] > 0.05).mean())   # would abstention catch the matched negatives?
        ens_block = {"mean_abs_dprob": float(ens_std.mean()),
                     "neg_mean_abs_dprob": float(ens_std[neg_i].mean()),
                     "neg_abstain_rate@0.05": abst}
        print(f"\nENSEMBLE disagreement (ens_std): overall {ens_std.mean():.3f} | matched-NEG {ens_std[neg_i].mean():.3f} "
              f"| NEG abstain@0.05 = {abst:.2f}  (high = disagreement flags the blind-spot negatives)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    blob = json.load(open(a.out)) if os.path.exists(a.out) else {}
    blob[a.tag] = {"ckpt": a.ckpt, "backbone": a.backbone, "ensemble_with": (a.backbone2 if a.ckpt2 else None),
                   "n_pos": n_pos, "n_neg": n_neg, "n_soft": n_soft,
                   "hard18_auroc": hard_auroc, "pos_recall@0.5": float((pos_p >= 0.5).mean()),
                   "neg_fp@0.5": float((neg_p >= 0.5).mean()), "neg_mean": float(neg_p.mean()),
                   "matched_pair_accuracy": mp_acc, "matched_pair_mean_margin": mp_mean,
                   "ensemble": ens_block,
                   "pairs": pair_rows, "per_seq": {name[i]: {"label": label[i], "prob": float(p[i])} for i in range(len(name))}}
    json.dump(blob, open(a.out, "w"), indent=2)
    print(f"\nSaved -> {a.out}  (key '{a.tag}')")
    print("INTERPRETATION: matched_pair_accuracy near 0.50 and mean_margin near 0 = the model cannot tell "
          "LLPS-flipping flank/spacer changes apart (the structure-specificity blind spot). Higher is better.")


if __name__ == "__main__":
    main()
