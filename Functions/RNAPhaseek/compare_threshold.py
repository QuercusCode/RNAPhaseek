"""
Before/after threshold comparison: does synthetic augmentation sharpen the
model's repeat-threshold response?

Scores BOTH final models on the same synthetic (CAG/CUG/CGG)_n ladder, each
with its own correct bio-normalisation, and reports held-out threshold AUROC
on an IDENTICAL held-out n-set (excluding every repeat-number either model
trained on). Produces Figure 10 (overlay) + threshold_compare.json.

Run AFTER the augmented eval finishes (needs model/strict_eval_aug/final_model.pt):
  python -m Functions.RNAPhaseek.compare_threshold
"""
import os, sys, json
import numpy as np
import torch
import multimolecule  # noqa
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils         import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                     import RNABiophysicalExtractor

BASE_MODEL = "model/strict_eval/final_model.pt"
AUG_MODEL  = "model/strict_eval_aug/final_model.pt"
LADDER_FA  = "report_assets/synthetic_repeats.fasta"
LADDER_FEGS= "Data/processed/fegs_synth_threshold"
THRESH, MOTIFS, NS = 31, ["CAG", "CUG", "CGG"], list(range(10, 61))


def recover_norm(include_synth: bool):
    """Recompute the bio (m, sd) the given final model trained with."""
    pos = read_fasta("Data/raw/multispecies/strict_pool_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_negatives_all.fasta")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
    bio = np.vstack([np.load("Data/splits/biophys_strict_pos.npy"),
                     np.load("Data/splits/biophys_strict_neg.npy")]).astype(np.float32)
    dev_idx, _ = train_test_split(np.arange(len(y)), test_size=0.15, random_state=999, stratify=y)
    f_tr, _ = train_test_split(dev_idx, test_size=0.15, random_state=7, stratify=y[dev_idx])
    bio_tr = bio[f_tr]
    if include_synth:
        bio_tr = np.vstack([bio_tr, np.load("Data/splits/biophys_synth_train.npy").astype(np.float32)])
    return bio_tr.mean(0), bio_tr.std(0).clip(min=1e-8)


def score_ladder(model_path, m, sd, recs, paths, bio_ladder, device, tok, args):
    model = RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    seqs = [s for _, s in recs]
    ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), (bio_ladder - m) / sd, args.max_nucleotides)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
    out = []
    with torch.no_grad():
        for tk, at, Lh, bi, _ in loader:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device)
            bi = bi.to(device) if bi is not None else None
            lg, _ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True)
            lg = torch.where(fin, lg, torch.zeros_like(lg))
            out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    del model
    return np.concatenate(out)


def to_grid(recs, probs):
    P = {m_: {} for m_ in MOTIFS}
    for (h, _), p in zip(recs, probs):
        _, mot, ntag = h.split("|"); P[mot][int(ntag.split("=")[1])] = float(p)
    return P


def main():
    set_seed(42); device = setup_device()
    if not os.path.exists(AUG_MODEL):
        print(f"ERROR: {AUG_MODEL} not found — augmented eval not finished yet."); sys.exit(1)

    recs = read_fasta(LADDER_FA)
    paths = list_npz_sorted(LADDER_FEGS)
    ext = RNABiophysicalExtractor(normalize=False)
    bio_ladder = np.stack([ext._compute_one(s) for _, s in recs]).astype(np.float32)

    args = HybridTrainArgs(bio_dim=33, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

    # Held-out n-set: exclude every repeat-number EITHER model trained on
    meta = json.load(open("Data/splits/synthetic_train_meta.json"))
    seen = set(meta["trained_n_values"]) | {18, 20, 22, 25, 31, 47}
    heldout_ns = [n for n in NS if n not in seen]
    print(f"Held-out n (neither model trained on, in 10..60): {heldout_ns}")

    # Score both
    mb, sb = recover_norm(include_synth=False)
    ma, sa = recover_norm(include_synth=True)
    Pb = to_grid(recs, score_ladder(BASE_MODEL, mb, sb, recs, paths, bio_ladder, device, tok, args))
    Pa = to_grid(recs, score_ladder(AUG_MODEL,  ma, sa, recs, paths, bio_ladder, device, tok, args))

    def held_auroc(P):
        labs, scs = [], []
        for mot in MOTIFS:
            for n in heldout_ns:
                labs.append(1 if n >= THRESH else 0); scs.append(P[mot][n])
        return roc_auc_score(labs, scs)

    au_b, au_a = held_auroc(Pb), held_auroc(Pa)
    print("\n" + "=" * 56)
    print(f"HELD-OUT THRESHOLD AUROC (n<{THRESH} vs n>={THRESH}, n={len(heldout_ns)*3} points)")
    print(f"  Baseline (no aug)  = {au_b:.4f}")
    print(f"  Augmented          = {au_a:.4f}   (Δ {au_a-au_b:+.4f})")
    for mot in MOTIFS:
        gb = np.mean([Pb[mot][n] for n in NS if n >= THRESH]) - np.mean([Pb[mot][n] for n in NS if n < THRESH])
        ga = np.mean([Pa[mot][n] for n in NS if n >= THRESH]) - np.mean([Pa[mot][n] for n in NS if n < THRESH])
        print(f"  {mot}: above-below gap  baseline={gb:+.3f}  augmented={ga:+.3f}  (Δ {ga-gb:+.3f})")

    # Figure 10: overlay
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, mot in zip(axes, MOTIFS):
        yb = [Pb[mot][n] for n in NS]; ya = [Pa[mot][n] for n in NS]
        ax.plot(NS, yb, color="#c0392b", lw=1.6, alpha=0.6, label="Baseline", marker="o", ms=3)
        ax.plot(NS, ya, color="#8e44ad", lw=2.0, label="Augmented", marker="o", ms=3)
        ax.axvline(THRESH, color="black", ls="--", lw=1.2)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_title(f"(${mot}$)$_n$", fontweight="bold")
        ax.set_xlabel("repeat number n"); ax.grid(alpha=0.25, ls="--")
    axes[0].set_ylabel("P(LLPS)"); axes[0].set_ylim(0, 1); axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Figure 10 — Threshold response before/after synthetic augmentation   "
                 f"(held-out AUROC: {au_b:.3f} → {au_a:.3f})", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("report_assets/fig10_threshold_compare.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("\nSaved -> report_assets/fig10_threshold_compare.png")

    json.dump({"baseline_heldout_auroc": float(au_b), "augmented_heldout_auroc": float(au_a),
               "delta": float(au_a - au_b), "heldout_ns": heldout_ns,
               "baseline_grid": Pb, "augmented_grid": Pa},
              open("report_assets/threshold_compare.json", "w"), indent=2)
    print("Saved -> report_assets/threshold_compare.json")


if __name__ == "__main__":
    main()
