"""
Synthetic repeat-ladder test: does the final strict model resolve the
phase-separation repeat threshold (~31, Jain & Vale 2017)?

Generates (CAG/CUG/CGG)_n across n=10..60 (one per n, most NEVER seen in
training), runs the full pipeline (RNA-FM tokens + FEGS + 33 biophysical
features), and scores with model/strict_eval/final_model.pt. A model that
learned the threshold should show P(LLPS) rising across n~31.

The final model's bio-normalisation stats are reconstructed from its exact
training split (the same deterministic splits the eval script used).

Run:
  python -m Functions.RNAPhaseek.synthetic_threshold_test
"""
import os, sys, tempfile
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
from Functions.precompute_fegs                     import process_fasta
from pathlib import Path

FINAL = "model/strict_eval/final_model.pt"
THRESH = 31
MOTIFS = ["CAG", "CUG", "CGG"]
NS = list(range(10, 61))
# repeat-numbers the model DID train on (exclude from held-out AUROC)
TRAINED_NEG_N = {18, 20, 22, 25}     # hard negatives (CAG/CUG)
TRAINED_POS_N = {31, 47}             # strict positives present at these n


def main():
    set_seed(42); device = setup_device()

    # ── 1. Generate synthetic repeats ──
    recs = [(f"syn|{m}|n={n}", m * n) for m in MOTIFS for n in NS]
    synth = "report_assets/synthetic_repeats.fasta"
    with open(synth, "w") as f:
        for h, s in recs:
            f.write(f">{h}\n{s}\n")
    print(f"Generated {len(recs)} synthetic repeats ({len(MOTIFS)} motifs x {len(NS)} n-values)")

    # ── 2. FEGS ──
    fegs_dir = Path("Data/processed/fegs_synth_threshold")
    process_fasta(Path(synth), fegs_dir, topk=10, seq_len=1024, overwrite=True, workers=4)
    paths = list_npz_sorted(str(fegs_dir))

    # ── 3. Biophysical (33-dim) + normalise with the FINAL model's train stats ──
    ext = RNABiophysicalExtractor(normalize=False)
    bio_synth = np.stack([ext._compute_one(s) for _, s in recs]).astype(np.float32)

    # Reconstruct the final model's exact training split to recover (m, sd)
    pos = read_fasta("Data/raw/multispecies/strict_pool_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_negatives_all.fasta")
    npos, nneg = len(pos), len(neg)
    y = np.concatenate([np.ones(npos), np.zeros(nneg)]).astype(int)
    bio_all = np.vstack([np.load("Data/splits/biophys_strict_pos.npy"),
                         np.load("Data/splits/biophys_strict_neg.npy")]).astype(np.float32)
    all_idx = np.arange(len(y))
    dev_idx, _ = train_test_split(all_idx, test_size=0.15, random_state=999, stratify=y)
    f_tr, _ = train_test_split(dev_idx, test_size=0.15, random_state=7, stratify=y[dev_idx])
    m = bio_all[f_tr].mean(0); sd = bio_all[f_tr].std(0).clip(min=1e-8)
    bio_synth_n = (bio_synth - m) / sd

    # ── 4. Score with the final model ──
    args = HybridTrainArgs(bio_dim=33, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
    model = RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(FINAL, map_location=device, weights_only=True))
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

    seqs = [s for _, s in recs]
    ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), bio_synth_n, args.max_nucleotides)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
    probs = []
    with torch.no_grad():
        for token_ids, att, Lhat, biob, _ in loader:
            token_ids = token_ids.to(device); att = att.to(device); Lhat = Lhat.to(device)
            biob = biob.to(device) if biob is not None else None
            lg, _ = model(token_ids, att, labels=None, Lhat_stack=Lhat, bio_features=biob)
            fin = torch.isfinite(lg).all(-1, keepdim=True)
            lg = torch.where(fin, lg, torch.zeros_like(lg))
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)

    # organise
    P = {m_: {} for m_ in MOTIFS}
    for (h, _), p in zip(recs, probs):
        _, mot, ntag = h.split("|"); n = int(ntag.split("=")[1])
        P[mot][n] = float(p)

    # ── 5. Held-out AUROC (exclude trained n) + report ──
    labels, scores, heldout = [], [], []
    for mot in MOTIFS:
        for n in NS:
            seen = (n in TRAINED_NEG_N) or (n in TRAINED_POS_N)
            lab = 1 if n >= THRESH else 0
            labels.append(lab); scores.append(P[mot][n]); heldout.append(not seen)
    labels = np.array(labels); scores = np.array(scores); heldout = np.array(heldout)
    au_all = roc_auc_score(labels, scores)
    au_ho = roc_auc_score(labels[heldout], scores[heldout])
    print("\n" + "=" * 56)
    print(f"THRESHOLD CLASSIFICATION (n>={THRESH} = should-condense)")
    print(f"  AUROC all n          = {au_all:.4f}  (n={len(labels)})")
    print(f"  AUROC held-out n only= {au_ho:.4f}  (n={int(heldout.sum())}, excludes trained {sorted(TRAINED_NEG_N|TRAINED_POS_N)})")
    for mot in MOTIFS:
        lo = np.mean([P[mot][n] for n in NS if n < THRESH])
        hi = np.mean([P[mot][n] for n in NS if n >= THRESH])
        print(f"  {mot}: mean P(LLPS) below thr={lo:.3f}  above thr={hi:.3f}  (gap {hi-lo:+.3f})")

    # ── 6. Plot dose-response ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, mot in zip(axes, MOTIFS):
        ns = NS; ys = [P[mot][n] for n in ns]
        cols = ["#27ae60" if n >= THRESH else "#e74c3c" for n in ns]
        ax.scatter(ns, ys, c=cols, s=28, zorder=3, edgecolor="black", linewidth=0.3)
        ax.plot(ns, ys, color="#888", lw=1, alpha=0.6, zorder=2)
        # mark trained n
        for n in (TRAINED_NEG_N | TRAINED_POS_N):
            if n in P[mot]:
                ax.scatter([n], [P[mot][n]], facecolors="none", edgecolors="blue", s=90, lw=1.6, zorder=4)
        ax.axvline(THRESH, color="black", ls="--", lw=1.2)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_title(f"(${mot}$)$_n$", fontweight="bold")
        ax.set_xlabel("repeat number n")
        ax.grid(alpha=0.25, ls="--")
    axes[0].set_ylabel("P(LLPS) — final strict model")
    axes[0].set_ylim(0, 1)
    fig.suptitle(f"Synthetic repeat-ladder test — held-out AUROC(n<{THRESH} vs n≥{THRESH}) = {au_ho:.3f}  "
                 f"(blue rings = repeat-numbers seen in training)", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "report_assets/fig9_threshold_test.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\nSaved figure -> {out}")

    import json
    json.dump({"auroc_all": float(au_all), "auroc_heldout": float(au_ho),
               "per_motif": {mot: {str(n): P[mot][n] for n in NS} for mot in MOTIFS}},
              open("report_assets/threshold_test.json", "w"), indent=2)
    print("Saved report_assets/threshold_test.json")


if __name__ == "__main__":
    main()
