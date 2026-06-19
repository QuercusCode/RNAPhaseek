"""Figures for the v4 specificity section of the report.
  fig15: v3->v4 specificity gain (pos-vs-structural-negative AUROC) + no overall regression
  fig16: the kissing-loop OOD finding (A/A_bar resolved by kl_loop_comp; training has no KL class)
"""
import json, random, os, sys
sys.path.insert(0, os.getcwd())
import paths  # project path bootstrap (see paths.py)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from validate_kl_feature import features, read_named

ev = json.load(open("model/strict_eval_v4/eval_summary.json"))
ext = json.load(open("model/strict_eval_v4/external_validation_v4.json"))

# ── Figure 15: specificity gain ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

# left: pos-vs-structural-neg AUROC (the targeted metric)
labels = ["v3\n(documented)", "v4 CV-OOF\n(n=154)", "v4 locked\n(n=30)"]
vals = [0.50, ev["cv_oof_struct_auroc"], ev["locked_test_struct_auroc"]]
cols = ["#bdc3c7", "#27ae60", "#2ecc71"]
b = ax1.bar(labels, vals, color=cols, edgecolor="black", lw=0.6)
ax1.axhline(0.5, color="grey", ls=":", lw=1)
ax1.set_ylim(0, 1.0); ax1.set_ylabel("AUROC")
ax1.set_title("pos vs composition-matched\nstructure-destroyed negative", fontweight="bold", fontsize=10)
for bar, v in zip(b, vals):
    ax1.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax1.text(0.5, 0.93, "the gap v4 set out to close", transform=ax1.transAxes, ha="center",
         fontsize=8, style="italic", color="#555")

# right: overall AUROC unchanged
labels2 = ["v3 CV", "v4 CV", "v3 locked*", "v4 locked"]
vals2 = [0.847, ev["cv_oof_auroc"], 0.84, ev["locked_test_auroc"]]
cols2 = ["#bdc3c7", "#2980b9", "#bdc3c7", "#3498db"]
b2 = ax2.bar(labels2, vals2, color=cols2, edgecolor="black", lw=0.6)
ax2.set_ylim(0, 1.0); ax2.set_ylabel("AUROC")
ax2.set_title("Overall discrimination — no regression", fontweight="bold", fontsize=10)
for bar, v in zip(b2, vals2):
    ax2.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
ax2.text(0.99, -0.16, "*v3 locked not directly comparable (different pool)", transform=ax2.transAxes,
         ha="right", fontsize=7, color="#777")
fig.suptitle("Figure 15 — v4 closes the in-distribution specificity gap at no cost to overall AUROC",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("report_assets/fig15_v4_specificity.png", dpi=140, bbox_inches="tight"); plt.close()
print("saved fig15")

# ── Figure 16: kissing-loop OOD finding ──
# external KL controls
names = ["A", "A_bar", "B", "B_bar"]
extseq = read_named("Data/raw/multispecies/external/external_deleaked.fasta")
def find(nm):
    for h, s in extseq.items():
        if "|" in h and h.split("|")[2] == nm:
            return s
    return None
kl_ext = {nm: features(find(nm))["kl_loop_comp"] for nm in names}

# training positives vs their structural-negative shuffles (paired) — does kl separate?
pos = read_named("Data/raw/multispecies/strict_pool_v3_positives.fasta")
plist = list(pos.items())
sneg = read_named("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
kl_pos, kl_neg = [], []
for h, ns in sneg.items():
    pidx = int(h.split("parent=")[1].split("|")[0])
    kl_pos.append(features(plist[pidx][1])["kl_loop_comp"])
    kl_neg.append(features(ns)["kl_loop_comp"])
kl_pos = np.array(kl_pos); kl_neg = np.array(kl_neg)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))
xb = ["A", "A_bar", "B", "B_bar"]
yb = [kl_ext["A"], kl_ext["A_bar"], kl_ext["B"], kl_ext["B_bar"]]
cb = ["#27ae60", "#e74c3c", "#27ae60", "#e74c3c"]
bars = axA.bar(xb, yb, color=cb, edgecolor="black", lw=0.6)
axA.set_ylabel("kissing-loop complementarity (bp)")
axA.set_title("Designed KL controls:\nfeature resolves what v4 could not", fontweight="bold", fontsize=10)
for bar, v in zip(bars, yb):
    axA.text(bar.get_x()+bar.get_width()/2, v+0.1, str(int(v)), ha="center", fontweight="bold")
axA.text(0.5, 0.9, "condensing = green, scrambled control = red", transform=axA.transAxes,
         ha="center", fontsize=8, style="italic", color="#555")

bins = np.arange(0, 9) - 0.5
axB.hist(kl_pos, bins=bins, color="#27ae60", alpha=0.55, edgecolor="black", lw=0.5,
         label=f"training positives (mean {kl_pos.mean():.1f})")
axB.hist(kl_neg, bins=bins, color="#e74c3c", alpha=0.55, edgecolor="black", lw=0.5,
         label=f"their shuffled negatives (mean {kl_neg.mean():.1f})")
axB.set_xlabel("kissing-loop complementarity (bp)"); axB.set_ylabel("# sequences")
axB.set_title("Training: feature carries NO label signal\n(positives ≈ their shuffled negatives)",
              fontweight="bold", fontsize=10)
axB.legend(frameon=False, fontsize=8, loc="upper right")
axB.text(0.5, 0.62, "designed-KL signal (6 vs 2)\nnever appears in training",
         transform=axB.transAxes, ha="center", fontsize=8, style="italic", color="#555")
fig.suptitle("Figure 16 — The external gap is out-of-distribution: the KL signal is unlearnable from this corpus",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("report_assets/fig16_kl_ood.png", dpi=140, bbox_inches="tight"); plt.close()
print(f"saved fig16 (pos kl mean={kl_pos.mean():.2f}, neg kl mean={kl_neg.mean():.2f}, n={len(kl_pos)})")
