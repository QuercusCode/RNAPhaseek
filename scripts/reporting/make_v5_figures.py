"""Figure 19 for the v5 section:
  Left  — the honest AUROC story (leaky v4 -> honest v4 -> v5) + variance collapse + specificity.
  Right — the yeast-generalization gap (the acid test): yeast vs non-yeast, CV and locked test."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

cc = json.load(open("model/strict_eval_v4_clustercv/eval_summary.json"))
v5 = json.load(open("model/strict_eval_v5/eval_summary.json"))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.7))

# Panel A: AUROC progression + variance
labels = ["v4\n(per-positive,\nleaky)", "v4\n(cluster,\nhonest)", "v5\n(cluster,\n3.16x data)"]
vals = [0.8474, cc["cv_oof_auroc"], v5["cv_oof_auroc"]]
errs = [0.0, float(np.std(cc["cv_fold_scores"])), float(np.std(v5["cv_fold_scores"]))]
cols = ["#bdc3c7", "#e67e22", "#27ae60"]
b = axA.bar(labels, vals, yerr=errs, color=cols, edgecolor="black", lw=0.6, capsize=5)
axA.set_ylim(0.5, 1.0); axA.set_ylabel("CV pooled-OOF AUROC")
axA.axhline(0.8474, color="grey", ls=":", lw=0.8)
for bar, v, e in zip(b, vals, errs):
    axA.text(bar.get_x()+bar.get_width()/2, v+e+0.012, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
axA.text(1.0, 0.55, "paralog leak\n-0.025", ha="center", fontsize=7.5, color="#c0392b")
axA.text(2.0, 0.55, "+data: +0.061\nvariance 4x lower\nspecificity 0.80->0.91",
         ha="center", fontsize=7.5, color="#1e8449")
axA.set_title("Honest AUROC progression (leak-free)", fontweight="bold", fontsize=10)

# Panel B: yeast generalization gap
groups = ["CV (OOF)", "Locked test"]
yeast = [v5["cv_yeast_auroc"], np.nan]            # locked yeast not separately stored; show overall
overall = [v5["cv_oof_auroc"], v5["locked_test_auroc"]]
nonyeast = [v5["cv_nonyeast_auroc"], v5["locked_test_nonyeast_auroc"]]
x = np.arange(len(groups)); w = 0.27
axB.bar(x - w, [v5["cv_yeast_auroc"], v5["locked_test_auroc"]], w, label="yeast / overall",
        color="#2980b9", edgecolor="black", lw=0.5)
axB.bar(x, overall, w, label="all positives", color="#95a5a6", edgecolor="black", lw=0.5)
axB.bar(x + w, nonyeast, w, label="NON-yeast positives", color="#e74c3c", edgecolor="black", lw=0.5)
for xi, (yv, ov, nv) in enumerate(zip([v5["cv_yeast_auroc"], v5["locked_test_auroc"]], overall, nonyeast)):
    axB.text(xi - w, yv+0.012, f"{yv:.2f}", ha="center", fontsize=8)
    axB.text(xi + w, nv+0.012, f"{nv:.2f}", ha="center", fontsize=8, fontweight="bold", color="#c0392b")
axB.set_ylim(0.5, 1.0); axB.set_xticks(x); axB.set_xticklabels(groups)
axB.set_ylabel("AUROC"); axB.legend(frameon=False, fontsize=8, loc="lower left")
axB.set_title("The yeast-generalization gap (acid test)", fontweight="bold", fontsize=10)
axB.text(0.5, 0.92, "~15-20 pt gap: strong on yeast,\nmoderate on everything else",
         transform=axB.transAxes, ha="center", fontsize=8, style="italic", color="#555")

fig.suptitle("Figure 19 — v5 expansion: stability + specificity up, but the gain is yeast-concentrated",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("report_assets/fig19_v5_expansion.png", dpi=140, bbox_inches="tight")
print("saved fig19")
