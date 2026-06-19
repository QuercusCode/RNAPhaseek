"""Figure 20: organism-balanced training (v6) vs v5 — both 5-fold cluster-grouped CV.
Shows the non-yeast generalization lift at flat overall AUROC (the near-free win)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

v5 = json.load(open("model/strict_eval_v5/eval_summary.json"))
v6 = json.load(open("model/strict_eval_v6_cv/eval_summary.json"))

metrics = ["Overall\n(OOF)", "Yeast-pos", "NON-yeast\n(n=268)", "pos-vs-struct\n(specificity)"]
v5v = [v5["cv_oof_auroc"], v5["cv_yeast_auroc"], v5["cv_nonyeast_auroc"], v5["cv_struct_auroc"]]
v6v = [v6["cv_oof_auroc"], v6["cv_yeast_auroc"], v6["cv_nonyeast_auroc"], v6["cv_struct_auroc"]]

x = np.arange(len(metrics)); w = 0.36
fig, ax = plt.subplots(figsize=(10, 4.8))
b1 = ax.bar(x - w/2, v5v, w, label="v5 (unbalanced)", color="#95a5a6", edgecolor="black", lw=0.6)
b2 = ax.bar(x + w/2, v6v, w, label="v6 (organism-balanced)", color="#27ae60", edgecolor="black", lw=0.6)
# highlight the non-yeast bar
b2[2].set_color("#16a085"); b2[2].set_edgecolor("#0e6655"); b2[2].set_linewidth(1.6)
for bars, vals in [(b1, v5v), (b2, v6v)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=8)
# delta annotations
deltas = [v6v[i]-v5v[i] for i in range(4)]
for i, d in enumerate(deltas):
    c = "#16a085" if (i == 2) else ("#7f8c8d")
    ax.text(x[i], 0.56, f"{d:+.3f}", ha="center", fontsize=8.5,
            fontweight="bold" if i == 2 else "normal", color=c)
ax.axhline(0.5, color="grey", ls=":", lw=0.8)
ax.set_ylim(0.5, 1.0); ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("AUROC"); ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("Figure 20 — Organism-balanced training (v6): +0.036 non-yeast generalization at flat overall AUROC",
             fontweight="bold", fontsize=10.5)
ax.text(0.5, 0.93, "down-weighting the yeast bloc narrows the generalization gap ~30% (15.7->11.1 pt) at no overall cost",
        transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="#555")
fig.tight_layout()
fig.savefig("report_assets/fig20_v6_orgbalanced.png", dpi=140, bbox_inches="tight")
print("saved fig20")
