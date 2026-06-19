"""Figure 18: structure-dependence of de novo designs under the v4 model.
Dumbbell — for each group, P(design) vs P(composition-matched scramble); the gap is
the structure-dependence (trustworthiness). v4-GA designs show the largest gap."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("model/strict_eval_v4/design_structure_dependence.json"))
order = ["random", "GA", "SeqProp", "real_pos", "DEN", "GA_v4"]
order = [g for g in order if g in d]
labels = {"GA": "GA (v3-era)", "GA_v4": "GA on v4 (new)", "DEN": "DEN (v3-era)",
          "SeqProp": "SeqProp (v3-era)", "real_pos": "real LLPS positives", "random": "random RNA"}

fig, ax = plt.subplots(figsize=(9.5, 4.8))
for i, g in enumerate(order):
    r = d[g]; ds, sc = r["v4_design"], r["v4_scramble"]
    hl = (g == "GA_v4")
    ax.plot([sc, ds], [i, i], "-", color="#16a085" if hl else "#bdc3c7", lw=3 if hl else 2, zorder=1)
    ax.scatter([sc], [i], color="#e74c3c", s=70, zorder=2, edgecolor="black", lw=0.5)
    ax.scatter([ds], [i], color="#27ae60", s=70, zorder=2, edgecolor="black", lw=0.5)
    ax.text(max(ds, sc) + 0.015, i, f"Δ={ds-sc:+.2f}", va="center", fontsize=9,
            fontweight="bold" if hl else "normal", color="#16a085" if hl else "#555")

ax.axvline(0.5, color="grey", ls=":", lw=1)
ax.text(0.5, -0.85, "decision threshold", ha="center", fontsize=7, color="grey")
ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[g] for g in order])
ax.set_xlabel("v4-model P(LLPS)"); ax.set_xlim(0.3, 1.12)
ax.scatter([], [], color="#27ae60", s=70, edgecolor="black", lw=0.5, label="design")
ax.scatter([], [], color="#e74c3c", s=70, edgecolor="black", lw=0.5, label="composition-matched scramble")
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_title("Figure 18 — Structure-dependence of designs under the v4 model\n"
             "(wide gap = score is structure-driven = trustworthy)", fontweight="bold", fontsize=11)
fig.tight_layout()
fig.savefig("report_assets/fig18_design_structure_dependence.png", dpi=140, bbox_inches="tight")
print("saved fig18")
