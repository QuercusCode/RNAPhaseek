"""Figure 23: v6 generators — GA vs DEN. Left: structure-dependence (design vs scramble)
shows both are structure-driven; right: the diversity/structure tradeoff (GA = one
maximally-grounded motif; DEN = a diverse library still as structure-dependent as real positives)."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_fasta(p):
    out=[]; h=None; s=""
    for ln in open(p):
        ln=ln.rstrip()
        if ln.startswith(">"):
            if h is not None: out.append(s)
            h=ln; s=""
        else: s+=ln
    if h is not None: out.append(s)
    return out


def mean_pairwise_identity(seqs):
    ids=[]
    for i in range(len(seqs)):
        for j in range(i+1,len(seqs)):
            a,b=seqs[i],seqs[j]; L=min(len(a),len(b))
            if L: ids.append(sum(a[k]==b[k] for k in range(L))/L)
    return float(np.mean(ids)) if ids else 1.0


d = json.load(open("model/strict_eval_v6_production/design_structure_dependence.json"))
den = json.load(open("model/strict_eval_v6_production/den_v6_summary.json"))
ga_id = mean_pairwise_identity(read_fasta("outputs/designs/designed_ga_v6.fasta"))
den_id = den["mean_pairwise_identity_top15"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.7))

# Panel A: structure-dependence dumbbell
order = ["random", "real_pos", "v6_DEN", "v6_GA"]
lbl = {"v6_GA": "GA (v6)", "v6_DEN": "DEN (v6)", "real_pos": "real LLPS positives", "random": "random RNA"}
for i, g in enumerate(order):
    ds, sc = d[g]["design"], d[g]["scramble"]
    hl = g.startswith("v6")
    axA.plot([sc, ds], [i, i], "-", color="#16a085" if hl else "#bdc3c7", lw=3 if hl else 2, zorder=1)
    axA.scatter([sc], [i], color="#e74c3c", s=70, zorder=2, edgecolor="black", lw=0.5)
    axA.scatter([ds], [i], color="#27ae60", s=70, zorder=2, edgecolor="black", lw=0.5)
    axA.text(max(ds, sc)+0.015, i, f"Δ={ds-sc:+.2f}", va="center", fontsize=9,
             fontweight="bold" if hl else "normal", color="#16a085" if hl else "#555")
axA.axvline(0.5, color="grey", ls=":", lw=0.8)
axA.set_yticks(range(len(order))); axA.set_yticklabels([lbl[g] for g in order])
axA.set_xlabel("v6-model P(LLPS)"); axA.set_xlim(0.0, 1.18)
axA.scatter([], [], color="#27ae60", s=70, edgecolor="black", lw=0.5, label="design")
axA.scatter([], [], color="#e74c3c", s=70, edgecolor="black", lw=0.5, label="scramble")
axA.legend(loc="lower right", frameon=False, fontsize=8)
axA.set_title("Both generators are structure-driven", fontweight="bold", fontsize=10)

# Panel B: diversity vs structure-dependence tradeoff
gens = ["GA (v6)", "DEN (v6)"]
diversity = [1-ga_id, 1-den_id]            # higher = more diverse
delta = [d["v6_GA"]["delta"], d["v6_DEN"]["delta"]]
pmax = [0.996, den["max_full_prob"]]
x = np.arange(2); w = 0.26
axB.bar(x-w, diversity, w, label="diversity (1 - pairwise id)", color="#8e44ad", edgecolor="black", lw=0.5)
axB.bar(x,   delta,     w, label="structure-dependence Δ",      color="#16a085", edgecolor="black", lw=0.5)
axB.bar(x+w, pmax,      w, label="best P(LLPS)",                color="#3498db", edgecolor="black", lw=0.5)
for xi in x:
    for off, vals in zip([-w,0,w],[diversity,delta,pmax]):
        axB.text(xi+off, vals[xi]+0.015, f"{vals[xi]:.2f}", ha="center", fontsize=7.5)
axB.set_xticks(x); axB.set_xticklabels(gens); axB.set_ylim(0,1.1)
axB.legend(frameon=False, fontsize=8, loc="upper center", ncol=1)
axB.set_title("The diversity ↔ structure tradeoff", fontweight="bold", fontsize=10)
axB.text(0.5, -0.16, "GA: one maximally-grounded motif  ·  DEN: diverse library, still as structural as real positives",
         transform=axB.transAxes, ha="center", fontsize=7.5, style="italic", color="#555")

fig.suptitle("Figure 23 — v6 de novo generators: GA (one optimal design) vs DEN (diverse library), both verified structure-grounded",
             fontweight="bold", fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig("report_assets/fig23_v6_generators.png", dpi=140, bbox_inches="tight")
print(f"saved fig23 (GA pairwise id={ga_id:.2f}, DEN pairwise id={den_id:.2f})")
