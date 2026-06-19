"""
Generate all performance + dataset figures for the comprehensive RNAPhaseek report.
Outputs PNGs to report_assets/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

os.makedirs("report_assets", exist_ok=True)
perf = json.load(open("report_assets/perf_data.json"))
data = json.load(open("report_assets/dataset_data.json"))

# House style
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 140, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": "--",
})
C = {"base": "#7f8c8d", "phase1": "#2980b9", "holdout": "#16a085",
     "strict": "#c0392b", "accent": "#8e44ad", "ok": "#27ae60", "bad": "#e74c3c"}


# ── FIG 1 — Model performance comparison ─────────────────────────────────────
def fig1():
    models = ["Base\nRNAPhaseek", "Hybrid\nPhase-1", "Hybrid\nHoldout",
              "Strict\nbaseline", "Strict v1\n(rigorous)", "Strict v2\n(+data hunt)"]
    auroc  = [0.7267, 0.7707, 0.7601, 0.6320, 0.6364, 0.7213]   # last two = leakage-free CV mean
    prauc  = [0.8167, 0.8516, 0.8419, 0.6569, 0.6415, 0.7169]   # last two = pooled OOF PR-AUC
    errs   = [None, None, None, None, 0.0469, 0.0334]           # CV std
    test_auroc = [None, 0.7642, 0.7642, None, 0.7369, 0.7824]   # held-out test (strict = locked test)
    cols   = [C["base"], C["phase1"], C["holdout"], C["strict"], C["accent"], C["ok"]]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    for xi, (a, p, e, t) in enumerate(zip(auroc, prauc, errs, test_auroc)):
        ax.bar(xi - w/2, a, w, color=cols[xi], edgecolor="black", linewidth=0.6,
               yerr=e, capsize=4, error_kw={"elinewidth": 1.2},
               label="Val/CV AUROC" if xi == 0 else None)
        ax.bar(xi + w/2, p, w, color=cols[xi], alpha=0.45, edgecolor="black", linewidth=0.6,
               label="PR-AUC" if xi == 0 else None)
        ax.text(xi - w/2, a + (e or 0) + 0.012, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(xi + w/2, p + 0.012, f"{p:.3f}", ha="center", fontsize=9, alpha=0.8)
        if t is not None:
            ax.plot(xi - w/2, t, marker="D", color="black", markersize=7, zorder=5)
            ax.text(xi - w/2, t + 0.018, f"test {t:.3f}", ha="center", fontsize=8, color="black")
    ax.axhline(0.5, color="grey", ls=":", lw=1.2)
    ax.text(5.35, 0.51, "chance", color="grey", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("Figure 1 — Model performance across the project")
    ax.legend(loc="upper right", frameon=False)
    ax.text(0, -0.18, "◆ = held-out test AUROC · 'Strict (final)' bar = leakage-free 5-fold CV "
            "mean±std; its ◆ is the locked-test AUROC",
            transform=ax.transAxes, fontsize=8, color="#555")
    fig.savefig("report_assets/fig1_model_performance.png"); plt.close(fig)
    print("fig1 ok")


# ── FIG 2 — Training curves ──────────────────────────────────────────────────
def fig2():
    fig, ax = plt.subplots(figsize=(9, 5))
    order = [("base_RNAPhaseek", C["base"], "Base RNAPhaseek"),
             ("hybrid_phase1",   C["phase1"], "Hybrid Phase-1"),
             ("hybrid_holdout",  C["holdout"], "Hybrid Holdout"),
             ("strict_baseline", C["strict"], "Strict baseline")]
    for key, col, lab in order:
        c = perf["curves"].get(key, {}).get("curve", [])
        if not c: continue
        au = [r["auroc"] for r in c]
        ax.plot(range(1, len(au)+1), au, color=col, lw=2, label=lab, alpha=0.9)
        # mark best
        bi = int(np.argmax(au))
        ax.plot(bi+1, au[bi], marker="o", color=col, markersize=6)
    ax.axhline(0.5, color="grey", ls=":", lw=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUROC")
    ax.set_ylim(0.45, 0.85)
    ax.set_title("Figure 2 — Validation AUROC vs. epoch (● = best)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig("report_assets/fig2_training_curves.png"); plt.close(fig)
    print("fig2 ok")


# ── FIG 3 — Dataset evolution (scale) ────────────────────────────────────────
def fig3():
    stages = ["Phase-1\n(human)", "Unified\nmulti-species", "Strict\nRNA-self-LLPS"]
    pos = [data["phase1_pool"]["positives"],
           data["unified_pool"]["totals"]["positives"],
           data["strict_pool"]["positives"]]
    neg = [data["phase1_pool"]["negatives"],
           data["unified_pool"]["totals"]["negatives"],
           data["strict_pool"]["matched_negatives"] + data["strict_pool"]["hard_negatives"]]
    x = np.arange(len(stages)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, pos, w, label="Positives", color=C["phase1"], edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, neg, w, label="Negatives", color=C["base"], edgecolor="black", linewidth=0.6)
    for xi, (p, n) in enumerate(zip(pos, neg)):
        ax.text(xi - w/2, p*1.05, f"{p:,}", ha="center", fontsize=9, fontweight="bold")
        ax.text(xi + w/2, n*1.05, f"{n:,}", ha="center", fontsize=9)
    ax.set_yscale("log"); ax.set_ylabel("Sequences (log scale)")
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_title("Figure 3 — Dataset evolution: expand broadly, then rigorously filter")
    ax.legend(frameon=False)
    fig.savefig("report_assets/fig3_dataset_evolution.png"); plt.close(fig)
    print("fig3 ok")


# ── FIG 4 — Multi-species composition ────────────────────────────────────────
def fig4():
    sp = data["unified_pool"]["positives"]
    items = sorted(sp.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, len(names)))
    ax.barh(names[::-1], vals[::-1], color=cols, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + max(vals)*0.01, i, f"{v:,}", va="center", fontsize=9)
    ax.set_xlabel("LLPS-positive sequences")
    ax.set_title("Figure 4 — Unified multi-species pool composition (positives)")
    fig.savefig("report_assets/fig4_species_composition.png"); plt.close(fig)
    print("fig4 ok")


# ── FIG 5 — Curation & verification funnel ───────────────────────────────────
def fig5():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    # Left: two-pass funnel
    p1 = data["curation_pass1"]; rc = data["curation_rechase"]
    cats = ["Pass-1\ncandidates", "Pass-1\nconfirmed", "Re-chase\nleads", "Re-chase\npromoted"]
    vals = [p1["total_candidates"], p1["confirmed"], rc["leads"], rc["promoted"]]
    cols = [C["base"], C["ok"], C["base"], C["ok"]]
    axL.bar(cats, vals, color=cols, edgecolor="black", linewidth=0.6)
    for i, v in enumerate(vals):
        axL.text(i, v + 0.8, str(v), ha="center", fontsize=10, fontweight="bold")
    axL.set_ylabel("RNA candidates")
    axL.set_title("Figure 5a — Adversarial curation funnel")
    # Right: evidence grade of confirmed
    g = data["curation_pass1"]["by_grade"]
    glabels = {"A_in_vitro_rna_only": "A: in-vitro\nRNA-only",
               "C_repeat_or_designed": "C: repeat/\ndesigned",
               "B_in_vivo_scaffold": "B: in-vivo\nscaffold"}
    order = ["A_in_vitro_rna_only", "C_repeat_or_designed", "B_in_vivo_scaffold"]
    gv = [g.get(k, 0) for k in order]
    gc = [C["ok"], C["phase1"], C["accent"]]
    axR.bar([glabels[k] for k in order], gv, color=gc, edgecolor="black", linewidth=0.6)
    for i, v in enumerate(gv):
        axR.text(i, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")
    axR.set_ylabel("Confirmed positives")
    axR.set_title("Figure 5b — Evidence grade (pass-1 confirmed)")
    fig.savefig("report_assets/fig5_curation_funnel.png"); plt.close(fig)
    print("fig5 ok")


# ── FIG 6 — Diagnostic separability (key scientific figure) ──────────────────
def fig6():
    d = data["diagnostic_strict_baseline"]["separability_full_pool"]
    labels = ["pos vs\nmatched\ntranscripts", "pos vs\nTERRA\n(validated neg)",
              "pos vs\nsub-threshold\nrepeats", "pos vs\ncomposition\nshuffles"]
    vals = [d["pos_vs_matched"], d["pos_vs_TERRA"], d["pos_vs_subthreshold"], d["pos_vs_shuffle"]]
    cols = [C["ok"], C["ok"], C["bad"], C["bad"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="grey", ls=":", lw=1.4)
    ax.text(3.45, 0.515, "chance", color="grey", fontsize=8)
    ax.set_ylabel("AUROC (positives vs. negative type)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 6 — Where the strict-baseline model succeeds and fails")
    leg = [Patch(facecolor=C["ok"], label="Solves (composition signal)"),
           Patch(facecolor=C["bad"], label="Fails (needs repeat-count / structure)")]
    ax.legend(handles=leg, loc="upper right", frameon=False, fontsize=9)
    fig.savefig("report_assets/fig6_diagnostic_separability.png"); plt.close(fig)
    print("fig6 ok")


# ── FIG 7 — Mean predicted probability by group ──────────────────────────────
def fig7():
    mp = data["diagnostic_strict_baseline"]["mean_prob"]
    labels = ["positives", "matched\nneg", "hard:TERRA", "hard:sub-\nthreshold", "hard:\nshuffle"]
    keys = ["positives", "matched_neg", "hard_TERRA", "hard_subthreshold", "hard_shuffle"]
    vals = [mp[k] for k in keys]
    cols = [C["ok"], C["base"], C["ok"], C["bad"], C["bad"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, vals, color=cols, edgecolor="black", linewidth=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="grey", ls=":", lw=1.2)
    ax.set_ylabel("Mean predicted P(LLPS)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Figure 7 — Mean model confidence by group (strict baseline)")
    fig.savefig("report_assets/fig7_mean_prob.png"); plt.close(fig)
    print("fig7 ok")


# ── FIG 8 — New feature discrimination (CAG repeat ladder) ──────────────────
def fig8():
    import sys; sys.path.insert(0, ".")
    from Functions.RNA_biophysical import RNABiophysicalExtractor
    ext = RNABiophysicalExtractor(normalize=False)
    ns = [10, 15, 20, 25, 31, 40, 47, 60, 100]
    tri = []
    for n in ns:
        f = ext._compute_one("CAG" * n)
        tri.append(f[31])  # tri_repeat_copies
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = [C["bad"] if n < 31 else C["ok"] for n in ns]
    ax.bar([str(n) for n in ns], tri, color=cols, edgecolor="black", linewidth=0.6)
    ax.axvspan(-0.5, 3.5, color=C["bad"], alpha=0.07)
    ax.axvspan(3.5, 8.5, color=C["ok"], alpha=0.07)
    ax.axvline(3.5, color="black", ls="--", lw=1.2)
    ax.text(3.55, max(tri)*0.9, "phase-separation\nthreshold ≈ 31\n(Jain & Vale 2017)", fontsize=8)
    for i, v in enumerate(tri):
        ax.text(i, v + 1, f"{int(v)}", ha="center", fontsize=8)
    ax.set_xlabel("(CAG)ₙ repeat number")
    ax.set_ylabel("tri_repeat_copies feature value")
    ax.set_title("Figure 8 — New absolute-repeat feature resolves the threshold")
    leg = [Patch(facecolor=C["bad"], label="Sub-threshold (hard negative)"),
           Patch(facecolor=C["ok"], label="Condensing (positive)")]
    ax.legend(handles=leg, loc="upper left", frameon=False)
    fig.savefig("report_assets/fig8_feature_discrimination.png"); plt.close(fig)
    print("fig8 ok")


for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
    try:
        fn()
    except Exception as e:
        import traceback; print(f"{fn.__name__} FAILED: {e}"); traceback.print_exc()

print("\nAll figures ->", os.path.abspath("report_assets"))
