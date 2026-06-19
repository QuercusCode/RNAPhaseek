"""Figure 1 — RNAPhaseek architecture schematic (clean box-and-arrow diagram)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

def box(x, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.2, edgecolor="#34495e", facecolor=fc))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1c2833", wrap=True)

def arrow(x1, y1, x2, y2, color="#7f8c8d"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=color))

# input
box(2, 26, 13, 8, "RNA sequence\n(any length)", "#eaf2f8", 10, True)
# three feature streams
box(22, 44, 24, 9, "RNA-FM foundation model\n(640-d, last 2 layers fine-tuned)", "#d6eaf8", 8.5)
box(22, 30, 24, 9, "FEGSTrans adapter\n(graph-bias structural encoding)", "#d5f5e3", 8.5)
box(22, 16, 24, 9, "38 biophysical features\n(+5 self-complementarity)", "#fcf3cf", 8.5)
for yy in (48.5, 34.5, 20.5):
    arrow(15, 30, 22, yy)
# fusion + classifier
box(53, 28, 17, 12, "Attention pooling\n+ fusion\n+ classifier", "#d6eaf8", 9, True)
for yy in (48.5, 34.5, 20.5):
    arrow(46, yy, 53, 34)
# P(LLPS)
box(74, 38, 14, 8, "P(LLPS)\nleak-free AUROC 0.88", "#abebc6", 9, True)
arrow(70, 36, 74, 42)
# generators
box(74, 22, 14, 10, "De novo generators\nSeqProp / GA / DEN", "#fadbd8", 8.5, True)
arrow(70, 33, 74, 27)
# designs + validation
box(74, 7, 14, 8, "Candidate RNAs\n+ structure-dependence\ntrustworthiness (Δ)", "#f9e79f", 8, True)
arrow(81, 22, 81, 15)
arrow(88, 42, 92, 42); ax.text(93, 42, "score", fontsize=8, va="center", color="#1e8449")
arrow(88, 11, 92, 11); ax.text(93, 11, "design\n+ validate", fontsize=8, va="center", color="#b9770e")

ax.text(50, 57, "RNAPhaseek", ha="center", fontsize=15, fontweight="bold", color="#1c2833")
ax.text(50, 53.5, "foundation-model prediction and de novo design of self–phase-separating RNA",
        ha="center", fontsize=9.5, style="italic", color="#566573")
fig.savefig("docs/figures/Figure1_architecture.png", dpi=200, bbox_inches="tight")
print("saved Figure1_architecture.png")
