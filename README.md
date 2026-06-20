# RNAPhaseek

Predict and de-novo **design RNA that undergoes liquid–liquid phase separation by itself**
(protein-free, RNA–RNA-driven LLPS). RNA-FM foundation model + FEGS structure adapter +
38-dim biophysics, with a genetic-algorithm / Deep-Exploration-Network generator.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuercusCode/RNAPhaseek/blob/main/notebooks/RNAPhaseek_colab.ipynb) — score sequences and design new RNAs in your browser, no install.

**Production model performance** (leakage-free 5-fold cluster-grouped CV):

| Metric | Value |
|---|---|
| Overall AUROC | **0.88** |
| Structural-specificity AUROC | **0.90** |
| Non-yeast generalization AUROC | **0.80** |
| Matched-pair accuracy (held-out structure-specificity benchmark) | **1.00** |

---

## Use the Colab notebook (no install)

The fastest path for new users: open [`notebooks/RNAPhaseek_colab.ipynb`](notebooks/RNAPhaseek_colab.ipynb) via the badge above. It clones this repo, downloads the weights from Hugging Face Hub, and exposes:

- **Score** — paste a FASTA, get a P(LLPS) CSV
- **GA design** — generate one optimal LLPS RNA + variants
- **DEN design** — generate a mutually-diverse library of LLPS RNAs

---

## Quick start (local)

All scripts run **from the project root** with the project conda env:

```bash
PY=/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python

# score arbitrary RNA for P(LLPS)
$PY rnaphaseek.py score my_rnas.fasta -o scores.csv

# design new phase-separating RNA
$PY rnaphaseek.py design --method ga  -o ga_designs.fasta     # one optimal motif + variants
$PY rnaphaseek.py design --method den -o den_library.fasta    # diverse library

# trustworthiness: is a design structure-driven or a composition artifact?
$PY rnaphaseek.py validate den_library.fasta -o trust.csv
```

See [`docs/RNAPHASEEK_CLI.md`](docs/RNAPHASEEK_CLI.md) for the full CLI reference.

---

## The model in one paragraph

Trained on the largest strict RNA-self-LLPS corpus assembled to date (Van Treeck protein-free
yeast self-assembly screen + repeat / G-quadruplex / riboswitch / RNase-P diversity), with
structural hard negatives, self-complementarity features, CD-HIT cluster-grouped leakage
control, organism-balanced sampling, and matched training pairs that close the structure-
specificity blind spot of earlier recipes. It is a strong yeast predictor (0.91) with solid
transfer to other organisms (0.80); its one real limit — cross-organism generalization — is
a **data-scarcity** constraint of the field, not a modeling one. Predictions are
*model-believed* candidates awaiting experimental validation.
