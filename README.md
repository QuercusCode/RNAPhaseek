# RNAPhaseek

Predict and de-novo **design RNA that undergoes liquid–liquid phase separation by itself**
(protein-free, RNA–RNA-driven LLPS). RNA-FM foundation model + FEGS structure adapter +
38-dim biophysics, with a genetic-algorithm / Deep-Exploration-Network generator.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuercusCode/RNAPhaseek/blob/main/notebooks/RNAPhaseek_colab.ipynb) — score sequences and design new RNAs in your browser, no install.

**Model performance** (leakage-free 5-fold cluster-grouped CV):

<div align="center">

| Metric | Value |
|---|---|
| Overall AUROC | **0.88** |
| Structural-specificity AUROC | **0.90** |
| Non-yeast generalization AUROC | **0.80** |
| Matched-pair accuracy (held-out structure-specificity benchmark) | **1.00** |

</div>

---

## Use the Colab notebook (no install)

The fastest path for new users: open [`notebooks/RNAPhaseek_colab.ipynb`](notebooks/RNAPhaseek_colab.ipynb) via the badge above. It clones this repo, downloads the weights from Hugging Face Hub, and exposes:

- **Score** — paste a FASTA, get a P(LLPS) CSV
- **GA design** — generate one optimal LLPS RNA + variants
- **DEN design** — generate a mutually-diverse library of LLPS RNAs

---

## Installation

```bash
# clone the repo
git clone https://github.com/QuercusCode/RNAPhaseek.git
cd RNAPhaseek

# create the conda environment
conda create -n rnaphaseek python=3.10 -y
conda activate rnaphaseek
pip install -r requirements.txt
```

Model weights are downloaded automatically from [Hugging Face Hub](https://huggingface.co/quercuscode/rnaphaseek) on first run.

---

## Quick start (local)

All commands run **from the project root** with the conda env activated:

```bash
conda activate rnaphaseek

# score arbitrary RNA for P(LLPS)
python rnaphaseek.py score my_rnas.fasta -o scores.csv

# design new phase-separating RNA
python rnaphaseek.py design --method ga  -o ga_designs.fasta     # one optimal motif + variants
python rnaphaseek.py design --method den -o den_library.fasta    # diverse library

# trustworthiness: is a design structure-driven or a composition artifact?
python rnaphaseek.py validate den_library.fasta -o trust.csv
```

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

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
