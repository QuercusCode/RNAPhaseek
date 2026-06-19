# RNAPhaseek

Predict and de-novo **design RNA that undergoes liquid–liquid phase separation by itself**
(protein-free, RNA–RNA-driven LLPS). RNA-FM foundation model + FEGS structure adapter +
38-dim biophysics, with a genetic-algorithm / Deep-Exploration-Network generator.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuercusCode/RNAPhaseek/blob/main/notebooks/RNAPhaseek_colab.ipynb) — score sequences and design new RNAs in your browser, no install.

**Production model performance**: leakage-free 5-fold cluster-grouped CV
AUROC **0.88** · structural-specificity **0.90** · non-yeast generalization
**0.80** · matched-pair accuracy **1.00** on the held-out structure-specificity
benchmark.

---

## Use the Colab notebook (no install)

The fastest path for new users: open [`notebooks/RNAPhaseek_colab.ipynb`](notebooks/RNAPhaseek_colab.ipynb) via the badge above. It clones this repo, downloads the weights from Hugging Face Hub, and exposes:

- **Score** — paste a FASTA, get a P(LLPS) CSV
- **GA design** — generate one optimal LLPS RNA + variants
- **DEN design** — generate a mutually-diverse library of LLPS RNAs

Edit `GITHUB_REPO` and `HF_REPO_ID` at the top of the notebook to point at your fork/release.

### Releasing the weights to Hugging Face Hub (one-time, for the project owner)

1. Create a free account at [huggingface.co](https://huggingface.co) and a write token at *Settings → Access Tokens*.
2. `pip install huggingface_hub && huggingface-cli login` (paste the token).
3. `python scripts/release/upload_to_hf.py <hf_user>/rnaphaseek` — creates the model repo and pushes `final_model.pt`, `norm_stats.npz`, `model_card.json`, plus an auto-generated `README.md` model card.

---

## Quick start

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

## Folder layout

```
RNAPhaseek_scripts/
├── rnaphaseek.py          # ← main CLI (score / design / validate). Run from here.
├── paths.py               #   import bootstrap so scripts/ subfolders find each other
├── requirements.txt
│
├── Functions/             # core importable package (model, data, FEGS, biophysics)
├── Data/                  # all corpora, splits, precomputed FEGS/biophysics (large)
├── model/                 # trained checkpoints + per-experiment eval summaries
│   └── strict_eval_v6_production/   # ← THE final accepted model (final_model.pt, model_card.json)
├── report_assets/         # report figures (fig1–fig23)
├── outputs/designs/       # generated candidate RNAs (designed_*.fasta)
├── docs/                  # comprehensive report (PDF) + CLI doc + next-steps
│
└── scripts/               # pipeline scripts, by purpose (all run from project root):
    ├── data_prep/         #   build structural negatives, precompute features, cluster grouping
    ├── training/          #   run_v4 … run_v7, final production re-fit
    ├── generation/        #   GA + DEN de-novo design generators
    ├── analysis/          #   structure-dependence, external validation, KL diagnostics
    ├── reporting/         #   build the PDF report and its figures
    └── legacy/            #   superseded Phase-1 training launchers (kept for provenance)
```

**Run convention:** always invoke from the project root, e.g.
`python scripts/training/run_v6_production.py`. Scripts use root-relative `Data/` and
`model/` paths and import each other through `paths.py`; running from elsewhere will break paths.

---

## Key artifacts

| What | Where |
|---|---|
| Final model + model card | `model/strict_eval_v6_production/{final_model.pt, model_card.json}` |
| Strict positive corpus (1,352) | `Data/raw/multispecies/strict_pool_v5_positives.fasta` |
| Candidate designs (wet-lab panel) | `outputs/designs/designed_ga_v6.fasta`, `designed_den_v6.fasta` |
| Comprehensive report (19 pp) | `docs/RNAPhaseek_Comprehensive_Report.pdf` |
| Reproduce the report | `python scripts/reporting/make_full_report.py` |

## Storage layout (internal lean + external archive)

To keep the internal disk free, the **278 GB regenerable FEGS cache lives on the external
LaCie**, with a full project archive there too:

- **`Data/processed/` is a symlink** → `/Volumes/LaCie/RNAPhaseek_scripts/Data/processed`
  (the precomputed FEGS matrices — large, regenerable).
- Everything else (model, code, raw FASTAs, splits, docs) is **on the internal disk** (~14 GB).

What this means:
- **Using the model — always works, even with the LaCie unplugged.** `rnaphaseek score/design/validate`
  computes FEGS on the fly and never touches the cache.
- **Re-training — needs the LaCie plugged in** (the symlink reads the cached FEGS from it). If the drive
  is absent, regenerate the cache from the FASTAs instead:
  `python scripts/data_prep/precompute_v5_features.py` (and `precompute_v4_features.py`).
- The external also holds a **complete, runnable copy** of the whole project at
  `/Volumes/LaCie/RNAPhaseek_scripts/` (relocatable — paths derive from file location).

> Note: the external is exFAT (not journaled). It's the archive, not the only copy of anything
> irreplaceable — the model, code, and raw FASTAs all live on the internal disk too.

## The model in one paragraph

Trained on the largest strict RNA-self-LLPS corpus that exists (Van Treeck protein-free yeast
self-assembly screen + repeat/G4/riboswitch/RNase-P diversity), with structural hard negatives,
self-complementarity features, CD-HIT cluster-grouped leakage control, and organism-balanced
sampling. It is a strong *yeast* predictor (0.91) with solid transfer to other organisms (0.80);
its one real limit — cross-organism generalization — is a **data-scarcity** constraint of the field,
not a modeling one. Predictions are *model-believed* candidates awaiting experimental validation.
