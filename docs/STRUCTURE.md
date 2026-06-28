# RNAPhaseek — Directory Structure

A reference for where everything lives, across the local repo and any external
archive. **This document is for internal use** (not the user-facing README).

---

## 1. Local working directory

`~/Documents/RNAPhaseek_scripts/` — the active project root.

```
RNAPhaseek_scripts/
├── README.md                  ← public README (user-facing, minimal)
├── LICENSE
├── requirements.txt
├── paths.py                   ← sys.path bootstrap for the scripts/ package
├── rnaphaseek.py              ← main CLI (score / design / validate)
├── .gitignore
│
├── Functions/                 ← core importable package
│   ├── RNAPhaseek/              (model + dataset + trainer + hybrid backbones)
│   ├── RNA_FEGS/                (FEGS structural feature extractor)
│   ├── RNA_biophysical/         (38-dim biophysics features)
│   ├── HMM_augmentation/        (HMM-based data augmentation)
│   ├── data_collection*/        (corpus mining)
│   └── *.py                     (top-level helpers: precompute, evaluate, runner, …)
│
├── scripts/                   ← pipeline scripts (run from project root)
│   ├── analysis/                (benchmarks, uncertainty, longrange, KL diagnostics)
│   ├── data_prep/               (build pools, structural negatives, cluster grouping)
│   ├── training/                (training scripts)
│   ├── generation/              (GA + DEN de-novo design)
│   ├── reporting/               (PDF report + figure builders)
│   ├── release/                 (HF Hub upload, etc.)
│   └── legacy/                  (superseded launchers)
│
├── Data/                      ← thin proxy; bulk corpora may live on external storage
│
├── model/                     ← model checkpoints (see model/README.md)
│   └── production/              (final_model.pt, norm_stats.npz, README.md,
│                                 model_card.json — weights gitignored)
│
├── notebooks/                 ← one-click Colab notebook
│   └── RNAPhaseek_colab.ipynb
│
├── docs/                      ← documentation, reports, manuscript
│   ├── STRUCTURE.md             ← this file
│   ├── RNAPHASEEK_CLI.md        ← CLI reference
│   ├── NEXT_STEPS.md            ← computational backlog
│   ├── RNAPhaseek_manuscript.md ← the paper draft
│   ├── RNAPhaseek_cover_letter.md
│   ├── RNAPhaseek_*Report.pdf   ← reproducible reports
│   ├── *.docx                    ← exported manuscript / cover letter
│   └── figures/                  ← additional figure assets
│
├── outputs/                   ← run-time outputs (logs, JSONs, designs)
│   ├── designs/                 ← FASTA libraries (referenced by README)
│   └── *.log / *.json           ← analysis run artifacts
│
├── report_assets/             ← PNG figures used by make_report_pdf.py
│
└── archive/                   ← gitignored; old setup logs + rollback .bak files
```

---

## 2. External archive (optional)

Heavy artifacts (large training corpora, historical run dumps, ensemble-member
checkpoints) can live on an external drive when local space is tight.

```
RNAPhaseek_scripts/
├── README.md
├── Data/
│   ├── raw/             (FASTA corpora + download logs)
│   ├── splits/          (precomputed biophysics + FEGS features)
│   └── processed/       (runtime cache)
│
└── model/
    ├── README.md
    ├── ensemble/        ← optional ensemble-member checkpoints
    ├── development/     ← internal experiments
    └── historical/      ← archived legacy runs
```

---

## 3. How the CLI bridges the two

`rnaphaseek.py` looks for the optional ensemble-member checkpoints under the
local `model/` directory first, then falls back to an external archive path.

```python
DEFAULT_ENSEMBLE_ROOT = "<local model/ or external fallback>"
```

Override the lookup with `--ensemble-from <root>` or set
`RNAPHASEEK_ENSEMBLE_ROOT=<root>` in the environment.

For default scoring of any sequence (short or long), **no external storage is
needed** — the local `model/production/` is sufficient. The ensemble members
are only used for `--uncertainty` mode.

---

## 4. What's where, by topic

| If you need… | Look in |
|---|---|
| The CLI source | `~/Documents/.../rnaphaseek.py` |
| Model weights (offline) | `~/Documents/.../model/production/` |
| Optional ensemble weights (for `--uncertainty`) | `~/Documents/.../model/` subdirectories |
| The training corpus | `Data/raw/multispecies/` |
| Precomputed features for retraining | `Data/splits/` |
| The manuscript draft | `docs/RNAPhaseek_manuscript.md` |
| The Colab notebook | `notebooks/RNAPhaseek_colab.ipynb` |
| Report figures (PNG) | `report_assets/` |
| Per-experiment results | `model/development/`, `model/historical/` (if archived externally) |
