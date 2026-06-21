# RNAPhaseek — Directory Structure

A reference for where everything lives, across the internal disk and the LaCie
external drive. **This document is for internal use** (not the user-facing README).

---

## 1. Local working directory

`~/Documents/RNAPhaseek_scripts/` — the active project root, ~430 MB total.

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
│   ├── training/                (per-version training scripts)
│   ├── generation/              (GA + DEN de-novo design)
│   ├── reporting/               (PDF report + figure builders)
│   ├── release/                 (HF Hub upload, etc.)
│   ├── legacy/                  (superseded Phase-1 launchers)
│   ├── split_to_external.sh     (one-time setup; archived in archive/)
│   └── deleak_structnegs_v4.sh / queue_v5_training.sh
│
├── Data/                      ← thin proxy; bulk corpora live on LaCie
│   └── processed/  → /Volumes/LaCie/RNAPhaseek_scripts/Data/processed  (symlink)
│
├── model/                     ← production checkpoint only (local)
│   └── production/              (final_model.pt, norm_stats.npz, README.md,
│                                 model_card.json, RESULT.md — last 3 gitignored)
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
    ├── rnaphaseek.py.pre-extmove.bak
    └── split_to_external.log
```

---

## 2. External archive (LaCie)

`/Volumes/LaCie/RNAPhaseek_scripts/` — heavy artifacts that don't fit locally.

```
RNAPhaseek_scripts/
├── README.md          ← LaCie archive overview
├── Data/
│   ├── README.md
│   ├── raw/             (FASTA corpora + download logs)
│   ├── splits/          (precomputed biophysics + FEGS features)
│   └── processed/       (runtime cache; symlinked from local Data/processed)
│
└── model/
    ├── README.md
    │
    ├── production_ensemble/   ← the 3 checkpoints rnaphaseek.py needs
    │   ├── strict_eval_v6_production/
    │   ├── strict_eval_v6_orgbalanced/
    │   └── strict_eval_v7_mil/
    │
    ├── strict_eval_v6_production    → symlink to production_ensemble/...
    ├── strict_eval_v6_orgbalanced   → symlink (preserves hard-coded paths)
    ├── strict_eval_v7_mil           → symlink
    │
    ├── development/           ← v8 through v16 + ERNIE experiments
    ├── historical/            ← strict / strict_eval / v2aug–v6 / phase1
    ├── failed_experiments/    ← fullseq_*_failed (documented)
    └── pre_v3_legacy/         ← hybrid_*, rna_phaseek_*, BPE artifacts
```

### Sibling backup

`/Volumes/LaCie/RNAPhaseek_scripts.OLD_20260612/` — 503 GB snapshot from 2026-06-12,
before the architecture cleanup. See `ARCHIVE_NOTE.md` inside it for details.
Not referenced by current code; kept as insurance against needing the raw upstream Data.

---

## 3. How the CLI bridges the two

`rnaphaseek.py` looks for the ensemble checkpoints on LaCie via:

```python
DEFAULT_ENSEMBLE_ROOT = "/Volumes/LaCie/RNAPhaseek_scripts/model"
```

The 3 ensemble member dirs (`strict_eval_v6_production`, `strict_eval_v6_orgbalanced`,
`strict_eval_v7_mil`) still resolve at this path thanks to top-level symlinks
pointing into `production_ensemble/`. Override the lookup with `--ensemble-from <root>`
or set `RNAPHASEEK_ENSEMBLE_ROOT=<root>` in the environment.

For inference of the production model alone, **no LaCie access is needed** — the
local `model/production/` is sufficient.

---

## 4. What's where, by topic

| If you need… | Look in |
|---|---|
| The CLI source | `~/Documents/.../rnaphaseek.py` |
| Production weights (offline) | `~/Documents/.../model/production/` |
| Ensemble weights (for `--uncertainty`) | `/Volumes/LaCie/.../model/production_ensemble/` |
| MIL weights (for `--long-model mil`) | `/Volumes/LaCie/.../model/production_ensemble/strict_eval_v7_mil/` |
| The strict corpus | `/Volumes/LaCie/.../Data/raw/multispecies/` |
| Precomputed features for retraining | `/Volumes/LaCie/.../Data/splits/` |
| The manuscript draft | `~/Documents/.../docs/RNAPhaseek_manuscript.md` |
| The Colab notebook | `~/Documents/.../notebooks/RNAPhaseek_colab.ipynb` |
| Report figures (PNG) | `~/Documents/.../report_assets/` |
| Per-experiment results (CV summaries, logs) | `/Volumes/LaCie/.../model/{development,historical}/<run>/` |
| Old backup (just in case) | `/Volumes/LaCie/RNAPhaseek_scripts.OLD_20260612/` |
