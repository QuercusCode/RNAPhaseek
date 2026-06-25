# RNAPhaseek model weights

Self-contained model directory for RNAPhaseek. All four checkpoints needed for
the production CLI (`rnaphaseek.py`) live here — no LaCie drive required.

`rnaphaseek.py` auto-detects this directory as the ensemble root (no env var
needed) as long as the four subdirectories below exist. If they're missing it
falls back to the LaCie archive path.

## Layout

```
model/
├── README.md                          (this file)
│
├── production/                        # v13 — DEFAULT scorer (sequences <= 1022 nt)
│   ├── final_model.pt                 # 426 MB  — production weights
│   ├── norm_stats.npz                 # biophysics feature mean/std
│   ├── model_card.json                # full provenance (v13 promotion 2026-06-19)
│   ├── RESULT.md                      # held-out benchmark vs v6
│   ├── README.md                      # HuggingFace-style model card
│   └── den_100nt_summary.json
│
├── strict_eval_v6_production/         # v6 — archived predecessor / ensemble member
│   ├── final_model.pt                 # 426 MB
│   ├── norm_stats.npz
│   ├── model_card.json
│   ├── train.log
│   ├── den_v6.log, den_v6_summary.json
│   ├── ga_v6.log
│   └── design_structure_dependence.json
│
├── strict_eval_v6_orgbalanced/        # v6 organism-balanced — ensemble member
│   ├── final_model.pt                 # 426 MB
│   ├── norm_stats.npz
│   ├── eval_summary.json
│   └── train.log
│
└── strict_eval_v7_mil/                # MIL attention pooler for sequences > 1022 nt
    ├── final_model.pt                 # 426 MB
    ├── norm_stats.npz
    ├── eval_summary.json
    └── train.log
```

Total: ~1.7 GB on disk.

## Which model gets used when

| CLI invocation | Models loaded |
|---|---|
| `rnaphaseek score in.fa` | `production/` only (v13, single model, <=1022 nt) |
| `rnaphaseek score in.fa --uncertainty` | `production/` + `strict_eval_v6_production/` + `strict_eval_v6_orgbalanced/` (3-model ensemble) |
| `rnaphaseek score in.fa --long-model mil` | `production/` for short, `strict_eval_v7_mil/` for >1022 nt |
| `rnaphaseek score in.fa --uncertainty --long-model mil` | All four models (full ensemble + MIL routing) |

## Provenance

- **v13** (PROMOTED 2026-06-19): RNA-FM + FEGSTrans + 38-dim biophysics. Trained on v5/v6 corpus + 83 de-leaked matched training pairs. Closes the structure-specificity gap (matched-pair accuracy 1.00 vs v6 0.67, hard-18 AUROC 0.812 vs 0.612) with no general-corpus regression.
- **v6_production**: previous production model. Same architecture as v13, trained without matched pairs. Kept for ensemble disagreement / uncertainty.
- **v6_orgbalanced**: v6 trained with organism-balanced sampler. Ensemble member that improves non-yeast generalization.
- **v7_mil**: full-sequence attention-MIL variant. Tiles RNAs into <=1022 nt windows (stride 512), encodes each with RNA-FM, attention-pools over windows. Only triggered for sequences > 1022 nt via `--long-model mil`.

## Git

`*.pt` and `*.npz` are gitignored — weights never get pushed to GitHub. They're hosted on Hugging Face Hub (`quercuscode/rnaphaseek`). Copy this folder manually when transferring between computers.

## Transferring to another computer

1. Copy the whole `model/` directory across (1.7 GB).
2. That's it — `rnaphaseek.py` finds it automatically. No env var, no LaCie mount.

If you only want the production model (single-model scoring, no uncertainty, no long sequences), copy `model/production/` alone (426 MB) — the CLI runs fine, the ensemble/MIL flags will surface a clear error pointing you here.
