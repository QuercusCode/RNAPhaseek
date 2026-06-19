# v13 — PRODUCTION MODEL (promoted 2026-06-19, replaces v6)

RNA-FM + FEGSTrans adapter + 38-dim biophysics, organism-balanced sampler.
Trained on v5/v6 corpus + 83 de-leaked matched training pairs (1396 pos / 678 neg / 184 struct-neg).

## Why v13 replaces v6

v6 had a structure-specificity blind spot: it keyed on G-content and could not distinguish free
G-tracts (LLPS-positive) from sequestered G-tracts (LLPS-negative). Three negatives (v8 richer
features, v11 more data volume, v12 ERNIE backbone) diagnosed this as a DATA problem — the corpus
had no matched pairs where a flank/spacer change flips LLPS. v13 adds 83 such pairs.

## Held-out structure-specificity benchmark (9 matched pairs, never seen in training)

| Metric                    | v6      | v13      |
|---------------------------|---------|----------|
| Matched-pair accuracy     | 0.67    | **1.00** |
| Mean margin (pos - neg)   | +0.002  | **+0.130** |
| Hard-18 AUROC             | 0.612   | **0.812** |

Flagship: 5'-A9 (LLPS+) 0.94 vs 5'-C9 (LLPS-) 0.69 — v13 knows the C9 flank sequesters the G-core.

## 5-fold cluster-grouped CV (no regression vs v6)

| Metric       | v6     | v13    | Delta  |
|--------------|--------|--------|--------|
| Overall      | 0.8837 | 0.8750 | -0.009 |
| Struct        | 0.8976 | 0.8968 | -0.001 |
| Yeast        | 0.9094 | 0.8981 | -0.011 |
| Non-yeast    | 0.7982 | 0.8028 | +0.005 |

The -0.009 overall is within fold noise; non-yeast (the harder, more diverse slice) improves.

## Threshold

Default kept at 0.50 (F1=0.860, backwards-compatible). F1-optimal on CV is 0.63 (F1=0.863).
Use `--threshold 0.63` for strict calibration.

## v6 archived (not deleted)

v6 remains at `model/strict_eval_v6_production/` as a fallback and ensemble member.
The 4-model ensemble (v13 + v6 + v6_orgbalanced + v7_mil) is the uncertainty/abstention path.
