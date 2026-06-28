# RNAPhaseek CLI

Command-line tool for **RNA-self-LLPS prediction and de novo design**, wrapping the
RNAPhaseek model (`model/`) and the GA/DEN generators.

Run with the project conda env:

```bash
PY=/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python
$PY rnaphaseek.py <command> ...
```

The model is RNA-FM + FEGS adapter + 38-dim biophysics, organism-balanced, leakage-free CV
AUROC **0.88** (non-yeast 0.80, structural-specificity 0.90). Predictions are *model-believed*
candidates, not experimentally confirmed.

---

## `score` — predict P(LLPS) for each RNA

```bash
$PY rnaphaseek.py score input.fasta -o scores.csv      # or omit -o for stdout
$PY rnaphaseek.py score input.fasta -t 0.5             # call threshold
cat input.fasta | $PY rnaphaseek.py score -            # stdin
```
Output CSV: `id, length, GC_percent, P_LLPS, model, call@<t>`. Sequences may be RNA or DNA
(T→U handled), any length. Sequences longer than 1022 nt are scored by sliding-window
inference (tile into overlapping 1022-nt windows, score each, take the highest
per-window score). Add `--per-window-out PATH` to emit per-window scores.

Example:
```
id,length,GC_percent,P_LLPS,model,call@0.5
ga_design_0,200,44.5,0.9962,model,LLPS
random_control,200,59.0,0.0226,model,no
CAG_repeat,180,66.7,0.9698,model,LLPS
```

## `design` — generate de novo phase-separating RNA

```bash
$PY rnaphaseek.py design --method ga  --n 10 --length 200 -o designs.fasta   # one optimal motif + variants
$PY rnaphaseek.py design --method den --length 200 -o designs.fasta          # diverse library
```
- **ga** — genetic algorithm optimizing the model's P(LLPS) through the full pipeline.
  Most structure-grounded (single motif family). Tunable: `--generations`, `--seed`.
- **den** — Deep Exploration Network (diversity-penalized); a diverse library of distinct
  scaffolds, still as structure-dependent as real LLPS positives.

## `validate` — trustworthiness (structure-dependence)

```bash
$PY rnaphaseek.py validate designs.fasta -o trust.csv -k 3
```
Scores each sequence vs `k` composition-matched scrambles. `Delta = P(design) − P(scramble)`.
**Delta > 0** ⇒ the score is *structure-driven* (trustworthy); **Delta ≈ 0** ⇒ composition-driven
(a real-repeat RNA scores the same scrambled — legitimate for repeats, a red flag for a design).

Output CSV: `id, P_design, P_scramble_mean, Delta, verdict`.

---

## Notes
- Default model: `--model model/production/final_model.pt --norm model/production/norm_stats.npz`
  (override with `--model` / `--norm` to use a compatible checkpoint).
- Diagnostics print to **stderr**; CSV/FASTA results to **stdout**, so piping is clean.
- The optional `--uncertainty` mode runs a 4-checkpoint ensemble and reports per-sequence
  standard deviation across members + an `ABSTAIN` flag for out-of-distribution inputs.
