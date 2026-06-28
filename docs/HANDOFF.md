# RNAPhaseek — Project Handoff

A one-stop onboarding document. Read this end-to-end if you're picking the project
up — it points at every file, every storage location, every published URL, and
every pending task.

**Author / contact**: Amir M. Cheraghali (INSERM) · amirmohammad.cheraghali@inserm.fr
**Last updated**: 2026-06-21

---

## 1. What RNAPhaseek is

A framework for **predicting and de-novo designing RNA sequences that undergo
protein-free liquid–liquid phase separation (LLPS)** — RNAs that condense by
themselves, through RNA–RNA multivalent interactions, in the complete absence of
protein.

To our knowledge it is the first published learned predictor of protein-free
RNA-self-LLPS, and the first to couple such a predictor to de novo RNA-condensate
design.

### Architecture (one paragraph)

RNA-FM foundation backbone (640-dim, last 2 layers fine-tuned) + FEGSTrans
structural adapter (graph-bias attention) + 38-dim biophysics features (MFE,
GC%, G4-potential, self-complementarity) fused into an MLP classification head.
The corpus combines a strictly-curated protein-free RNA-LLPS pool (1,352
positives, 641 negatives, 184 structural hard negatives) with 83 mechanistically
matched G-quadruplex training pairs that close the structure-specificity gap.
Training uses CD-HIT cluster-grouped leakage control and organism-balanced
sampling.

### Model performance (5-fold cluster-grouped CV)

| Metric | Value |
|---|---|
| Overall AUROC | **0.88** |
| Structural-specificity AUROC | **0.90** |
| Non-yeast generalization AUROC | **0.80** |
| Matched-pair accuracy (held-out G-quadruplex benchmark) | **1.00** |

---

## 2. Where everything lives

| Location | Purpose | URL / path |
|---|---|---|
| **GitHub** | All source code, the Colab notebook, manuscript draft, all docs | https://github.com/QuercusCode/RNAPhaseek |
| **Hugging Face Hub** | Model weights (`final_model.pt`, `norm_stats.npz`) | https://huggingface.co/quercuscode/rnaphaseek |
| **Local working dir** | Active development | `~/Documents/RNAPhaseek_scripts/` |
| **LaCie external** | Raw corpus + precomputed features + all training checkpoints | `/Volumes/LaCie/RNAPhaseek_scripts/` |
| **LaCie OLD backup** | Snapshot from 2026-06-12 (pre-architecture-cleanup, 503 GB) | `/Volumes/LaCie/RNAPhaseek_scripts.OLD_20260612/` |
| **Project memory** | Per-milestone decision log | `~/.claude/projects/-Users-synbaiteam-Documents-RNAPhaseek-scripts/memory/*.md` |

The exact local + LaCie file layout is documented in
[`docs/STRUCTURE.md`](STRUCTURE.md). **Read that next** if you need the file map.

---

## 3. Get started in 5 minutes

```bash
# 1. Clone the repo
git clone https://github.com/QuercusCode/RNAPhaseek.git
cd RNAPhaseek

# 2. Set up the conda env (Mac/Linux)
conda create -n rnaphaseek python=3.11 -y
conda activate rnaphaseek
conda install -c bioconda viennarna -y
pip install -r requirements.txt

# 3. Pull weights from Hugging Face
huggingface-cli download quercuscode/rnaphaseek \
    --local-dir model/production \
    --local-dir-use-symlinks False

# 4. Try it
echo -e ">test\nGGGAGGGAGGGAGGGUUUUUUUUUUUUUUU" > /tmp/test.fa
python rnaphaseek.py score /tmp/test.fa -o /tmp/test_scores.csv
cat /tmp/test_scores.csv
```

That's enough for **default inference** (`score` / `design` / `validate`). The
`--uncertainty` mode additionally needs the optional ensemble-member checkpoints
(see §5).

For users who don't want to install anything, the
[**one-click Colab notebook**](../notebooks/RNAPhaseek_colab.ipynb) does the
same thing in the browser.

---

## 4. Using the tool

### CLI (the primary interface)

```bash
# Score RNA sequences (sliding-window applied automatically for >1022 nt)
python rnaphaseek.py score input.fasta -o scores.csv

# Design new high-P(LLPS) sequences
python rnaphaseek.py design --method ga  -o ga.fasta      # one optimal motif + variants
python rnaphaseek.py design --method den -o den.fasta     # diverse library

# Trustworthiness check (is the high score from structure or composition?)
python rnaphaseek.py validate designs.fasta -o trust.csv

# Uncertainty / abstention mode (needs the optional ensemble-member checkpoints)
python rnaphaseek.py score input.fasta --uncertainty -o scores.csv

# Per-window detail for long sequences
python rnaphaseek.py score input.fasta --per-window-out windows.csv -o scores.csv
```

Full CLI reference and option list: [`docs/RNAPHASEEK_CLI.md`](RNAPHASEEK_CLI.md).

### Colab notebook (no-install path)

[`notebooks/RNAPhaseek_colab.ipynb`](../notebooks/RNAPhaseek_colab.ipynb). Open
via the Colab badge in the [README](../README.md). Pulls weights from HF, then
exposes:

- Score sequences (with bar-chart visualization)
- GA / DEN design (with cross-cell sharing — top designs auto-populate the
  visualization cells)
- 2D structure (ViennaRNA + inline SVG + interactive `forna` link)
- 3D structure (Boltz-2 + py3Dmol; ~1–3 min on Colab GPU)

---

## 5. Retraining or re-running experiments

The strict corpus and pre-computed features live in `Data/` (large files may
be archived on external storage; see [`docs/STRUCTURE.md`](STRUCTURE.md)):

- Strict corpus FASTAs: `Data/raw/multispecies/`
- Biophysics + FEGS feature caches: `Data/splits/`

Training scripts: [`scripts/training/`](../scripts/training/). Each launcher
script is self-contained for one experiment. The CLI tool
[`rnaphaseek.py`](../rnaphaseek.py) resolves its default model path to
`model/production/`; the optional ensemble-member checkpoints are looked up
under `model/`. Override the lookup with `--ensemble-from <root>` or
`RNAPHASEEK_ENSEMBLE_ROOT=<root>`.

Completed training runs are organized under `model/` into themed subdirs
(see [`model/README.md`](../model/README.md)).

---

## 6. Project summary

**Architecture**: RNA-FM foundation backbone + FEGSTrans structural adapter +
38-dim biophysics + MLP head. Trained on a strictly-curated protein-free
RNA-LLPS corpus (≈1,400 positives, ≈700 negatives, ≈200 structural hard
negatives) with CD-HIT cluster-grouped leakage control, organism-balanced
sampling, and matched G-quadruplex training pairs that close the
structure-specificity blind spot. The matched-pair pairs are the
breakthrough — three earlier directions (richer intramolecular features, more
data volume, alternative foundation backbone) were tried and didn't help.

---

## 7. Project state — what's done and what's pending

### Done
- ✅ Model trained, evaluated, frozen
- ✅ CLI tool (score / design / validate / uncertainty)
- ✅ One-click Colab notebook with score / GA / DEN / 2D / 3D visualization
- ✅ Code published on GitHub (https://github.com/QuercusCode/RNAPhaseek)
- ✅ Weights published on Hugging Face Hub
- ✅ Manuscript draft (research article + cover letter) in [`docs/`](.)
- ✅ Local + LaCie file layouts organized and documented
- ✅ All historical training runs + failed experiments preserved and catalogued

### Pending — manuscript (for preprint submission to bioRxiv)
- ⬜ Co-author list and affiliations (placeholder at manuscript line 3)
- ⬜ Author Contributions section (placeholder at line ~114)
- ⬜ Funding sources (placeholder at line ~110)
- ⬜ Confirm "no competing interests"
- ⬜ 5 `[verify]` citations in the references list
- ⬜ Generate clean PDF from the markdown source (via pandoc) and submit to bioRxiv

### Pending — wet-lab validation
- ⬜ Synthesize a panel of ~10 RNAs (GA designs + DEN designs + di-shuffled controls + positive/negative reference RNAs)
- ⬜ Standard buffer: 50 mM Tris pH 7.4, 150 mM KCl, 5–10 mM MgCl2, ± 3 mM spermine
- ⬜ Readouts: turbidity (A350) → DIC microscopy → SYTO RNAselect (RNA-in-droplet) → FRAP (liquid vs gel) on top hits
- ⬜ Optional: in vivo extension (yeast or U2OS smFISH ± stress conditions)

### Pending — computational backlog
See [`docs/NEXT_STEPS.md`](NEXT_STEPS.md) for the full list. Highlights:
- ⬜ Wire `species_id` end-to-end (dataset → collate → trainer → model) for multi-species training
- ⬜ Regenerate the FEGS cache with SHA1 sidecar for the multispecies pool (silent-mispairing bug)
- ⬜ Stratified species × label CV split

### Optional polish (not blocking)
- ⬜ `CITATION.cff` in the repo root → enables GitHub's "Cite this repository" button
- ⬜ Tag a GitHub Release (`v1.0.0`) with release notes
- ⬜ HuggingFace Space (Gradio web app) for a no-install alternative to the Colab

---

## 8. Reading list — where to find what

| You want to… | Look at |
|---|---|
| Understand the project at a glance | [`README.md`](../README.md) |
| Understand the full file layout | [`docs/STRUCTURE.md`](STRUCTURE.md) |
| Use the CLI | [`docs/RNAPHASEEK_CLI.md`](RNAPHASEEK_CLI.md) |
| Read the paper draft | [`docs/RNAPhaseek_manuscript.md`](RNAPhaseek_manuscript.md) |
| Read the cover letter | [`docs/RNAPhaseek_cover_letter.md`](RNAPhaseek_cover_letter.md) |
| Get the deep technical/scientific report | [`docs/RNAPhaseek_Comprehensive_Report.pdf`](RNAPhaseek_Comprehensive_Report.pdf) |
| Understand the development history | The memory files at `~/.claude/projects/.../memory/` (in numbered order) |
| See the computational backlog | [`docs/NEXT_STEPS.md`](NEXT_STEPS.md) |
| Find a specific training checkpoint | `/Volumes/LaCie/RNAPhaseek_scripts/model/README.md` |
| Find the corpus + features | `/Volumes/LaCie/RNAPhaseek_scripts/Data/README.md` |
| Re-derive any preprocessing step | The relevant `scripts/data_prep/*.py` script |
| Train a new variant | The relevant `scripts/training/run_*.py` script |

---

## 9. Recommended onboarding (first day)

1. Read this document (you just did)
2. Read [`docs/STRUCTURE.md`](STRUCTURE.md) for the file map
3. Run the [Colab notebook](../notebooks/RNAPhaseek_colab.ipynb) end-to-end —
   gives you a feel for what the tool does
4. Run the CLI locally with the §3 setup — verify your environment works
5. Read the manuscript draft [`docs/RNAPhaseek_manuscript.md`](RNAPhaseek_manuscript.md)
6. Skim the memory files in chronological order (§6) — gives you the design
   rationale
7. Pick a pending item from §7 and ask Amir for context

---

## 10. Contact

- **Maintainer**: Amir M. Cheraghali
- **Email**: amirmohammad.cheraghali@inserm.fr
- **Affiliation**: INSERM, France
- **GitHub**: https://github.com/QuercusCode

Open an issue at https://github.com/QuercusCode/RNAPhaseek/issues for anything
that can be addressed publicly; email for project-specific or unpublished
questions.
