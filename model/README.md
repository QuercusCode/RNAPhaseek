---
license: mit
tags:
  - rna
  - llps
  - phase-separation
  - rna-fm
  - biology
library_name: pytorch
---

# RNAPhaseek

Predicts the probability that an RNA sequence undergoes **protein-free
liquid–liquid phase separation (LLPS)**, and powers a de-novo generator
for new LLPS-prone RNAs.

## Try it (no install)

Open the Colab notebook from the project's GitHub repo for one-click
scoring and de-novo design.

## What's in this repo

- `final_model.pt` — RNA-FM + FEGSTrans adapter + 38-dim biophysics + MLP head, 426 MB
- `norm_stats.npz` — biophysics feature mean/std (must accompany the checkpoint)

## Architecture

Three streams fused in a single MLP head:

1. **RNA-FM backbone** (`multimolecule/rnafm`, 640-dim, last 2 layers fine-tuned)
2. **FEGSTrans adapter** that pools backbone embeddings with a structural FEGS bias
3. **38 biophysical features** (MFE, GC%, G4-potential, self-complementarity, etc.)

Trained on a strict protein-free RNA-LLPS corpus (positives + negatives +
structural hard negatives) plus matched training pairs that teach the model
the free-vs-sequestered G-tract distinction — closing the structure-specificity
blind spot of earlier training recipes.

## Headline numbers

| Metric | Value |
|---|---|
| 5-fold cluster-grouped CV AUROC          | **0.88** |
| Structural-specificity AUROC             | **0.90** |
| Non-yeast generalization AUROC           | **0.80** |
| Matched-pair accuracy (held-out)         | **1.00** |

## Programmatic use

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="quercuscode/rnaphaseek", filename="final_model.pt")
norm_path  = hf_hub_download(repo_id="quercuscode/rnaphaseek", filename="norm_stats.npz")

# then load with the project code (see the GitHub repo):
from rnaphaseek import RNAPhaseekScorer, read_fasta
scorer = RNAPhaseekScorer(model_path=model_path, norm_path=norm_path)
probs  = scorer.score(["GGGAGGGAGGGAGGGUUUUUUUUUUUUUUU"])
print(probs)
```

## Citation

If you use RNAPhaseek, please cite the accompanying manuscript (Cheraghali et al.).

## License

MIT for the code; weights released for academic use under the same license.
