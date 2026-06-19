"""Upload the RNAPhaseek production weights to the Hugging Face Hub so the Colab notebook
can pull them at runtime. Run once after creating the HF repo.

Prereqs:
  pip install huggingface_hub
  huggingface-cli login        # paste a write token from huggingface.co/settings/tokens
  (or set HF_TOKEN in the env)

Usage:
  python scripts/release/upload_to_hf.py <hf_repo_id>
  # e.g.
  python scripts/release/upload_to_hf.py quercuscode/rnaphaseek

Creates the repo if missing (public by default; pass --private to make it private)
and pushes:
  final_model.pt    (the trained checkpoint)
  norm_stats.npz    (per-model biophysics mean/std)
  model_card.json   (machine-readable metadata, includes internal version lineage)
  README.md         (the HF model card; auto-generated below)
"""
import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

from huggingface_hub import HfApi, create_repo

LOCAL_MODEL_DIR = Path("model/strict_eval_v13_production")
FILES = ["final_model.pt", "norm_stats.npz", "model_card.json"]


def build_hf_readme(repo_id: str, card: dict) -> str:
    return dedent(f"""\
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
        - `model_card.json` — machine-readable training/eval metadata (includes internal training lineage for reproducibility)

        ## Architecture

        Three streams fused in a single MLP head:

        1. **RNA-FM backbone** (`multimolecule/rnafm`, 640-dim, last 2 layers fine-tuned)
        2. **FEGSTrans adapter** that pools backbone embeddings with a structural FEGS bias
        3. **38 biophysical features** (MFE, GC%, G4-potential, self-complementarity, etc.)

        Trained on a strict protein-free RNA-LLPS corpus (1,396 positives / 678 negatives
        / 184 structural negatives) plus 83 de-leaked matched training pairs that teach
        the model the free-vs-sequestered G-tract distinction — closing the
        structure-specificity blind spot of earlier training recipes.

        ## Headline numbers

        | Metric | Value |
        |---|---|
        | 5-fold cluster-grouped CV AUROC          | **0.875** |
        | Structural-specificity AUROC             | **0.897** |
        | Non-yeast generalization AUROC           | **0.803** |
        | Matched-pair accuracy (held-out)         | **1.00**  |
        | Hard-18 held-out AUROC                   | **0.812** |
        | Mean margin (pos − neg) on matched pairs | **+0.130** |

        ## Programmatic use

        ```python
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id="{repo_id}", filename="final_model.pt")
        norm_path  = hf_hub_download(repo_id="{repo_id}", filename="norm_stats.npz")

        # then load with the project code (see the GitHub repo):
        from rnaphaseek import RNAPhaseekScorer, read_fasta
        scorer = RNAPhaseekScorer(model_path=model_path, norm_path=norm_path)
        probs  = scorer.score(["GGGAGGGAGGGAGGGUUUUUUUUUUUUUUU"])
        print(probs)
        ```

        ## Citation

        If you use RNAPhaseek, please cite the accompanying manuscript (Pandi et al.).

        ## License

        MIT for the code; weights released for academic use under the same license.
        """)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo_id", help="HF model repo, e.g. quercuscode/rnaphaseek")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    ap.add_argument("--no-create", action="store_true", help="don't try to create the repo; assume it exists")
    args = ap.parse_args()

    if not LOCAL_MODEL_DIR.exists():
        sys.exit(f"missing {LOCAL_MODEL_DIR} — run this from the repo root")

    for f in FILES:
        if not (LOCAL_MODEL_DIR / f).exists():
            sys.exit(f"missing {LOCAL_MODEL_DIR / f}")

    if not args.no_create:
        print(f"[upload] create_repo {args.repo_id} (private={args.private}) ...")
        create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    card = json.load(open(LOCAL_MODEL_DIR / "model_card.json"))
    readme = build_hf_readme(args.repo_id, card)
    readme_path = LOCAL_MODEL_DIR / "README.md"
    readme_path.write_text(readme)

    api = HfApi()
    print(f"[upload] pushing {len(FILES) + 1} files to {args.repo_id} ...")
    for f in FILES + ["README.md"]:
        src = LOCAL_MODEL_DIR / f
        print(f"  uploading {f} ({src.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=f,
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(f"\nDone. View the release at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
