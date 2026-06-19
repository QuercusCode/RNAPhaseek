"""
Master download script — collects all new LLPS-positive databases.

Runs in order:
  1. RPS 2.0          (~517 reviewed, reliable)
  2. Parker SG        (~2,457 enriched genes, requires Ensembl fetch)
  3. smOOPs           (~3,060 mouse genes × ortholog mapping)
  4. RNAPhaSep        (~325 curated, server intermittent)
  5. G4RNA            (~334 G4-forming, use with caution)

After all downloads:
  6. Merge all new FASTA files + existing positives
  7. Run CD-HIT-EST at 90% identity to remove duplicates
  8. Report final unique-positive count

Prerequisites (install on new machine):
    pip install requests openpyxl beautifulsoup4
    conda install -c bioconda cd-hit   # or: brew install cd-hit

Run:
    python -m Functions.data_collection.download_all [--out-dir Data/raw]
    python -m Functions.data_collection.download_all --skip-g4rna --skip-smoops
"""

import argparse
import os
import subprocess
import sys
import time


def _run_module(module: str, out_dir: str) -> bool:
    """Run a sibling download module. Returns True on success."""
    cmd = [sys.executable, "-m", module, "--out-dir", out_dir]
    print(f"\n{'='*60}", flush=True)
    ret = subprocess.run(cmd, check=False)
    return ret.returncode == 0


def merge_fastas(fasta_files: list[str], merged_path: str) -> int:
    """Concatenate FASTA files; return total sequence count."""
    count = 0
    with open(merged_path, "w") as out:
        for fasta in fasta_files:
            if not os.path.exists(fasta):
                print(f"  [skip] {fasta} not found", flush=True)
                continue
            with open(fasta) as fh:
                for line in fh:
                    out.write(line)
                    if line.startswith(">"):
                        count += 1
    return count


def run_cdhit(merged: str, deduplicated: str, identity: float = 0.90) -> int:
    """Run CD-HIT-EST. Returns number of unique clusters (≈ sequences)."""
    cmd = [
        "cd-hit-est",
        "-i", merged,
        "-o", deduplicated,
        "-c", str(identity),
        "-n", "8",          # word length for 0.90 identity
        "-T", "0",          # use all available threads
        "-M", "16000",      # 16 GB memory limit
        "-d", "0",          # no description truncation
    ]
    print(f"\nRunning CD-HIT-EST (identity={identity}) …", flush=True)
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print("  [WARN] CD-HIT-EST failed — is it installed?", flush=True)
        print("    Install: conda install -c bioconda cd-hit", flush=True)
        print("    or:       brew install cd-hit", flush=True)
        return -1

    # Count clusters from the .clstr file
    clstr_path = deduplicated + ".clstr"
    if os.path.exists(clstr_path):
        n_clusters = sum(1 for l in open(clstr_path) if l.startswith(">Cluster"))
        return n_clusters
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all LLPS databases")
    parser.add_argument("--out-dir",      default="Data/raw")
    parser.add_argument("--skip-rps2",    action="store_true")
    parser.add_argument("--skip-parker",  action="store_true")
    parser.add_argument("--skip-smoops",  action="store_true")
    parser.add_argument("--skip-rnaphase",action="store_true")
    parser.add_argument("--skip-g4rna",   action="store_true")
    parser.add_argument("--skip-cdhit",   action="store_true",
                        help="Skip CD-HIT-EST deduplication step")
    parser.add_argument("--existing-positives", default="",
                        help="Path to your existing positives FASTA to include in merge")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = "Functions.data_collection"

    t0 = time.time()

    if not args.skip_rps2:
        _run_module(f"{base}.download_rps2",   args.out_dir)

    if not args.skip_parker:
        _run_module(f"{base}.download_parker_sg", args.out_dir)

    if not args.skip_smoops:
        _run_module(f"{base}.download_smoops",    args.out_dir)

    if not args.skip_rnaphase:
        _run_module(f"{base}.download_rnaphase",  args.out_dir)

    if not args.skip_g4rna:
        _run_module(f"{base}.download_g4rna",     args.out_dir)

    # ── Merge ─────────────────────────────────────────────────────────────────
    new_fastas = [
        os.path.join(args.out_dir, "rps2_positives.fasta"),
        os.path.join(args.out_dir, "parker_sg_positives.fasta"),
        os.path.join(args.out_dir, "smoops_positives.fasta"),
        os.path.join(args.out_dir, "rnaphase_positives.fasta"),
        os.path.join(args.out_dir, "g4rna_positives.fasta"),
    ]
    if args.existing_positives:
        new_fastas.insert(0, args.existing_positives)

    merged_path = os.path.join(args.out_dir, "all_positives_raw.fasta")
    total = merge_fastas(new_fastas, merged_path)
    print(f"\n{'='*60}", flush=True)
    print(f"Merged {total} sequences → {merged_path}", flush=True)

    # ── CD-HIT-EST deduplication ──────────────────────────────────────────────
    if not args.skip_cdhit:
        dedup_path = os.path.join(args.out_dir, "all_positives_dedup.fasta")
        n_unique   = run_cdhit(merged_path, dedup_path)
        if n_unique > 0:
            print(f"\nAfter 90% deduplication: {n_unique} unique sequences → {dedup_path}")
        else:
            print(f"\nDeduplication skipped. Use {merged_path} directly.")
    else:
        print("\nCD-HIT-EST step skipped (--skip-cdhit).")
        print(f"Run manually: cd-hit-est -i {merged_path} -o {args.out_dir}/all_positives_dedup.fasta "
              f"-c 0.90 -n 8")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print("\nNext steps:")
    print("  1. Review Data/raw/all_positives_dedup.fasta for quality")
    print("  2. Run: python -m Functions.prepare_negatives  (to regenerate negatives)")
    print("  3. Run: python -m Functions.build_dataset")
    print("  4. Run: python -m Functions.precompute_fegs")
    print("  5. Run: python -m Functions.precompute_biophysical")
    print("  6. Run: python -m Functions.runner  (train)")


if __name__ == "__main__":
    main()
