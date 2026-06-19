"""Regenerate width-38 biophysics + FEGS for the v11 pool (v5 + 18 additions) in FASTA order so
rows align positionally. Mirrors precompute_v10_features. is_yeast is NOT written here — the
canonical generator (with struct-neg parent inheritance) is cluster_groups_v11.py.

CPU/multiprocessing only — safe to run concurrently with MPS training. FEGS is ~10ms/seq.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/data_prep/precompute_v11_features.py
"""
import os, sys, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.getcwd())
from Functions.RNA_biophysical.biophysical_features import RNABiophysicalExtractor, N_FEATURES
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta
from Functions.precompute_fegs import process_fasta
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted

ROOT = Path(__file__).resolve().parents[2]
SP = ROOT / "Data/splits"
W = max(4, os.cpu_count() - 4)
_EXT = RNABiophysicalExtractor(normalize=False)


def _one(seq):
    return _EXT._compute_one(seq).astype(np.float32)


def biophys_for(fasta, out_npy):
    seqs = [s for _, s in read_fasta(str(fasta))]
    t = time.time()
    with Pool(W) as p:
        mat = np.stack(p.map(_one, seqs, chunksize=4)).astype(np.float32)
    assert mat.shape == (len(seqs), N_FEATURES), mat.shape
    np.save(out_npy, mat)
    print(f"  {Path(out_npy).name:26s} {mat.shape}  ({time.time()-t:.0f}s)", flush=True)


def main():
    print(f"N_FEATURES={N_FEATURES} | {W} workers")
    POS = ROOT / "Data/raw/multispecies/strict_pool_v11_positives.fasta"
    NEG = ROOT / "Data/raw/multispecies/strict_pool_v11_negatives_all.fasta"

    print("== biophysics (width 38) ==")
    biophys_for(POS, SP / "biophys_v11_pos.npy")
    biophys_for(NEG, SP / "biophys_v11_neg.npy")

    print("== FEGS ==")
    for fa, d in [("strict_pool_v11_positives.fasta", "fegs_v11_pos"),
                  ("strict_pool_v11_negatives_all.fasta", "fegs_v11_neg")]:
        t = time.time()
        process_fasta(ROOT / "Data/raw/multispecies" / fa, ROOT / "Data/processed" / d,
                      topk=10, seq_len=1024, overwrite=True, workers=8)
        n = len(list_npz_sorted(str(ROOT / "Data/processed" / d)))
        print(f"  {d}: {n} npz  ({time.time()-t:.0f}s)", flush=True)
    print("DONE precompute_v11_features", flush=True)


if __name__ == "__main__":
    main()
