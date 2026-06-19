"""
Regenerate width-38 biophysics + FEGS for the FULL v5 pool (positives 1352, real
negatives 641) in FASTA order so rows align positionally. Struct-negatives (184) and
synthetic aug (147) reuse the v4 width-38 arrays + FEGS dirs (same sequences).

CPU/multiprocessing only — safe to run concurrently with the MPS cluster-CV training.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python precompute_v5_features.py
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
_EXT = RNABiophysicalExtractor(normalize=False)


def _one(seq):
    return _EXT._compute_one(seq).astype(np.float32)


def biophys_for(fasta, out_npy, workers=5):
    recs = read_fasta(str(fasta)); seqs = [s for _, s in recs]
    t = time.time()
    with Pool(workers) as p:
        mat = np.stack(p.map(_one, seqs, chunksize=4)).astype(np.float32)
    assert mat.shape == (len(seqs), N_FEATURES), mat.shape
    np.save(out_npy, mat)
    print(f"  {Path(out_npy).name:26s} {mat.shape}  ({time.time()-t:.0f}s)", flush=True)


def main():
    print(f"N_FEATURES={N_FEATURES}")
    jobs = [
        (ROOT / "Data/raw/multispecies/strict_pool_v5_positives.fasta",     SP / "biophys_v5_pos.npy"),
        (ROOT / "Data/raw/multispecies/strict_pool_v5_negatives_all.fasta", SP / "biophys_v5_neg.npy"),
    ]
    print("== biophysics (width 38) ==")
    for fa, out in jobs:
        biophys_for(fa, out)

    print("== FEGS ==")
    for fa, d in [("strict_pool_v5_positives.fasta", "fegs_v5_pos"),
                  ("strict_pool_v5_negatives_all.fasta", "fegs_v5_neg")]:
        t = time.time()
        process_fasta(ROOT / "Data/raw/multispecies" / fa, ROOT / "Data/processed" / d,
                      topk=10, seq_len=1024, overwrite=True, workers=4)
        n = len(list_npz_sorted(str(ROOT / "Data/processed" / d)))
        print(f"  {d}: {n} npz  ({time.time()-t:.0f}s)", flush=True)
    print("DONE precompute_v5_features", flush=True)


if __name__ == "__main__":
    main()
