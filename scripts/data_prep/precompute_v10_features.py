"""Regenerate width-38 biophysics + FEGS for the corrected v10 pool (positives 1321,
negatives 649) in FASTA order so rows align positionally. Mirrors precompute_v5_features.
Struct-negatives + synthetic aug reuse the v4 arrays/FEGS (same sequences).

CPU/multiprocessing only — safe to run concurrently with the MPS training.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/data_prep/precompute_v10_features.py
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


def is_yeast_header(h):
    hl = h.lower()
    return any(k in hl for k in ["cerevisiae", "saccharomyces", "vantreeck", "yeast", "scer", "|sc_"])


def main():
    print(f"N_FEATURES={N_FEATURES} | {W} workers")
    POS = ROOT / "Data/raw/multispecies/strict_pool_v10_positives.fasta"
    NEG = ROOT / "Data/raw/multispecies/strict_pool_v10_negatives_all.fasta"

    print("== biophysics (width 38) ==")
    biophys_for(POS, SP / "biophys_v10_pos.npy")
    biophys_for(NEG, SP / "biophys_v10_neg.npy")

    print("== organism labels (is_yeast, pool order pos+neg+structneg) ==")
    # Reuse the curated v5 labels by exact-sequence lookup (kept seqs); header-parse only the new ones.
    def nrm(s):
        return "".join(c for c in s.upper().replace("T", "U") if c in "ACGU")
    v5_seqs = ([s for _, s in read_fasta(str(ROOT / "Data/raw/multispecies/strict_pool_v5_positives.fasta"))]
               + [s for _, s in read_fasta(str(ROOT / "Data/raw/multispecies/strict_pool_v5_negatives_all.fasta"))]
               + [s for _, s in read_fasta(str(ROOT / "Data/raw/multispecies/strict_struct_negatives_v4.fasta"))])
    y5 = np.load(SP / "is_yeast_v5.npy")
    ymap = {nrm(s): bool(y5[i]) for i, s in enumerate(v5_seqs)}
    v10 = (read_fasta(str(POS)) + read_fasta(str(NEG))
           + read_fasta(str(ROOT / "Data/raw/multispecies/strict_struct_negatives_v4.fasta")))
    is_yeast = np.array([ymap.get(nrm(s), is_yeast_header(h)) for h, s in v10], dtype=bool)
    np.save(SP / "is_yeast_v10.npy", is_yeast)
    print(f"  is_yeast_v10.npy {is_yeast.shape}  yeast frac={is_yeast.mean():.3f}  "
          f"(new/header-fallback: {sum(1 for h,s in v10 if nrm(s) not in ymap)})", flush=True)

    print("== FEGS ==")
    for fa, d in [("strict_pool_v10_positives.fasta", "fegs_v10_pos"),
                  ("strict_pool_v10_negatives_all.fasta", "fegs_v10_neg")]:
        t = time.time()
        process_fasta(ROOT / "Data/raw/multispecies" / fa, ROOT / "Data/processed" / d,
                      topk=10, seq_len=1024, overwrite=True, workers=8)
        n = len(list_npz_sorted(str(ROOT / "Data/processed" / d)))
        print(f"  {d}: {n} npz  ({time.time()-t:.0f}s)", flush=True)
    print("DONE precompute_v10_features", flush=True)


if __name__ == "__main__":
    main()
