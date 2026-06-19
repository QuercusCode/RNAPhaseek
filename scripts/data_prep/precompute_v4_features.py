"""
Recompute biophysical features at WIDTH 38 (adds Block-5 self-complementarity,
indices 33-37) for every v4 pool component, and FEGS for the new structural
negatives. v3 width-33 arrays are left intact (separate v4_ filenames) so the
v3 model stays reproducible.

Outputs:
  Data/splits/biophys_v4_pos.npy        (427, 38)
  Data/splits/biophys_v4_neg.npy        (636, 38)
  Data/splits/biophys_v4_synth.npy      (147, 38)
  Data/splits/biophys_v4_structneg.npy  (184, 38)
  Data/processed/fegs_struct_neg_v4/    (184 npz)  -- FEGS for structural negatives
  Data/splits/biophys_v4_canary.json    -- determinism canary for score_external

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python precompute_v4_features.py
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.getcwd())
from Functions.RNA_biophysical.biophysical_features import RNABiophysicalExtractor, N_FEATURES
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta

ROOT = Path(__file__).resolve().parents[2]
_EXT = RNABiophysicalExtractor(normalize=False)


def _one(seq):
    return _EXT._compute_one(seq).astype(np.float32)


def biophys_for(fasta, out_npy, workers=6):
    recs = read_fasta(str(fasta))
    seqs = [s for _, s in recs]
    t = time.time()
    with Pool(workers) as pool:
        mat = np.stack(pool.map(_one, seqs, chunksize=4)).astype(np.float32)
    assert mat.shape == (len(seqs), N_FEATURES), f"{out_npy}: {mat.shape} != ({len(seqs)},{N_FEATURES})"
    np.save(out_npy, mat)
    print(f"  {Path(out_npy).name:32s} {mat.shape}  ({time.time()-t:.0f}s)", flush=True)
    return mat


def main():
    print(f"N_FEATURES = {N_FEATURES} (expect 38)")
    SP = ROOT / "Data/splits"

    jobs = [
        ("Data/raw/multispecies/strict_pool_v3_positives.fasta",     SP / "biophys_v4_pos.npy"),
        ("Data/raw/multispecies/strict_pool_v3_negatives_all.fasta", SP / "biophys_v4_neg.npy"),
        ("Data/raw/multispecies/synthetic_train.fasta",              SP / "biophys_v4_synth.npy"),
        ("Data/raw/multispecies/strict_struct_negatives_v4.fasta",   SP / "biophys_v4_structneg.npy"),
    ]
    print("== biophysical features (width 38) ==")
    for fa, out in jobs:
        biophys_for(ROOT / fa, out)

    # ── Determinism canary: store the 38-vector of a fixed sequence so
    #    score_external can assert its featurization path matches train-time. ──
    canary_seq = "GGGGAAAACCCCUUUUGCGCGCGCAUAUAUAUGGGGCCGGGGCCGGGGCC"
    cv = _one(canary_seq).tolist()
    json.dump({"seq": canary_seq, "vec": cv, "n_features": N_FEATURES},
              open(SP / "biophys_v4_canary.json", "w"))
    print(f"  canary stored ({len(cv)} dims)")

    # ── FEGS for the structural negatives ──
    print("== FEGS for structural negatives ==")
    from Functions.precompute_fegs import process_fasta
    fegs_dir = ROOT / "Data/processed/fegs_struct_neg_v4"
    t = time.time()
    process_fasta(ROOT / "Data/raw/multispecies/strict_struct_negatives_v4.fasta",
                  fegs_dir, topk=10, seq_len=1024, overwrite=True, workers=4)
    from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted
    npz = list_npz_sorted(str(fegs_dir))
    print(f"  fegs_struct_neg_v4: {len(npz)} npz  ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
