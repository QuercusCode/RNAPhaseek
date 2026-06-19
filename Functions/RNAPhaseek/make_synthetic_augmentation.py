"""
Generate synthetic repeat-RNA TRAINING augmentation to sharpen the model's
phase-separation repeat threshold.

Biophysical grounding: repeat RNAs undergo a sol-gel transition above a
critical repeat number (~31 for CAG/CUG/CGG; Jain & Vale 2017). We label
sub-threshold repeats NEGATIVE and above-threshold repeats POSITIVE, across
4 motifs, embedded in short random flanks (K variants) for realism + diversity.

These are TRAINING-ONLY: they are appended to each fold's training set in the
eval protocol, never to the locked test or CV outer folds. So the evaluation
remains on real held-out RNA and the comparison to the non-augmented run is fair.

Outputs:
  Data/raw/multispecies/synthetic_train.fasta
  Data/processed/fegs_synth_train/        (FEGS npz, FASTA order)
  Data/splits/biophys_synth_train.npy     (33-dim, FASTA order)
  Data/splits/synthetic_train_meta.json   (labels + which n-values were used)

Run:
  python -m Functions.RNAPhaseek.make_synthetic_augmentation
"""
import os, sys, json, random
import numpy as np
from pathlib import Path
sys.path.insert(0, os.getcwd())
from Functions.RNA_biophysical import RNABiophysicalExtractor
from Functions.precompute_fegs import process_fasta
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted

# ── Threshold-grounded design ──
TRIPLETS  = ["CAG", "CUG", "CGG"]
TRI_NEG_N = [8, 10, 12, 14, 16, 24, 26, 28]   # below ~31 threshold
TRI_POS_N = [34, 38, 42, 50, 60, 75]          # above threshold (avoids real 31/47)
HEX       = "GGGGCC"
HEX_NEG_N = [3, 5, 8, 12]
HEX_POS_N = [28, 40, 55]
K         = 3          # flank variants per (motif, n)
FLANK     = 20         # nt of random flank each side
SEED      = 123

FASTA   = "Data/raw/multispecies/synthetic_train.fasta"
FEGS_D  = Path("Data/processed/fegs_synth_train")
BIO_OUT = "Data/splits/biophys_synth_train.npy"
META    = "Data/splits/synthetic_train_meta.json"


def main():
    rng = random.Random(SEED)
    recs = []  # (header, seq, label)

    def add(motif, ns, label):
        for n in ns:
            for k in range(K):
                lf = "".join(rng.choice("AUGC") for _ in range(FLANK))
                rf = "".join(rng.choice("AUGC") for _ in range(FLANK))
                seq = (lf + motif * n + rf).replace("T", "U")
                tag = "synpos" if label == 1 else "synneg"
                recs.append((f"{tag}|{motif}|n={n}|v={k}", seq, label))

    for m in TRIPLETS:
        add(m, TRI_NEG_N, 0)
        add(m, TRI_POS_N, 1)
    add(HEX, HEX_NEG_N, 0)
    add(HEX, HEX_POS_N, 1)

    n_pos = sum(1 for _, _, l in recs if l == 1)
    n_neg = sum(1 for _, _, l in recs if l == 0)
    print(f"Generated {len(recs)} synthetic repeats  (pos={n_pos} neg={n_neg})")

    # Write FASTA (FASTA order == label order)
    with open(FASTA, "w") as f:
        for h, s, _ in recs:
            f.write(f">{h}\n{s}\n")

    # FEGS
    process_fasta(Path(FASTA), FEGS_D, topk=10, seq_len=1024, overwrite=True, workers=4)
    paths = list_npz_sorted(str(FEGS_D))
    assert len(paths) == len(recs), f"FEGS count {len(paths)} != recs {len(recs)}"

    # Biophysical (33-dim)
    ext = RNABiophysicalExtractor(normalize=False)
    bio = np.stack([ext._compute_one(s) for _, s, _ in recs]).astype(np.float32)
    np.save(BIO_OUT, bio)
    print(f"Saved biophys {bio.shape} -> {BIO_OUT}")

    labels = [l for _, _, l in recs]
    json.dump({
        "n_records": len(recs), "n_pos": n_pos, "n_neg": n_neg,
        "labels": labels,
        "trained_n_values": sorted(set(TRI_NEG_N + TRI_POS_N + HEX_NEG_N + HEX_POS_N)),
        "fasta": FASTA, "fegs_dir": str(FEGS_D), "bio": BIO_OUT,
        "motifs": TRIPLETS + [HEX], "flank_nt": FLANK, "k_variants": K,
    }, open(META, "w"), indent=2)
    print(f"Saved meta -> {META}")
    print(f"  trained n-values (excluded from held-out ladder test): "
          f"{sorted(set(TRI_NEG_N + TRI_POS_N + HEX_NEG_N + HEX_POS_N))}")


if __name__ == "__main__":
    main()
