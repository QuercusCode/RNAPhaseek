"""v15 pool — PURELY ADDITIVE: v5/v6 production pool + the de-leaked matched TRAINING pairs
(scripts/data_prep/generate_matched_pairs_v15.py). Teaches the structure-specificity discrimination
that the corpus lacked (G-tracts must be FREE). The held-out Williams benchmark (v11_additions) is
NOT included — it stays a clean test. Writes strict_pool_v15_{positives,negatives_all}.fasta.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/data_prep/build_v15_pool.py
"""
import os, sys
sys.path.insert(0, os.getcwd())

POS_IN = "Data/raw/multispecies/strict_pool_v5_positives.fasta"
NEG_IN = "Data/raw/multispecies/strict_pool_v5_negatives_all.fasta"
ADD = "Data/raw/multispecies/staging/v15_matched_pairs.fasta"
POS_OUT = "Data/raw/multispecies/strict_pool_v15_positives.fasta"
NEG_OUT = "Data/raw/multispecies/strict_pool_v15_negatives_all.fasta"


def read_fa(f):
    recs = []; h = None; s = ""
    for ln in open(f):
        ln = ln.rstrip()
        if ln.startswith(">"):
            if h is not None: recs.append((h, s))
            h = ln; s = ""
        else: s += ln
    if h is not None: recs.append((h, s))
    return recs


def norm(s):
    return "".join(c for c in s.upper().replace("T", "U") if c in "ACGU")


def main():
    pos = read_fa(POS_IN); neg = read_fa(NEG_IN); add = read_fa(ADD)
    add_pos = [(h, s) for h, s in add if h.startswith(">POS")]
    add_neg = [(h, s) for h, s in add if h.startswith(">NEG")]
    pos_seen = {norm(s) for _, s in pos}; neg_seen = {norm(s) for _, s in neg}
    new_pos = [(h, s) for h, s in add_pos if norm(s) not in pos_seen]
    new_neg = [(h, s) for h, s in add_neg if norm(s) not in neg_seen]

    with open(POS_OUT, "w") as f:
        for h, s in pos: f.write(f"{h}\n{s}\n")
        for h, s in new_pos: f.write(f">llps_synthetic|{h[5:]}\n{s}\n")
    with open(NEG_OUT, "w") as f:
        for h, s in neg: f.write(f"{h}\n{s}\n")
        for h, s in new_neg: f.write(f">neg_synthetic|{h[5:]}\n{s}\n")

    print("=== v15 pool build (v5 + de-leaked matched training pairs) ===")
    print(f"positives: {len(pos)} + {len(new_pos)} (matched) = {len(pos)+len(new_pos)}"
          + (f"  [skipped {len(add_pos)-len(new_pos)} already-in-pool]" if len(add_pos) != len(new_pos) else ""))
    print(f"negatives: {len(neg)} + {len(new_neg)} (matched) = {len(neg)+len(new_neg)}"
          + (f"  [skipped {len(add_neg)-len(new_neg)} already-in-pool]" if len(add_neg) != len(new_neg) else ""))
    print(f"wrote {POS_OUT}\nwrote {NEG_OUT}")


if __name__ == "__main__":
    main()
