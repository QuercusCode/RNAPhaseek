"""v11 pool build — PURELY ADDITIVE. Take the PRODUCTION v5/v6 pool unchanged and append the
adversarially-verified v11 additions (10 protein-free POS + 8 NEG; SOFT/EXCLUDE skipped).

This deliberately does NOT remove or relabel anything — unlike v10, whose removal of
protein-dependent (but non-yeast) positives was net-negative in CV. v11 isolates the effect of
ADDING genuine non-yeast RNA-G4-LLPS data + matched negatives, the exact thing the v10 lesson
said we needed. Writes strict_pool_v11_{positives,negatives_all}.fasta.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/data_prep/build_v11_pool.py
"""
import os, sys
sys.path.insert(0, os.getcwd())

POS_IN = "Data/raw/multispecies/strict_pool_v5_positives.fasta"
NEG_IN = "Data/raw/multispecies/strict_pool_v5_negatives_all.fasta"
ADD = "Data/raw/multispecies/staging/v11_additions.fasta"
POS_OUT = "Data/raw/multispecies/strict_pool_v11_positives.fasta"
NEG_OUT = "Data/raw/multispecies/strict_pool_v11_negatives_all.fasta"


def read_fa(f):
    recs = []; h = None; s = ""
    for ln in open(f):
        ln = ln.rstrip()
        if ln.startswith(">"):
            if h is not None: recs.append((h, s))
            h = ln; s = ""
        else:
            s += ln
    if h is not None: recs.append((h, s))
    return recs


def norm(s):
    return "".join(c for c in s.upper().replace("T", "U") if c in "ACGU")


def main():
    pos = read_fa(POS_IN); neg = read_fa(NEG_IN); add = read_fa(ADD)
    add_pos = [(h, s) for h, s in add if h.startswith(">POS")]
    add_neg = [(h, s) for h, s in add if h.startswith(">NEG")]
    skipped = [h for h, _ in add if not (h.startswith(">POS") or h.startswith(">NEG"))]

    # safety: only append additions whose normalized seq is not already in the pool (should be all-new)
    pos_seen = {norm(s) for _, s in pos}
    neg_seen = {norm(s) for _, s in neg}
    new_pos = [(h, s) for h, s in add_pos if norm(s) not in pos_seen]
    new_neg = [(h, s) for h, s in add_neg if norm(s) not in neg_seen]
    dup_pos = len(add_pos) - len(new_pos); dup_neg = len(add_neg) - len(new_neg)

    with open(POS_OUT, "w") as f:
        for h, s in pos:
            f.write(f"{h}\n{s}\n")
        for h, s in new_pos:
            f.write(f">llps_synthetic|v11_{h[5:]}\n{s}\n")   # h[5:] strips ">POS|"
    with open(NEG_OUT, "w") as f:
        for h, s in neg:
            f.write(f"{h}\n{s}\n")
        for h, s in new_neg:
            f.write(f">neg_synthetic|v11_{h[5:]}\n{s}\n")     # h[5:] strips ">NEG|"

    print("=== v11 pool build (PURELY ADDITIVE to v5/v6 pool) ===")
    print(f"positives: {len(pos)} (v5) + {len(new_pos)} (v11 add) = {len(pos)+len(new_pos)}"
          + (f"  [skipped {dup_pos} already-in-pool]" if dup_pos else ""))
    print(f"negatives: {len(neg)} (v5) + {len(new_neg)} (v11 add) = {len(neg)+len(new_neg)}"
          + (f"  [skipped {dup_neg} already-in-pool]" if dup_neg else ""))
    print(f"SOFT/other additions held out (not ingested): {len(skipped)}")
    print(f"wrote {POS_OUT}\nwrote {NEG_OUT}")


if __name__ == "__main__":
    main()
