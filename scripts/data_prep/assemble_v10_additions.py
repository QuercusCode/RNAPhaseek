"""v10 data curation — assemble the RECONSTRUCTABLE protein-free RNA-self-LLPS sequences
from the verified 2026 sweep (homopolymers + repeats + matched negatives), dedup vs the
existing pool, and emit a v10 staging FASTA + a manifest of label fixes / removals.

Only unambiguously-defined sequences are generated here; supplement-only designs (nanostars,
riboswitches, Wang&Xu G->A mutants) are listed for separate download. All NON-YEAST.
"""
import os, sys
sys.path.insert(0, os.getcwd())

# ── reconstructable POSITIVES (protein-free, from verified sources) ──
POS = [
    ("tom2022_polyA_rA15",   "A" * 15),
    ("tom2022_polyA_rA20",   "A" * 20),
    ("tom2022_polyA_rA30",   "A" * 30),
    ("raguseo2023_C9_GGGGCC6",  "GGGGCC" * 6),
    ("raguseo2023_C9_GGGGCC10", "GGGGCC" * 10),
    ("roschdi2024_pUG_GU12",  "GU" * 12),
    ("roschdi2024_pUG_GU18",  "GU" * 18),
    ("roschdi2024_pUG_GU36",  "GU" * 36),
    ("wadsworth2023_CAG20",   "CAG" * 20),
    ("wadsworth2023_CUG31",   "CUG" * 31),
    ("wan2025_sCAG6",         "CAG" * 6),
    ("wan2025_sCAG7",         "CAG" * 7),
]
# ── reconstructable NEGATIVES (protein-free, confirmed non-condensing — the scarce, valuable type) ──
NEG = [
    ("wadsworth2023_CUU31",   "CUU" * 31),    # RELABEL: was a positive; paper = no LLPS
    ("tom2022_rA10",          "A" * 10),
    ("tom2022_rC20",          "C" * 20),
    ("tom2022_rU20",          "U" * 20),
    ("raguseo2023_C9_GGGGCC2",  "GGGGCC" * 2),
    ("raguseo2023_C9_GGGGCC3",  "GGGGCC" * 3),
    ("iconrna_CAG10",         "CAG" * 10),
    ("pan2023_CAG12",         "CAG" * 12),
    ("pan2023_CAG25",         "CAG" * 25),
    ("roschdi2024_GGU2",      "GGU" * 2),
    ("roschdi2024_CUG12",     "CUG" * 12),
    ("xue2023_AC70",          "AC" * 70),
]
SUPPLEMENT_ONLY = [
    "Stewart 2024 nanostars (Suppl Data 1 xlsx — already staged)",
    "Tang 2025 / Li 2026 / You-lab nanostar designs (suppl + GitHub)",
    "Poudyal 2021 riboswitch condensates (Rfam-reconstructable; need exact edits)",
    "Wang&Xu 2024 short G-rich + G->A point-mutant negatives (Suppl Table 1)",
    "Aierken&Joseph + iConRNA computational +/- matrices (GitHub; soft labels)",
]


def norm(s):
    return "".join(c for c in s.upper().replace("T", "U") if c in "ACGU")


def load_seqs(f):
    out = set(); s = ""
    if not os.path.exists(f):
        return out
    for ln in open(f):
        if ln.startswith(">"):
            if s: out.add(norm(s))
            s = ""
        else:
            s += ln.strip()
    if s: out.add(norm(s))
    return out


def main():
    pos_pool = load_seqs("Data/raw/multispecies/strict_pool_v5_positives.fasta")
    neg_pool = load_seqs("Data/raw/multispecies/strict_pool_v5_negatives_all.fasta")
    print(f"existing pool: {len(pos_pool)} pos seqs, {len(neg_pool)} neg seqs")

    out_dir = "Data/raw/multispecies/staging"
    os.makedirs(out_dir, exist_ok=True)
    MIN_LEN = 10   # FEGS can't build a graph below ~8nt; drop ultra-short to keep features aligned
    add_pos, add_neg, dup_pos, dup_neg, short = [], [], [], [], []
    for name, seq in POS:
        sq = norm(seq)
        if len(sq) < MIN_LEN:
            short.append(name); continue
        (dup_pos if sq in pos_pool else add_pos).append((name, sq))
    for name, seq in NEG:
        sq = norm(seq)
        if len(sq) < MIN_LEN:
            short.append(name); continue
        # a negative is "new" unless already a negative; if it exists only as a positive it's a RELABEL (still add)
        (dup_neg if sq in neg_pool else add_neg).append((name, sq))
    if short:
        print(f"dropped {len(short)} ultra-short (<{MIN_LEN}nt, FEGS-incompatible): {short}")

    with open(f"{out_dir}/v10_additions.fasta", "w") as f:
        for name, sq in add_pos:
            f.write(f">POS|v10|{name}|reconstructed|protein_free\n{sq}\n")
        for name, sq in add_neg:
            tag = "relabel_from_pos" if sq in pos_pool else "protein_free"
            f.write(f">NEG|v10|{name}|reconstructed|{tag}\n{sq}\n")

    print(f"\nNEW positives: {len(add_pos)}  (skipped {len(dup_pos)} already in pool: {[n for n,_ in dup_pos]})")
    print(f"NEW negatives: {len(add_neg)}  (skipped {len(dup_neg)} already-neg)")
    relabels = [n for n, sq in add_neg if sq in pos_pool]
    print(f"RELABELS (seq currently a POSITIVE -> should be NEGATIVE): {relabels}")
    print(f"\nwrote {out_dir}/v10_additions.fasta  ({len(add_pos)} pos + {len(add_neg)} neg)")
    print("\nSTILL TO MINE (supplement/download required):")
    for s in SUPPLEMENT_ONLY:
        print(f"  - {s}")
    print("\nLABEL FIXES needed in pool rebuild:")
    print("  * REMOVE wadsworth2023_r_CUU31 from positives (it's the relabel above)")
    print("  * AUDIT/REMOVE protein-dependent positives: NEAT1/NORAD/MALAT1/Xist (16) + suspect RNAPhaSep (76)")


if __name__ == "__main__":
    main()
