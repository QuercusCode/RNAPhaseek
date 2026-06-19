"""v10 pool build — remove protein-dependent positives (lncRNA/mRNA clients from RNAPhaSep /
RPS2 lncRNA sources), relabel wadsworth r_CUU31 POS->NEG, and add the v10 reconstructable
sequences. Keeps genuinely protein-free RNAs (repeats, G4, designed) even if from those sources.

Writes Data/raw/multispecies/strict_pool_v10_{positives,negatives_all}.fasta and reports.
"""
import os, sys
sys.path.insert(0, os.getcwd())

POS_IN = "Data/raw/multispecies/strict_pool_v5_positives.fasta"
NEG_IN = "Data/raw/multispecies/strict_pool_v5_negatives_all.fasta"
ADD = "Data/raw/multispecies/staging/v10_additions.fasta"
POS_OUT = "Data/raw/multispecies/strict_pool_v10_positives.fasta"
NEG_OUT = "Data/raw/multispecies/strict_pool_v10_negatives_all.fasta"

# genuinely protein-free RNA mechanisms -> KEEP even if from a protein-centric source
KEEP_MECH = ["(CUG)", "(CAG)", "(CGG)", "(CCUG)", "G4C2", "GGGGCC", "quadruplex",
             "G-quad", "repeat", "nanostar", "designed", "synthetic", "homopol", "kissing"]


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


def is_protein_dependent(h):
    """True if this positive is a protein-condensate client (remove), False if protein-free RNA."""
    hl = h.lower()
    from_protein_src = ("rnaphasep" in hl) or ("|rps2|" in hl) or ("|rps_" in hl)
    named_lnc = any(x in h for x in ["NEAT1", "NORAD", "MALAT1", "XIST", "Xist", "FIRRE", "Firre"])
    keep = any(m.lower() in hl for m in KEEP_MECH)
    return (from_protein_src or named_lnc) and not keep


def main():
    report = "--report" in sys.argv
    pos = read_fa(POS_IN); neg = read_fa(NEG_IN)
    add = read_fa(ADD)
    add_pos = [(h, s) for h, s in add if h.startswith(">POS")]
    add_neg = [(h, s) for h, s in add if h.startswith(">NEG")]

    removed, kept = [], []
    for h, s in pos:
        if "r_CUU31" in h:            # the confirmed relabel -> goes to negatives, not kept here
            removed.append((h, "relabel->NEG"))
            continue
        if is_protein_dependent(h):
            removed.append((h, "protein-dependent"))
        else:
            kept.append((h, s))

    print(f"=== v10 pool build {'(REPORT ONLY)' if report else ''} ===")
    print(f"old positives: {len(pos)}  ->  kept {len(kept)}  |  removed {len(removed)}")
    print(f"  removed protein-dependent: {sum(1 for _, r in removed if r=='protein-dependent')}")
    print(f"  relabeled r_CUU31 -> NEG:  {sum(1 for _, r in removed if r.startswith('relabel'))}")
    print(f"new positives added: {len(add_pos)}  ->  v10 positives = {len(kept)+len(add_pos)}")
    print(f"old negatives: {len(neg)}  + new {len(add_neg)}  ->  v10 negatives = {len(neg)+len(add_neg)}")
    # show a sample of removals + any kept-by-mechanism exceptions
    print("\n  sample removals:")
    for h, r in removed[:8]:
        print(f"    [{r}] {h[:70]}")
    kept_exc = [h for h, s in kept if (("rnaphasep" in h.lower()) or ("|rps2|" in h.lower()) or ("|rps_" in h.lower()))]
    print(f"\n  RNAPhaSep/RPS entries KEPT (protein-free by mechanism): {len(kept_exc)}")
    for h in kept_exc[:6]:
        print(f"    {h[:70]}")

    if report:
        print("\n(report only — no files written; rerun without --report to write v10 pool)")
        return

    with open(POS_OUT, "w") as f:
        for h, s in kept:
            f.write(f"{h}\n{s}\n")
        for h, s in add_pos:
            f.write(f">llps_synthetic|{h[1:]}\n{s}\n")
    soft_excl = 0
    with open(NEG_OUT, "w") as f:
        for h, s in neg:
            # Wadsworth scrambled-CAG: weakened LCST but STILL condenses -> NOT a clean negative,
            # and as a composition-preserving scramble that condenses it would corrupt struct-specificity.
            if "scrambled_CAG" in h:
                soft_excl += 1
                continue
            f.write(f"{h}\n{s}\n")
        for h, s in add_neg:
            f.write(f">neg_synthetic|{h[1:]}\n{s}\n")
    n_neg_out = len(neg) - soft_excl + len(add_neg)
    print(f"\n  excluded {soft_excl} soft scrambled-CAG from clean negatives")
    print(f"wrote {POS_OUT} ({len(kept)+len(add_pos)}) and {NEG_OUT} ({n_neg_out})")


if __name__ == "__main__":
    main()
