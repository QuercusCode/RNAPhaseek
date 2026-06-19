"""v14 scale-up of the v13 matched-pair set (which closed the structure-specificity gap on the
held-out benchmark: matched-pair acc 0.67->1.00). Scales the SAME teaching signal:
  - ~40 diverse G4 cores generated programmatically (G-tract in {GGG,GGGG} x G-free spacer x reps),
    ALL outside the benchmark (G3An)4-pure-A family.
  - per core: 2 POS (free: bare core + A-flank) + 4 NEG, SKEWED to negatives to push the absolute
    sequestered-NEG scores below 0.5 (the one item the pilot left open). NEG = complementary C/U
    flank that sequesters the G-tracts (5'/3', varied length) + spacer-disruption / tract-count.
  - reconstructable REAL motif pairs (Tom homopolymers, Roschdi poly-UG, repeat thresholds).
De-leaked vs the 26 held-out benchmark seqs by 8-mer Jaccard (drop >=0.60).

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/data_prep/generate_matched_pairs_v14.py
"""
import os, sys
sys.path.insert(0, os.getcwd())

BENCH = "Data/raw/multispecies/staging/v11_additions.fasta"
OUT_FA = "Data/raw/multispecies/staging/v14_matched_pairs.fasta"

# G-free spacers (no 'G'); none is pure-A (so cores stay outside the benchmark GGG+polyA family)
SPACERS2 = ["UA", "AC", "CA", "UU", "AU", "CU", "UC", "CC"]
SPACERS3 = ["ACA", "UUA", "CAU", "UCA", "AUC", "CUA", "UAC", "ACU", "UCU", "CAC"]
TRACTS = ["GGG", "GGGG"]
REPS = [4, 5]
FLANK_LEN = [6, 8, 9, 10, 12, 14]


def build_cores():
    cores = []  # (name, tract, spacer, reps)
    combos = [(t, s) for t in TRACTS for s in SPACERS2] + [(t, s) for t in TRACTS for s in SPACERS3]
    for i, (t, s) in enumerate(combos):
        r = REPS[i % len(REPS)]
        nm = f"{'g4' if t=='GGGG' else 'g3'}_{s}_{r}"
        cores.append((nm, t, s, r))
    # a few G4C2-style cores (different family again)
    cores += [("g4c2_4", "GGGG", "CC", 4), ("g4c2_5", "GGGG", "CC", 5), ("g4c2_6", "GGGG", "CC", 6)]
    return cores[:40]   # cap at 40 distinct cores


REAL_POS = [
    ("tom2022_rA20", "A" * 20, "polyA_condenses_Mg"), ("tom2022_rA30", "A" * 30, "polyA_condenses_Mg"),
    ("roschdi2024_GU18", "GU" * 18, "polyUG_G4"), ("roschdi2024_GU24", "GU" * 24, "polyUG_G4"),
    ("roschdi2024_GU30", "GU" * 30, "polyUG_G4"),
    ("repeat_CAG31", "CAG" * 31, "repeat_above_threshold"), ("repeat_CUG31", "CUG" * 31, "repeat_above_threshold"),
    ("repeat_CGG20", "CGG" * 20, "repeat_above_threshold"),
]
REAL_NEG = [
    ("tom2022_rA10", "A" * 10, "polyA_below_threshold"), ("tom2022_rC20", "C" * 20, "polyC_soluble"),
    ("tom2022_rU20", "U" * 20, "polyU_soluble"), ("tom2022_rC30", "C" * 30, "polyC_soluble"),
    ("roschdi2024_CUG12", "CUG" * 12, "non_G4_soluble"), ("repeat_CUU31", "CUU" * 31, "no_phase_separation"),
    ("repeat_CAG10", "CAG" * 10, "repeat_below_threshold"), ("repeat_AC35", "AC" * 35, "AC_repeat_soluble"),
]


def norm(s):
    return "".join(c for c in s.upper().replace("T", "U") if c in "ACGU")


def read_fa(f):
    out = []; h = None; s = ""
    for ln in open(f):
        ln = ln.rstrip()
        if ln.startswith(">"):
            if h is not None: out.append((h, s));
            h = ln[1:] if ln.startswith(">") else h; s = ""
        else: s += ln
    if h is not None: out.append((h, s))
    return out


def kmers(s, k=8):
    return {s[i:i+k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


def main():
    bench = [norm(s) for _, s in read_fa(BENCH)]
    bench_km = [kmers(b) for b in bench]; bench_set = set(bench)

    rows = []  # (label, name, seq, mech)
    cores = build_cores()
    for i, (name, tract, spacer, r) in enumerate(cores):
        core = (tract + spacer) * r
        L = FLANK_LEN[i % len(FLANK_LEN)]; L2 = FLANK_LEN[(i + 3) % len(FLANK_LEN)]
        # POS: free
        rows.append(("POS", f"{name}_core", core, "free_G4_core"))
        rows.append(("POS", f"{name}_5pA{L}", "A" * L + core, "Aflank_free"))
        # NEG: 4 per core, emphasising complementary-flank sequestration (5' and 3', C and U)
        rows.append(("NEG", f"{name}_5pC{L}", "C" * L + core, "Cflank5p_sequestered"))
        rows.append(("NEG", f"{name}_3pC{L2}", core + "C" * L2, "Cflank3p_sequestered"))
        rows.append(("NEG", f"{name}_{'5' if i%2 else '3'}pU{L}",
                     ("U" * L + core) if i % 2 else (core + "U" * L), "Uflank_wobble_sequestered"))
        if i % 2 == 0:
            rows.append(("NEG", f"{name}_longspacer", (tract + "A" * 8) * r, "longspacer_disrupted"))
        else:
            rows.append(("NEG", f"{name}_2tract", (tract + spacer) * 2, "two_tract_nonG4"))
    for name, seq, mech in REAL_POS: rows.append(("POS", f"real_{name}", seq, mech))
    for name, seq, mech in REAL_NEG: rows.append(("NEG", f"real_{name}", seq, mech))

    kept, leaked, seen, short = [], [], set(), 0
    for lab, name, seq, mech in rows:
        sq = norm(seq)
        if len(sq) < 10: short += 1; continue
        if sq in seen: continue
        seen.add(sq)
        ks = kmers(sq)
        mj = max((len(ks & rk) / len(ks | rk) for rk in bench_km if ks and rk), default=0)
        if sq in bench_set or mj >= 0.60:
            leaked.append((name, round(mj, 2))); continue
        kept.append((lab, name, sq, mech))

    np_, nn_ = sum(l == "POS" for l, *_ in kept), sum(l == "NEG" for l, *_ in kept)
    with open(OUT_FA, "w") as f:
        for lab, name, sq, mech in kept:
            src = "real" if name.startswith("real_") else "synth"
            f.write(f">{lab}|v14_{src}|{name}|synthetic|{mech}\n{sq}\n")
    mjmax = max((max((len(kmers(sq) & rk) / len(kmers(sq) | rk) for rk in bench_km), default=0)
                 for _, _, sq, _ in kept), default=0)
    print(f"cores: {len(cores)}  | generated kept: {len(kept)} ({np_} POS / {nn_} NEG)")
    print(f"  de-leaked (Jaccard>=0.60): {len(leaked)}  {leaked[:6]}")
    print(f"  dropped <10nt: {short} | intra-dup dropped: {len(rows)-len(kept)-len(leaked)-short}")
    print(f"  max Jaccard to benchmark among KEPT: {mjmax:.2f}  (want < 0.60)")
    print(f"wrote {OUT_FA}")


if __name__ == "__main__":
    main()
