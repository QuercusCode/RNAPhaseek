"""v15 matched pairs — refine v14 toward a clean-promote model. Same flank-sequestration signal,
but ADD distinct (core-dependent) spacer-disruption and tract-count negatives to close the
G2A2/G3A5 weak spot (v14 under-supplied these: longspacer variants deduped). Tagged 'v15_synth' /
'v15_real' so the trainer can DOWNWEIGHT the synthetic bulk (the fix for v14's small general
regression). De-leaked vs the 26 held-out benchmark seqs (8-mer Jaccard < 0.60).

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/data_prep/generate_matched_pairs_v15.py
"""
import os, sys
sys.path.insert(0, os.getcwd())

BENCH = "Data/raw/multispecies/staging/v11_additions.fasta"
OUT_FA = "Data/raw/multispecies/staging/v15_matched_pairs.fasta"

SPACERS2 = ["UA", "AC", "CA", "UU", "AU", "CU", "UC", "CC"]
SPACERS3 = ["ACA", "UUA", "CAU", "UCA", "AUC", "CUA", "UAC", "ACU", "UCU", "CAC"]
TRACTS = ["GGG", "GGGG"]
REPS = [4, 5]
FLANK_LEN = [6, 8, 9, 10, 12, 14]


def build_cores():
    combos = [(t, s) for t in TRACTS for s in SPACERS2] + [(t, s) for t in TRACTS for s in SPACERS3]
    cores = []
    for i, (t, s) in enumerate(combos):
        r = REPS[i % len(REPS)]
        cores.append((f"{'g4' if t=='GGGG' else 'g3'}_{s}_{r}", t, s, r))
    cores += [("g4c2_4", "GGGG", "CC", 4), ("g4c2_5", "GGGG", "CC", 5), ("g4c2_6", "GGGG", "CC", 6)]
    return cores[:40]


REAL_POS = [
    ("tom2022_rA20", "A" * 20, "polyA_Mg"), ("tom2022_rA30", "A" * 30, "polyA_Mg"),
    ("roschdi2024_GU18", "GU" * 18, "polyUG_G4"), ("roschdi2024_GU24", "GU" * 24, "polyUG_G4"),
    ("roschdi2024_GU30", "GU" * 30, "polyUG_G4"), ("repeat_CAG31", "CAG" * 31, "repeat_thresh"),
    ("repeat_CUG31", "CUG" * 31, "repeat_thresh"), ("repeat_CGG20", "CGG" * 20, "repeat_thresh"),
]
REAL_NEG = [
    ("tom2022_rA10", "A" * 10, "polyA_below"), ("tom2022_rC20", "C" * 20, "polyC_sol"),
    ("tom2022_rU20", "U" * 20, "polyU_sol"), ("tom2022_rC30", "C" * 30, "polyC_sol"),
    ("roschdi2024_CUG12", "CUG" * 12, "nonG4_sol"), ("repeat_CUU31", "CUU" * 31, "no_LLPS"),
    ("repeat_CAG10", "CAG" * 10, "repeat_below"), ("repeat_AC35", "AC" * 35, "AC_sol"),
]


def norm(s):
    return "".join(c for c in s.upper().replace("T", "U") if c in "ACGU")


def read_fa(f):
    out = []; h = None; s = ""
    for ln in open(f):
        ln = ln.rstrip()
        if ln.startswith(">"):
            if h is not None: out.append((h, s))
            h = ln[1:]; s = ""
        else: s += ln
    if h is not None: out.append((h, s))
    return out


def kmers(s, k=8):
    return {s[i:i+k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


def main():
    bench = [norm(s) for _, s in read_fa(BENCH)]; bench_km = [kmers(b) for b in bench]; bench_set = set(bench)
    rows = []
    for i, (name, tract, spacer, r) in enumerate(build_cores()):
        core = (tract + spacer) * r
        L, L2 = FLANK_LEN[i % 6], FLANK_LEN[(i + 3) % 6]
        rows.append(("POS", f"{name}_core", core, "free_G4_core"))
        rows.append(("POS", f"{name}_5pA{L}", "A" * L + core, "Aflank_free"))
        # flank-sequestration negatives (kept from v14)
        rows.append(("NEG", f"{name}_5pC{L}", "C" * L + core, "Cflank5p_seq"))
        rows.append(("NEG", f"{name}_3pC{L2}", core + "C" * L2, "Cflank3p_seq"))
        rows.append(("NEG", f"{name}_{'5' if i%2 else '3'}pU{L}",
                     ("U" * L + core) if i % 2 else (core + "U" * L), "Uflank_seq"))
        # NEW: distinct, core-dependent structural negatives (close the spacer/tract weak spot)
        rows.append(("NEG", f"{name}_2tract", (tract + spacer) * 2, "two_tract_nonG4"))
        rows.append(("NEG", f"{name}_3tract", (tract + spacer) * 3, "three_tract_nonG4"))
        rows.append(("NEG", f"{name}_longspacer", (tract + spacer * 4) * r, "longspacer_disrupted"))
    for name, seq, mech in REAL_POS: rows.append(("POS", f"real_{name}", seq, mech))
    for name, seq, mech in REAL_NEG: rows.append(("NEG", f"real_{name}", seq, mech))

    kept, leaked, seen, short = [], 0, set(), 0
    for lab, name, seq, mech in rows:
        sq = norm(seq)
        if len(sq) < 10: short += 1; continue
        if sq in seen: continue
        seen.add(sq); ks = kmers(sq)
        mj = max((len(ks & rk) / len(ks | rk) for rk in bench_km if ks and rk), default=0)
        if sq in bench_set or mj >= 0.60: leaked += 1; continue
        kept.append((lab, name, sq, mech))
    np_, nn_ = sum(l == "POS" for l, *_ in kept), sum(l == "NEG" for l, *_ in kept)
    with open(OUT_FA, "w") as f:
        for lab, name, sq, mech in kept:
            src = "real" if name.startswith("real_") else "synth"
            f.write(f">{lab}|v15_{src}|{name}|synthetic|{mech}\n{sq}\n")
    mjmax = max((max((len(kmers(sq) & rk) / len(kmers(sq) | rk) for rk in bench_km), default=0)
                 for _, _, sq, _ in kept), default=0)
    print(f"generated kept: {len(kept)} ({np_} POS / {nn_} NEG) | de-leaked {leaked} | short {short} | "
          f"dup {len(rows)-len(kept)-leaked-short}")
    print(f"  max Jaccard to benchmark among KEPT: {mjmax:.2f}  (want < 0.60)")
    print(f"wrote {OUT_FA}")


if __name__ == "__main__":
    main()
