"""v13 matched-pair generator — teach the structure-specificity discrimination by DATA.

Three negatives (v8 features, v11 volume, v12 ERNIE backbone) showed the gap is a DATA problem:
the head learns "G-rich -> positive" because the corpus has no matched pairs where a flank/spacer
change FLIPS LLPS. This generates such pairs for TRAINING, de-leaked from the held-out Williams
benchmark, so the head is taught: G-tracts must be FREE (a complementary flank that sequesters them,
or a disrupting spacer/too-few-tracts, makes a G-rich sequence a NEGATIVE).

Sources:
  (A) Synthetic flank-sequestration pairs on diverse G4 cores OUTSIDE the benchmark (G3An)4-pure-A
      family: free core (POS) vs complementary C/U flank that base-pairs the G-tracts (NEG). Plus
      spacer-disruption and tract-count negatives. Pseudo-labeled by the Williams mechanism
      (experimentally validated for that family), generalized conservatively to new cores.
  (B) Reconstructable REAL motif pairs (experimentally labeled) from the deep sweep: homopolymer
      length/composition (Tom 2022), poly(UG) G4 (Roschdi 2024), repeat thresholds.

De-leak: drop any generated seq with high 8-mer Jaccard to ANY of the 26 held-out benchmark seqs,
so a benchmark gain reflects learning the PRINCIPLE, not memorizing the test.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/data_prep/generate_matched_pairs_v13.py
"""
import os, sys
sys.path.insert(0, os.getcwd())

BENCH = "Data/raw/multispecies/staging/v11_additions.fasta"   # the held-out benchmark (de-leak target)
OUT_DIR = "Data/raw/multispecies/staging"
OUT_FA = f"{OUT_DIR}/v13_matched_pairs.fasta"

# ── (A) diverse G4-forming cores as (name, repeat_unit, n_repeats); spacers G-free; NONE in the
#        benchmark (G3An)4-pure-A family. unit = G-tract + non-G spacer. ──
CORES = [
    ("guA", "GGGUA", 4), ("gAC", "GGGAC", 4), ("gCA", "GGGCA", 4), ("gUU", "GGGUU", 4),
    ("gAU", "GGGAU", 4), ("gCU", "GGGCU", 4), ("gUC", "GGGUC", 4), ("gACA", "GGGACA", 4),
    ("gUUA", "GGGUUA", 4), ("gCAU", "GGGCAU", 4), ("g4UA", "GGGGUA", 4), ("g4AC", "GGGGAC", 4),
    ("g4UU", "GGGGUU", 4), ("g4c2", "GGGGCC", 4), ("g4c2b", "GGGGCC", 5), ("guA5", "GGGUA", 5),
    ("guA6", "GGGUA", 6), ("gAC5", "GGGAC", 5), ("g4A5", "GGGGA", 5), ("gUCA", "GGGUCA", 4),
]
FLANK_LEN = [6, 8, 9, 10, 12]   # vary so the head can't key on a fixed 9-nt prefix

POS, NEG = [], []
for i, (name, unit, n) in enumerate(CORES):
    core = unit * n
    gtract = unit[:len(unit) - len(unit.lstrip("G"))]    # leading G run, "GGG" or "GGGG"
    spacer = unit[len(gtract):]                           # the non-G spacer
    L = FLANK_LEN[i % len(FLANK_LEN)]
    POS.append((f"{name}_core", core, "free_G4_core"))
    POS.append((f"{name}_5pA{L}", "A" * L + core, "Aflank_free"))            # A-flank cannot pair G-tracts
    # NEG type 1: complementary flank SEQUESTERS the G-core (rotate type/position so the rule generalizes)
    t = i % 4
    if t == 0:   NEG.append((f"{name}_5pC{L}", "C" * L + core, "Cflank5p_sequestered"))
    elif t == 1: NEG.append((f"{name}_3pC{L}", core + "C" * L, "Cflank3p_sequestered"))
    elif t == 2: NEG.append((f"{name}_5pU{L}", "U" * L + core, "Uflank5p_wobble_sequestered"))
    else:        NEG.append((f"{name}_3pU{L}", core + "U" * L, "Uflank3p_wobble_sequestered"))
    # NEG type 2: alternate spacer-disruption vs tract-count collapse (the G3A5 / G2A2 mechanisms)
    if i % 2 == 0:
        NEG.append((f"{name}_longspacer", (gtract + "A" * 8) * n, "longspacer_disrupted"))  # tracts too far apart
    else:
        NEG.append((f"{name}_2tract", (gtract + spacer) * 2, "two_tract_nonG4"))             # only 2 tracts

# ── (B) reconstructable REAL motif pairs (experimentally labeled; non-Williams) ──
REAL_POS = [
    ("tom2022_rA20", "A" * 20, "polyA_condenses_Mg"),         # Tom 2022
    ("tom2022_rA30", "A" * 30, "polyA_condenses_Mg"),
    ("roschdi2024_GU18", "GU" * 18, "polyUG_G4"),             # Roschdi 2024 poly(UG)
    ("roschdi2024_GU24", "GU" * 24, "polyUG_G4"),
    ("repeat_CAG31", "CAG" * 31, "repeat_above_threshold"),
    ("repeat_CUG31", "CUG" * 31, "repeat_above_threshold"),
]
REAL_NEG = [
    ("tom2022_rA10", "A" * 10, "polyA_below_threshold"),      # Tom 2022 matched negatives
    ("tom2022_rC20", "C" * 20, "polyC_soluble"),
    ("tom2022_rU20", "U" * 20, "polyU_soluble"),
    ("roschdi2024_CUG12", "CUG" * 12, "non_G4_soluble"),
    ("repeat_CUU31", "CUU" * 31, "CUU_does_not_phase_separate"),  # Wadsworth: confirmed negative
    ("repeat_CAG10", "CAG" * 10, "repeat_below_threshold"),
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


def max_jaccard(seq, ref_kmers_list):
    ks = kmers(seq)
    best = 0.0
    for rk in ref_kmers_list:
        if not ks or not rk: continue
        j = len(ks & rk) / len(ks | rk)
        if j > best: best = j
    return best


def main():
    bench = [norm(s) for _, s in read_fa(BENCH)]
    bench_km = [kmers(b) for b in bench]
    bench_set = set(bench)
    MIN_LEN = 10
    rows = []  # (label, name, seq, mech)
    for name, seq, mech in POS: rows.append(("POS", name, norm(seq), mech))
    for name, seq, mech in NEG: rows.append(("NEG", name, norm(seq), mech))
    for name, seq, mech in REAL_POS: rows.append(("POS", f"real_{name}", norm(seq), mech))
    for name, seq, mech in REAL_NEG: rows.append(("NEG", f"real_{name}", norm(seq), mech))

    kept, leaked, short, dup, seen = [], [], [], [], set()
    for lab, name, sq, mech in rows:
        if len(sq) < MIN_LEN: short.append(name); continue
        if sq in seen: dup.append(name); continue
        seen.add(sq)
        if sq in bench_set or max_jaccard(sq, bench_km) >= 0.60:   # de-leak vs benchmark
            leaked.append((name, round(max_jaccard(sq, bench_km), 2))); continue
        kept.append((lab, name, sq, mech))

    np_, nn_ = sum(l == "POS" for l, *_ in kept), sum(l == "NEG" for l, *_ in kept)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FA, "w") as f:
        for lab, name, sq, mech in kept:
            src = "real" if name.startswith("real_") else "synth"
            f.write(f">{lab}|v13_{src}|{name}|synthetic|{mech}\n{sq}\n")

    print(f"benchmark de-leak target: {len(bench)} seqs")
    print(f"generated kept: {len(kept)}  ({np_} POS / {nn_} NEG)")
    print(f"  de-leaked (Jaccard>=0.60 to a benchmark seq): {len(leaked)}  {leaked[:8]}")
    if short: print(f"  dropped <{MIN_LEN}nt: {short}")
    if dup: print(f"  intra-dup dropped: {dup}")
    print(f"  max Jaccard to benchmark among KEPT: "
          f"{max((max_jaccard(sq, bench_km) for _,_,sq,_ in kept), default=0):.2f}  (want < 0.60)")
    print(f"wrote {OUT_FA}")


if __name__ == "__main__":
    main()
