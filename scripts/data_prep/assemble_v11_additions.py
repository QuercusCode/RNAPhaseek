"""v11 data curation — assemble the ADVERSARIALLY-VERIFIED protein-free RNA-self-LLPS
sequences from the deep-sweep-v2 verification (RNA G-quadruplex LLPS + matched negatives,
the Katz mixed-sequence soluble control, and the 2CZ self-assembling aptamer).

Provenance: every sequence below was retrieved + verified by workflow wf_ca674e7d-2d1
(retrieve full text -> per-sequence adversarial check: supported / protein-free / label).
Each is either (a) literal_from_text (Williams 2022 Suppl Table S1; Katz Methods; CZ/2CZ
patent WO2016057920A1) or (b) reconstructed_from_explicit_motif with an independent
char-for-char re-derivation (Williams 2021 G-tract naming convention). NO sequence is
guessed. See outputs/verification_batches.md and the .SOURCE.md manifest this writes.

ALL NON-YEAST (synthetic / in-vitro). Polyamine(spermine)/Mg2+/K+ driven — no protein.
This DOES NOT retrain. It only stages the additions (heed the v10 net-negative lesson:
add data + evaluate as one batch, do not promote incrementally).
"""
import os, sys
sys.path.insert(0, os.getcwd())

# Each record: (decision, source, name, seq, mech, note)
#   decision: POS / NEG / SOFT / EXCLUDE
#   SOFT  = borderline/disputed-label (incl. the 4 "needs_manual" verdicts) -> staged but
#           held OUT of clean training, exactly like the wadsworth scrambled-CAG handling.
#   EXCLUDE = retrieved + protein-free but NOT ingestible (mislabeled) -> recorded, not written.
RECORDS = [
    # ── Williams, Poudyal & Bevilacqua 2021 (Biochemistry, PMC8755445) ──
    # G-tract / G-quadruplex RNAs, spermine-driven (polyamine), reconstructed from explicit motif names.
    ("POS",  "williams2021", "G3A_4",       "GGGAGGGAGGGAGGGA",                   "G4_G3A_x4_spermine",        "(G3A)4 microaggregates @0.29mM spermine; new_value=overlap (G4-like)"),
    ("POS",  "williams2021", "G3A2_4",      "GGGAAGGGAAGGGAAGGGAA",               "G4_core_G3A2_x4_spermine",  "(G3A2)4 canonical G4 core; aggregates @0.29mM spermine; HIGH value"),
    ("POS",  "williams2021", "A3_G3A2_4_A", "AAAGGGAAGGGAAGGGAAGGGAAA",           "G4_core_polyA_flanked",     "A3(G3A2)4A; aggregates with spermine"),
    ("POS",  "williams2021", "G3332",       "GGGAAGGGAAGGGAAGG",                  "G4_hybrid_3G3tracts",       "G3332: three G3 tracts -> significant aggregation"),
    ("POS",  "williams2021", "G3323",       "GGGAAGGGAAGGAAGGG",                  "G4_hybrid_3G3tracts",       "G3323: three G3 tracts -> significant aggregation"),
    ("NEG",  "williams2021", "G3A5_4",      "GGGAAAAAGGGAAAAAGGGAAAAAGGGAAAAA",   "G4_long_A5_spacer_soluble", "(G3A5)4: long A5 spacer ABOLISHES aggregation @0.29mM spermine"),
    ("NEG",  "williams2021", "G2A2_4",      "GGAAGGAAGGAAGGAA",                   "G2tract_soluble_phys",      "(G2A2)4: soluble at physiological spermine (aggregates only >=1.5mM, non-phys). HIGH-value clean neg"),
    ("SOFT", "williams2021", "G3A3_4",      "GGGAAAGGGAAAGGGAAAGGGAAA",           "G4_A3_spacer_weak",         "(G3A3)4: minimal/weak aggregation — borderline"),
    ("SOFT", "williams2021", "G3322",       "GGGAAGGGAAGGAAGG",                   "G4_hybrid_2G3tracts_weak",  "G3322: two G3 tracts -> slight transition only"),
    ("SOFT", "williams2021", "G2322",       "GGAAGGGAAGGAAGG",                    "G4_hybrid_1G3tract_weak",   "needs_manual: source-labeled neg but shows weak polyamine-dependent aggregation -> SOFT, not hard neg"),

    # ── Williams, Dickson, Lagoa-Miguel & Bevilacqua 2022 (RNA journal, PMC9380743) ──
    # (G3A2)4 core + 5'/3' flanking matrix, LITERAL from Suppl Table S1 (image), spermine/K+, no protein.
    ("POS",  "williams2022", "5pmixed",  "ACAGUUUUUGGGAAGGGAAGGGAAGGGAA",          "G4_5p_ACAGU5_flank",  "5'-mixed: round ~2um condensates"),
    ("POS",  "williams2022", "5pA9",     "AAAAAAAAAGGGAAGGGAAGGGAAGGGAA",          "G4_5p_A9_flank",      "5'-A9: consistent round ~2um condensates"),
    ("POS",  "williams2022", "3pmixed",  "GGGAAGGGAAGGGAAGGGAAACAGUUUUU",          "G4_3p_ACAGU5_flank",  "3'-mixed: aggregates ~0.75-1um (half the 5'-flank size)"),
    ("POS",  "williams2022", "3pA9",     "GGGAAGGGAAGGGAAGGGAAAAAAAAAAA",          "G4_3p_A9_flank",      "3'-A9: aggregates ~0.75-1um"),
    ("NEG",  "williams2022", "5pC9",     "CCCCCCCCCGGGAAGGGAAGGGAAGGGAA",          "G4_5p_C9_flank_noLLPS","5'-C9: NO LLPS under any condition (C9 base-pairs with G-core). HIGH-value matched neg"),
    ("NEG",  "williams2022", "3pC9",     "GGGAAGGGAAGGGAAGGGAACCCCCCCCC",          "G4_3p_C9_flank_noLLPS","3'-C9: no aggregates observed"),
    ("NEG",  "williams2022", "3pU9",     "GGGAAGGGAAGGGAAGGGAAUUUUUUUUU",          "G4_3p_U9_flank_noLLPS","3'-U9: no aggregates observed. HIGH-value matched neg"),
    ("NEG",  "williams2022", "dual_U9",  "UUUUUUUUUGGGAAGGGAAGGGAAGGGAAUUUUUUUUU", "G4_dual_U9_flank_noLLPS","Dual flanked-U9: no aggregates found"),
    ("SOFT", "williams2022", "5pU9",     "UUUUUUUUUGGGAAGGGAAGGGAAGGGAA",          "G4_5p_U9_flank_weak", "5'-U9: only small/infrequent droplets (GU wobble partially suppresses)"),
    ("SOFT", "williams2022", "NRAS",     "UGUGGGAGGGGCGGGUCUGGGUGC",               "bio_NRAS_5pUTR_GQ",   "NRAS 5'UTR biological GQ: soluble at physiological spermine; condenses only >=1.5mM or +PEG -> SOFT"),
    ("SOFT", "williams2022", "dual_mixed","ACAGUUUUUGGGAAGGGAAGGGAAGGGAAACAGUUUUU","G4_dual_ACAGU5_weak", "needs_manual: minor/granular aggregation, weakest of dual set — soft, not clean LLPS positive"),
    ("SOFT", "williams2022", "dual_A9",  "AAAAAAAAAGGGAAGGGAAGGGAAGGGAAAAAAAAAAA", "G4_dual_A9_weak",     "needs_manual: small ~0.5um aggregates only — soft, not high-conf positive"),

    # ── Katz, Tolokh ... Pollack 2017 (Biophys J, PMC5232352) ──
    # spermine RNA condensation. The VALUE here is the mixed-sequence duplex that STAYS SOLUBLE.
    ("NEG", "katz2017", "mixed25_sense",     "GCAUCUGGGCUAUAAAAGGGCGUCG", "mixedseq_25bp_soluble_spermine", "Mixed-sequence 25bp RNA stays SOLUBLE @3mM spermine (verbatim Methods). G-rich yet soluble -> HIGH-value structure-specificity neg"),
    ("NEG", "katz2017", "mixed25_antisense", "CGACGCCCUUUUAUAGCCCAGAUGC", "mixedseq_25bp_soluble_spermine", "Reverse-complement strand of the same soluble duplex; negatives safely attributable to each strand"),

    # ── CZ/2CZ self-assembling RNA aptamer (Biomacromolecules 2017, PMID 28609610; patent WO2016057920A1) ──
    # Mg2+-driven hydrogel, no protein/crosslinker. The GELLING construct is the 237nt tandem 2CZ.
    ("POS",  "cz2cz", "2CZ",
     "GGGAGGCGGAUUCGAGAAUUCAACUGCCAUCUAGGCGGCGCAAAAAACGUAAAAUGGGUCAUGGGAAAGGGCAGGUGAGAGGACUAGUACUACAAGCUUCUGGACUCGGAUCCGUGACCCAAAGGUCAUACUCCCGGAGAAUUCAACUGCCAUCUAGGCGGCGCAAAAAACGUAAAAUGGGUCAUGGGAAAGGGCAGGUGAGAGGACUAGUACUACAAGCUUCUGGACUCCAAUAUU",
     "designed_aptamer_Mg2_hydrogel", "2CZ (237nt) tandem aptamer forms Mg2+-driven hydrogel (CGC ~0.8wt% @25mM MgCl2). HIGH-value designed positive"),
    ("SOFT", "cz2cz", "CZ",
     "GGGAGAAUUCAACUGCCAUCUAGGCGGCGCAAAAAACGUAAAAUGGGUCAUGGGAAAGGGCAGGUGAGAGGACUAGUACUACAAGCUUCUGGACUCGGU",
     "designed_aptamer_weak_monomer", "needs_manual: monomeric CZ (99nt) only forms 'elastic solution', does NOT gel (only 2CZ does) -> soft, not hard positive"),

    # ── EXCLUDED (retrieved + protein-free, but NOT ingestible) ──
    ("EXCLUDE", "katz2017", "polyA_A25", "A" * 25, "homopolymer_duplex_mislabel",
     "REJECTED by verifier: the poly(rA):poly(rU) DUPLEX condenses, not ssA25; + homopolymer overlap class"),
    ("EXCLUDE", "katz2017", "polyU_U25", "U" * 25, "homopolymer_duplex_mislabel",
     "Excluded by same logic as A25 (duplex condenses, not ssU25); ssU25-positive would contradict the existing rU20 NEGATIVE in pool"),
]

# Pending / supplement-only (no usable distinct sequence yet — for the record, mirrors v10 SUPPLEMENT_ONLY)
PENDING = [
    "Williams 2021 Table S1 — exact constant 5'/3' transcription-template flanking nt (if any) on the G-tract "
    "constructs are not in the main text (Europe PMC XML + PMC bin/ returned 404). The reconstructed cores above "
    "are the bare motifs; download ACS SI from pubs.acs.org/doi/suppl/10.1021/acs.biochem.1c00467 to confirm flanks.",
    "Katz poly(rA):poly(rU) DUPLEX positive — real but a two-strand observation; the single-sequence FASTA corpus "
    "cannot represent the duplex, and homopolymer single strands are excluded above. Not staged.",
]

POS_POOL = "Data/raw/multispecies/strict_pool_v5_positives.fasta"      # dedup vs the PRODUCTION (v6/v5) data
NEG_POOL = "Data/raw/multispecies/strict_pool_v5_negatives_all.fasta"
OUT_DIR = "Data/raw/multispecies/staging"
OUT_FA = f"{OUT_DIR}/v11_additions.fasta"
OUT_MD = f"{OUT_DIR}/v11_additions.SOURCE.md"
MIN_LEN = 10   # FEGS can't build a graph below ~8nt


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
    pos_pool = load_seqs(POS_POOL); neg_pool = load_seqs(NEG_POOL)
    print(f"existing v5/v6 pool: {len(pos_pool)} pos seqs, {len(neg_pool)} neg seqs")
    os.makedirs(OUT_DIR, exist_ok=True)

    written = {"POS": [], "NEG": [], "SOFT": []}
    excluded, dup, short, intra_dup = [], [], [], []
    seen = set()
    for decision, src, name, seq, mech, note in RECORDS:
        sq = norm(seq)
        if decision == "EXCLUDE":
            excluded.append((src, name, note)); continue
        if len(sq) < MIN_LEN:
            short.append((src, name)); continue
        if sq in seen:
            intra_dup.append((src, name)); continue
        seen.add(sq)
        # flag (don't drop) overlap with the existing pool, so we know what's genuinely new
        in_pool = (sq in pos_pool) or (sq in neg_pool)
        if in_pool:
            dup.append((decision, src, name))
        written[decision].append((src, name, sq, mech, note, in_pool))

    with open(OUT_FA, "w") as f:
        for decision in ("POS", "NEG", "SOFT"):
            for src, name, sq, mech, note, in_pool in written[decision]:
                tag = mech + ("|ALSO_IN_POOL" if in_pool else "")
                f.write(f">{decision}|{src}|{name}|synthetic|{tag}\n{sq}\n")

    npos, nneg, nsoft = len(written["POS"]), len(written["NEG"]), len(written["SOFT"])
    new_pos = sum(1 for r in written["POS"] if not r[5])
    new_neg = sum(1 for r in written["NEG"] if not r[5])

    # provenance manifest
    with open(OUT_MD, "w") as f:
        f.write("# v11_additions — provenance & label manifest\n\n")
        f.write("Adversarially verified by workflow `wf_ca674e7d-2d1` (retrieve full text -> per-sequence "
                "anti-fabrication + protein-free + label check). All NON-YEAST, protein-free "
                "(spermine/Mg2+/K+ driven). Staged only — NOT retrained (heed v10 net-negative lesson).\n\n")
        f.write(f"**Counts:** {npos} POS ({new_pos} new vs v5 pool), {nneg} NEG ({new_neg} new), "
                f"{nsoft} SOFT (held out of clean training), {len(excluded)} EXCLUDED.\n\n")
        f.write("SOFT = borderline/disputed label (incl. the 4 `needs_manual` verdicts: G2322, dual_mixed, "
                "dual_A9, CZ). Held out of clean pos/neg exactly like wadsworth scrambled-CAG; review before use.\n\n")
        for decision, title in (("POS", "Hard positives"), ("NEG", "Hard negatives"),
                                ("SOFT", "Soft / disputed (held out)")):
            f.write(f"## {title} ({decision})\n\n")
            for src, name, sq, mech, note, in_pool in written[decision]:
                flag = " _(also in pool)_" if in_pool else ""
                f.write(f"- **{name}** ({src}, {len(sq)}nt){flag} — `{sq}`\n  - {note}\n")
            f.write("\n")
        f.write("## Excluded (retrieved + protein-free, but not ingestible)\n\n")
        for src, name, note in excluded:
            f.write(f"- **{name}** ({src}) — {note}\n")
        f.write("\n## Pending (no usable distinct sequence)\n\n")
        for p in PENDING:
            f.write(f"- {p}\n")

    print(f"\nwrote {OUT_FA}")
    print(f"  POS  {npos:2d}  ({new_pos} new vs v5 pool)")
    print(f"  NEG  {nneg:2d}  ({new_neg} new vs v5 pool)")
    print(f"  SOFT {nsoft:2d}  (held out of clean training)")
    print(f"  ---  staged total = {npos + nneg + nsoft}")
    print(f"  EXCLUDED {len(excluded)}: {[n for _, n, _ in excluded]}")
    if short:     print(f"  dropped <{MIN_LEN}nt: {short}")
    if intra_dup: print(f"  intra-batch dups: {intra_dup}")
    if dup:       print(f"  already-in-pool (flagged ALSO_IN_POOL, still staged): {dup}")
    print(f"wrote provenance manifest {OUT_MD}")
    print("\nNEXT (deferred, do NOT auto-run): fold v11 POS/NEG into a pool + rebuild features + "
          "evaluate as ONE batch vs v6. Do not promote incrementally (v10 lesson).")


if __name__ == "__main__":
    main()
