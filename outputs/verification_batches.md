# Protein-free RNA-LLPS verification batches (from deep sweep v2)

Verify with direct WebFetch (main-loop, cheap) — confirm protein-free, extract pos/neg, note sequence availability.
Status: [ ] pending · [~] verifying · [Y] confirmed protein-free · [N] excluded (protein-dependent) · [P] partial

## BATCH 1 — wet-lab, MATCHED NEGATIVES (highest value: scarce confirmed negatives)
| # | source | DOI / PMC | claimed pos/neg | status |
|---|---|---|---|---|
| 1 | Williams, Dickson, Lagoa-Miguel & Bevilacqua 2022 — RNA G4 LLPS flanking-seq matrix | 10.1261/rna.079196.122 / PMC9380743 | 5 / 5 | [~] |
| 2 | Williams, Poudyal & Bevilacqua 2021 — long G-tract G4 self-aggregation | 10.1021/acs.biochem.1c00467 / PMC8755445 | 7 / 2 | [~] |
| 3 | Wadsworth, Aierken … Banerjee 2025 — 2'-OH role in RNA PS/percolation | bioRxiv 10.1101/2025.02.26.638501 | 4 / 5 | [~] |
| 4 | Onuchic, Milin … Banerjee 2019 — divalent-cation polyU coacervation (homotypic) | 10.1038/s41598-019-48457-x / PMC6704260 | 1 / 2 | [~] |
| 5 | Aumiller, Pir Cakmak, Davis & Keating 2016 — polyU/polyamine coacervates | 10.1021/acs.langmuir.6b02499 / PMID 27599198 | 2 / 0 | [~] |
| 6 | Marianelli, Miller & Keating 2017 — crowding on polyU/spermine coacervation | 10.1039/c7sm02146a / PMID 29265152 | 2 / 0 | [~] |

## BATCH 1 RESULTS (verified 2026-06-16, WebFetch — all 6 confirmed protein-free)
| # | source | verdict | new value | notes |
|---|---|---|---|---|
| 1 | Williams/Bevilacqua 2022 G4 | **[Y]** protein-free | **HIGH** | (G3A2)4 + flanking; MATCHED RNA negs (5'-C9 flank = no LLPS, 5'-U9 = weak). Suppl Table 1. K150/spermine, no protein. |
| 2 | Williams/Bevilacqua 2021 G4 | **[Y]** protein-free | **HIGH** | ~10 G-tract RNAs ((G3A)4,(G3A2)4,(G3A3)4,G3322/32/23...); negs = (G2A2)4 @ phys spermine, G2322, prokaryotic (no-spermine). Table S1, reconstructable. |
| 3 | Wadsworth 2025 2'-OH | [Y] protein-free | **LOW** | pos = known CAG/CUG repeats; negs = DNA + 2'-O-Me (MODIFIED backbones, NOT usable as ACGU RNA negatives). Little new. |
| 4 | Onuchic 2019 polyU | [Y] protein-free | MED | polyU + Mg2+/Ca2+/Sr2+/Zn2+ droplets; U10/A10 = partition-only (soft negs). polyU homopolymer (overlaps existing). |
| 5 | Aumiller 2016 polyU/spermine | [Y] protein-free | MED | polyU/spermine+spermidine coacervates (polyamine regime, allowed). Homopolymer. |
| 6 | Marianelli 2017 polyU/spermine+crowder | [Y] protein-free | MED | polyU/spermine + PEG/Ficoll; U15 partitions. Homopolymer + crowding. |

**Batch-1 net new mineable:** the **Williams G-quadruplex series (~25 RNA seqs WITH matched RNA negatives)** is the real catch — a NEW mechanism class (RNA-G4 LLPS) carrying the scarce confirmed negatives. The polyU/polyamine trio is valid protein-free non-yeast but overlaps existing homopolymer data. Wadsworth-2025 adds little (modified-backbone negatives only).

## BATCH 2 — wet-lab, sequences available
| # | source | DOI / PMC | claimed pos/neg | status |
|---|---|---|---|---|
| 7 | Schmoll, Novakovic & Allain 2025 — CAG-repeat condensate water-NMR | 10.1038/s41557-025-01968-9 / PMC12580330 | 5 / 0 | [ ] |
| 8 | O'Brien et al. 2024 — HTT exon-1 mRNA self-association (stick-slip) | 10.1038/s41467-024-52764-x / PMC11185545 | 2 / 3 | [ ] |
| 9 | Katz, Tolokh … Pollack 2017 — spermine condenses DNA not mixed RNA | 10.1016/j.bpj.2016.11.018 / PMC5232352 | 1 / 1 | [ ] |
| 10 | Huang, Kangovi … Niu — CZ/2CZ aptamer self-assembling hydrogel | 10.1021/acs.biomac.7b00314 / PMID 28609610 | 2 / 2 | [ ] |
| 11 | Zager … Plavec & Kragelj 2025 — (GGGGCC)48 gel fast-MAS NMR | bioRxiv 10.1101/2025.09.07.674584 | 1 / 0 | [ ] |
| 12 | Ahn et al. 2022 — RCT rG4 free-standing hydrogel | 10.1002/adma.202110424 / PMID 35263477 | 1 / 0 | [ ] |

## BATCH 3 — ribozyme/nanostar + computational (soft labels)
| # | source | DOI / PMC | claimed pos/neg | status |
|---|---|---|---|---|
| 13 | Giessler et al. 2026 — ribozyme-functionalized nanostar droplets | 10.1002/anie.202519002 / PMC12887606 | 10 / 4 | [ ] |
| 14 | Hauf et al. 2025 — ribozyme in protein-free yeast-RNA condensates (YEAST) | 10.1002/anie.202511332 / PMC12723469 | 2 / 0 | [ ] |
| 15 | Nguyen, Hori & Thirumalai 2022 — reptation dynamics repeats (computational) | 10.1038/s41557-022-00934-z | 5 / 2 | [ ] |
| 16 | Maity, Nguyen, Hori & Thirumalai 2023 — odd-even parity (computational) | 10.1073/pnas.2301409120 / PMC10268303 | 5 / 6 | [ ] |
| 17 | Maity, Hori & Thirumalai 2023 — salt-dependent self-association (computational) | 10.1021/acs.jpclett.3c03553 | 4 / 3 | [ ] |
| 18 | Kimchi, King & Brenner 2023 — reentrant aggregation theory (computational) | 10.1038/s41467-023-35803-x / PMC9852226 | ~ / ~ | [ ] |

## BATCH 2+3 RESULTS (verified 2026-06-16)
| # | source | verdict | value | notes |
|---|---|---|---|---|
| 7 | Schmoll/Allain CAG-NMR | [Y] protein-free | LOW | (CAG)10-44 LCST/Mg2+; CAG repeats already in corpus; neg = no-Mg condition |
| 8 | O'Brien HTT mRNA | **[N] not LLPS** | EXCLUDE | single-molecule force-spec inter-strand pairing, NOT bulk phase separation |
| 9 | Katz spermine RNA | [Y] protein-free | MED | poly(rA):poly(rU) condenses + **mixed-seq 25bp RNA stays SOLUBLE = real RNA negative** |
| 10 | CZ/2CZ aptamer hydrogel | [Y] protein-free | MED | SELEX aptamer self-assembles via 2 motifs; seqs in patent (CZ 99nt/2CZ 237nt); designed non-yeast |
| 11 | Zager (GGGGCC)48 | [Y] protein-free | LOW | G4C2 gel; (GGGGCC)n already in corpus (Raguseo) |
| 12 | Ahn rG4 hydrogel | [P] protein-free | LOW | rG4 RCT hydrogel; NO usable sequence in abstract |
| 13 | Giessler ribozyme nanostars | [Y] protein-free | MED | 4-arm KL nanostars (DrA/DrB/+ribozyme); KL family = OOD blind-spot; seqs in 55MB SI |
| 14 | Hauf yeast-RNA+ribozyme | [Y] protein-free | LOW | bulk YEAST total RNA (no defined non-yeast seqs) + 1 ribozyme |
| 15 | Nguyen 2022 reptation (comp) | [Y] computational | LOW | (CAG)n sim; no labeled +/- set |
| 16 | Maity PNAS parity (comp) | [Y] computational | MED | LABELED: (CAG)29/31 pos, (CAG)30 neg, M1/M2 ctrls (soft labels) |
| 17 | Maity JPCL salt (comp) | [Y] computational | MED | (CAG)30/31 pos vs scrambled-(CAG)31 neg (soft labels) |
| 18 | Kimchi theory (comp) | [Y] theory | LOW | sticker-spacer model, NO sequence labels |

## FINAL SYNTHESIS — verification of all 18 (3 batches, ~14 WebFetch, 0 agents)
- **Confirmed protein-free:** 17/18 (only O'Brien excluded — single-molecule pairing, not LLPS).
- **Materially NEW mineable data (the actual win):**
  1. **Williams/Bevilacqua G-quadruplex 2021+2022** — ~25 RNA-G4 sequences WITH matched RNA negatives (flanking/G-tract controls that don't condense). NEW non-yeast mechanism class + scarce negatives. **TOP PRIORITY to ingest.**
  2. **Katz mixed-sequence RNA negative** (stays soluble) — rare clean RNA negative.
  3. **CZ/2CZ aptamer** — new designed protein-free RNA self-assembler (patent seqs).
  4. **Maity ×2 computational** — CAG parity + scramble soft-labels.
- **Low/overlap (skip or deprioritize):** CAG/CUG/G4C2 repeats (already in corpus: Schmoll, Zager, Wadsworth-2025, Nguyen), bulk-yeast (Hauf), no-usable-seq (Ahn), KL-nanostars (Giessler — OOD blind-spot family).
- **HONEST BOTTOM LINE:** the deep sweep's high-value wet-lab tail is finite. The one genuinely new, directly-useful, NON-YEAST class is **RNA G-quadruplex LLPS (Williams/Bevilacqua) with matched negatives.** Everything else overlaps the corpus, is computational soft-labels, or is unusable. Mining the Williams G4 set (+Katz neg, +CZ aptamer) is the actionable next step; it will NOT dramatically rebalance the corpus but it adds a clean non-yeast mechanism with real negatives.

## INGESTED → v11 staging (2026-06-16)
Sequences retrieved + adversarially verified (anti-fabrication + protein-free + label) by Workflow `wf_ca674e7d-2d1`, then assembled by `scripts/data_prep/assemble_v11_additions.py` →
- `Data/raw/multispecies/staging/v11_additions.fasta` (26 recs) + `v11_additions.SOURCE.md` (provenance/label manifest).
- **10 POS + 8 NEG (all 18 NEW vs v5/v6 pool) + 8 SOFT (held) + 2 EXCLUDED.** All non-yeast, protein-free.
- Integrity: excluded ssA25 (verifier-rejected: duplex condenses, not ssRNA) AND ssU25 (same logic; would contradict existing rU20 neg). 4 `needs_manual` label-disputes demoted to SOFT, not forced.
- **STATUS: staged only, NOT retrained.** Per v10 net-negative lesson, fold POS/NEG into a pool + rebuild features + evaluate as ONE batch vs v6 before any promotion.
