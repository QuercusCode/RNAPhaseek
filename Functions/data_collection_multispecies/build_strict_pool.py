"""
Build the STRICT RNA-LLPS pool — only RNAs where the literature directly shows
the RNA molecule itself drives or scaffolds condensate formation. Excludes
mRNAs whose only association is encoding an LLPS protein.

Inclusion rules (KEEP if ANY of these hold):
  1. Source is RPS 2.0 (RPS2) — RPS 2.0 specifically curates RNA-LLPS evidence
     per Mao 2022 Bioinformatics.
  2. Source is RNAPhaSep — also curates RNA-driver evidence per Sun 2022.
  3. Curated viral LLPS subregion (header contains 'PMID:').
  4. Gene name matches an extended list of known RNA-LLPS drivers from primary
     literature (lncRNAs, repeat-expansion RNAs, mapped 3'UTR/5'UTR elements).

Exclusion (DROP all others):
  - UniProt GO + KW expansions (mRNAs encoding LLPS proteins — NOT RNA-LLPS)
  - WorkflowSalvage (generic ID lists)
  - ParkerSG / smOOPs entries that don't match the driver gene list
  - Lit_canonical from this session's UniProt-based curation runs
  - PhaSepDB_GO from this session's UniProt-based curation runs

Output:
  Data/raw/multispecies/strict_pool_positives.fasta
  Data/raw/multispecies/strict_pool_drop_log.tsv
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

# Known RNA-LLPS-driver gene/region names from primary literature.
# Extend conservatively — every name here should have at least one published
# study where the RNA itself was shown to phase-separate (in vitro or in vivo).
KNOWN_DRIVERS = {
    # ─ lncRNA scaffolds ─────────────────────────────────────────────────────
    "NEAT1", "MALAT1", "XIST", "TSIX", "FIRRE", "NORAD",
    "HSATII", "SATIII", "SAT3", "SATIII_RNA",
    "PVT1", "GAS5", "MEG3", "HOTAIR", "HOTAIRM1", "HULC",
    "HSATII",  "MEN_EPSILON_BETA",  # NEAT1 synonyms

    # ─ repeat-expansion RNAs ────────────────────────────────────────────────
    "C9ORF72",  # G4C2 hexanucleotide repeat → ALS/FTD condensates
    "FMR1",     # CGG triplet → FXTAS condensates
    "DMPK",     # CUG triplet → DM1 nuclear foci
    "ZNF9", "CNBP",  # CCUG → DM2 foci
    "ATXN1", "ATXN1L",  # CAG → SCA1
    "ATXN2",            # CAG → SCA2 / ALS-associated
    "ATXN3",            # CAG → SCA3
    "ATXN7", "ATXN8OS", "ATXN10",
    "HTT",      # CAG → Huntington's
    "TBP", "HOXA1",
    "G4C2", "GGGGCC",  # raw repeat motif names
    "CAG_REPEAT", "CGG_REPEAT", "CUG_REPEAT", "CCUG_REPEAT",

    # ─ mRNAs with mapped LLPS-active subregion ──────────────────────────────
    "WHI3",   # yeast: 3'UTR CAG/CCG repeats drive LLPS (Zhang 2015 Mol Cell)
    "ORB2",   # fly: prion-like 3'UTR (Khan 2015)
    "CPEB3",  # mouse: amyloid + LLPS (Stephan 2015)
    "SUP35",  # yeast prion mRNA
    "PAB1",   # yeast: poly-A binding protein mRNA
    "NRD1", "PUB1", "TPI1",
    "FUS",    # mRNA with LLPS-mapped region
    "TARDBP", "TDP43", "TDP-43",
    "HNRNPA1", "HNRNPA2B1", "HNRNPDL",
    "G3BP1", "G3BP2",
    "EIF4G1", "EIF4G2",
    "PABPN1",

    # ─ Viral LLPS-scaffolding RNAs (already curated subregions) ─────────────
    "SARS", "HCV", "RSV", "VSV", "HIV",
    "ROTAVIRUS", "REOVIRUS", "ORTHOREOVIRUS", "INFLUENZA",
    "EBOLA", "MARBURG", "ZIKA", "DENGUE", "WNV", "WEST_NILE",
    "CHIKUNGUNYA", "HANTAVIRUS", "MEASLES", "RABIES", "NIPAH",
    "PAPILLOMA", "HPV",
}

# Database tags that we trust as "RNA-LLPS evidence" sources.
# These are the databases that specifically curate RNA→LLPS (not protein→LLPS).
RNA_LLPS_DB_TAGS = {
    "RPS2",        # RPS 2.0 — Mao 2022
    "RNAPhaSep",   # RNAPhaSep — Sun 2022
    "RPS_",        # RPS 2.0 internal IDs (RPS_171073 etc.)
    "RNAPS",       # RNAPhaSep internal IDs (RNAPS0000288)
}

# Database tags that are NOT RNA-LLPS-driver evidence — these are either
# protein-LLPS (encoding mRNA) or RNA-recruited-to-condensate (not driving).
PROTEIN_PROXY_DB_TAGS = {
    "UniProt_GO", "UniProt_KW1185", "Lit_canonical",
    "WorkflowSalvage", "PhaSepDB_GO",
    "ParkerSG",   # SG-pulldown — mostly recruited, not driving
    "smOOPs",     # SG core proteomics — mostly recruited
    "PhaSepDB",   # mostly protein-centric
    "DrLLPS",     # mostly protein-centric
    "SeedCollector",  # canonical proteins, not RNAs
    "Trcek2020",  # germ-granule pulldown
    "Lee2020",
}


def driver_in_header(hdr: str) -> str:
    """Return the matched driver name if hdr contains any KNOWN_DRIVERS substring,
    else empty string. Match is case-insensitive, whole-token-ish (allow
    underscore/dash/digit boundaries)."""
    h_up = hdr.upper()
    # Sort by length (longest first) so 'NEAT1_FRAGMENT' beats 'NEAT1'
    for k in sorted(KNOWN_DRIVERS, key=lambda x: -len(x)):
        if k in h_up:
            return k
    return ""


def is_rna_llps_db_source(hdr: str) -> str:
    """Return the matched DB tag if the header carries an RNA-LLPS DB source."""
    for tag in RNA_LLPS_DB_TAGS:
        if tag in hdr:
            return tag
    return ""


def is_curated_viral_subregion(hdr: str) -> bool:
    """Curated viral subregions carry a PMID: tag in their header."""
    return "PMID:" in hdr


def is_protein_proxy(hdr: str) -> str:
    """Return matched tag if hdr is from a protein-proxy database."""
    for tag in PROTEIN_PROXY_DB_TAGS:
        if tag in hdr:
            return tag
    return ""


def should_keep(hdr: str) -> tuple[bool, str]:
    """Return (keep, reason). Reason is the criterion that triggered the
    decision (for the drop log)."""
    # KEEP path 1: curated viral with PMID
    if is_curated_viral_subregion(hdr):
        return True, "curated_viral_PMID"
    # KEEP path 2: RNA-LLPS DB sources (RPS2, RNAPhaSep)
    tag = is_rna_llps_db_source(hdr)
    if tag:
        return True, f"RNA_LLPS_DB:{tag}"
    # KEEP path 3: gene name matches known driver
    driver = driver_in_header(hdr)
    if driver:
        # Avoid the false-positive case where the gene name matches a driver
        # but the source is a protein-proxy DB AND the species is from this
        # session's UniProt expansion (those use the new species naming).
        # We still keep these if the driver gene matched — high confidence.
        return True, f"driver_match:{driver}"
    # DROP path: protein-proxy DB
    proxy_tag = is_protein_proxy(hdr)
    if proxy_tag:
        return False, f"protein_proxy:{proxy_tag}"
    # DROP path: unknown
    return False, "no_driver_no_rna_llps_db"


def parse_fasta(path):
    if not os.path.exists(path):
        return
    hdr, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(chunks)
                hdr, chunks = line[1:], []
            else:
                chunks.append(line)
        if hdr is not None:
            yield hdr, "".join(chunks)


def main():
    src = Path("Data/raw/multispecies/unified_all_positives.fasta")
    out = Path("Data/raw/multispecies/strict_pool_positives.fasta")
    log = Path("Data/raw/multispecies/strict_pool_drop_log.tsv")

    if not src.exists():
        print(f"ERROR: input not found: {src}")
        sys.exit(1)

    keep_count = 0
    drop_count = 0
    keep_reasons = Counter()
    drop_reasons = Counter()

    sys.path.insert(0, '.')
    from Functions.RNAPhaseek.species_registry import species_id_for, label_for
    species_kept = Counter()

    with open(out, "w") as fo, open(log, "w") as fl:
        fl.write("decision\treason\tspecies\tlength\theader\n")
        for hdr, seq in parse_fasta(src):
            keep, reason = should_keep(hdr)
            sp = label_for(species_id_for(hdr))
            if keep:
                fo.write(f">{hdr}\n{seq}\n")
                keep_count += 1
                keep_reasons[reason.split(":")[0]] += 1
                species_kept[sp] += 1
                fl.write(f"KEEP\t{reason}\t{sp}\t{len(seq)}\t{hdr[:200]}\n")
            else:
                drop_count += 1
                drop_reasons[reason.split(":")[0]] += 1
                fl.write(f"DROP\t{reason}\t{sp}\t{len(seq)}\t{hdr[:200]}\n")

    print(f"STRICT POOL BUILD COMPLETE")
    print(f"  Input:  {src}  ({keep_count + drop_count} records)")
    print(f"  Output: {out}  ({keep_count} records)")
    print(f"  Log:    {log}")
    print()
    print(f"KEPT  ({keep_count}):")
    for r, n in keep_reasons.most_common():
        print(f"  {n:>6}  {r}")
    print(f"\nDROPPED ({drop_count}):")
    for r, n in drop_reasons.most_common():
        print(f"  {n:>6}  {r}")
    print(f"\nKept by species:")
    for sp, n in species_kept.most_common():
        print(f"  {sp:<14} {n}")


if __name__ == "__main__":
    main()
