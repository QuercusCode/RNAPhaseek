"""
Yeast (Saccharomyces cerevisiae) LLPS-positive RNA collection.

Canonical literature for yeast LLPS / RNP granule transcriptomes:

  1. Hubstenberger A. et al. (2017) Cell  -- "P-Body Purification Reveals
     the Condensation of Repressed mRNA Regulons"
     -> Supplementary tables list mRNAs enriched in P-bodies (~700 entries)
     -> Source for systematic IDs

  2. Mitchell SF. et al. (2013) Nat Struct Mol Biol -- "Global analysis of
     yeast mRNPs"
     -> Identifies P-body and stress-granule mRNAs

  3. Buchan JR. et al. (2008) JCB -- first yeast SG paper
     -> Smaller dataset, foundational

  4. Cherry P. et al. (2018) -- Heat-stress SG (GEO GSE107797)
     -> RNA-seq of yeast SG cores

  5. Begovich K. & Wilhelm JE. (2020) -- SG in yeast under glucose starvation

How to use:
  1. Download the supplementary table(s) from each paper (PDF/xlsx, manual step).
  2. Extract systematic IDs (YAL001C, ... format) or standard gene names.
  3. Append them to the corresponding gene_list_<paper>() function below.
  4. Run this script. It fetches yeast ORFs from SGD and writes a FASTA.

Output:
  Data/raw/multispecies/yeast_positives.fasta
"""

import argparse
import os
from .species_cdna_fetcher import fetch_yeast_orf_from_sgd, write_positives_fasta


# ── Curated gene lists per paper ─────────────────────────────────────────────
# Each list should be filled with systematic IDs (YBR275C, ...) or standard names
# (HSP104, PUB1, ...) extracted from the paper's supplementary table.

def gene_list_hubstenberger_2017() -> list[str]:
    """
    P-body-enriched mRNAs (Hubstenberger et al. 2017, Cell).
    Supplementary table available at the Cell Press paper page.

    Returns a list of systematic IDs / standard names.

    TODO: paste the gene list from the supplementary table here.
    A high-confidence subset (top ~200 most-enriched mRNAs) is a good
    starting point.
    """
    # Example seed entries (canonical P-body mRNAs); to be expanded with
    # the actual supplementary-table contents.
    return [
        "PUB1", "PAB1", "DCP1", "DCP2", "DHH1", "EDC3", "LSM1", "PAT1", "SCD6",
        "EDC1", "EDC2", "XRN1", "CDC33", "EIF4G1", "EIF4G2",
        # ... append the supplementary table here
    ]


def gene_list_mitchell_2013() -> list[str]:
    """Mitchell et al. 2013, Nat Struct Mol Biol -- mRNP enrichment."""
    return [
        # TODO: paste curated systematic IDs from the paper
    ]


def gene_list_buchan_2008() -> list[str]:
    """Buchan et al. 2008, JCB -- early yeast stress-granule mRNAs."""
    return [
        # Classic SG mRNAs identified
        "TIF4631", "TIF4632", "PAB1", "PUB1", "EIF4G1", "EIF4G2", "RPL30",
    ]


def gene_list_cherry_2018() -> list[str]:
    """Cherry et al. 2018 -- heat-stress SG transcripts (GSE107797)."""
    return [
        # TODO: paste enriched-transcript list
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(out_dir: str = "Data/raw/multispecies") -> None:
    sources = {
        "Hubstenberger2017": gene_list_hubstenberger_2017(),
        "Mitchell2013":      gene_list_mitchell_2013(),
        "Buchan2008":        gene_list_buchan_2008(),
        "Cherry2018":        gene_list_cherry_2018(),
    }

    # Deduplicate while keeping (gene_id, source) pairings
    seen = set()
    pairs = []
    for src, lst in sources.items():
        for gid in lst:
            if gid not in seen:
                seen.add(gid)
                pairs.append((gid, src))

    print(f"Yeast LLPS gene-list size: {len(pairs)} (across {len(sources)} papers)")

    records = []
    for gid, src in pairs:
        seq, key = fetch_yeast_orf_from_sgd(gid)
        if seq:
            records.append((gid, key, seq, src))

    out_path = os.path.join(out_dir, "yeast_positives.fasta")
    n = write_positives_fasta(records, out_path, "yeast")
    print(f"Wrote {n} yeast LLPS-positive sequences -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw/multispecies")
    args = parser.parse_args()
    main(args.out_dir)
