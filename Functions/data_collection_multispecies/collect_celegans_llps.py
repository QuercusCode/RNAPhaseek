"""
Caenorhabditis elegans LLPS-positive RNA collection.

Canonical literature for C. elegans P-granule / RNP transcriptomes:

  1. Lee CYS. & Putnam A. et al. (2020) Cell Reports -- "Recruitment of mRNAs
     to P granules by condensation with intrinsically-disordered proteins"
     -> Supplementary tables list P-granule-enriched mRNAs (~1,000 entries)
     -> Most comprehensive single source

  2. Knutson AK. et al. (2017) Mol Cell -- PGL-1 and P-granule assembly
     -> Smaller but high-confidence set

  3. Updike DL. & Strome S. (2010) -- earlier P-granule transcriptome
     -> Foundational; smaller dataset

  4. Wang JT. et al. (2014) -- MEG-3 / PGL-1 / P-granule mRNAs

How to use:
  1. Download the supplementary table(s) from each paper (xlsx, manual step).
  2. Extract Ensembl IDs (WBGene...) or symbols.
  3. Paste them into the gene_list_<paper>() functions below.
  4. Run this script.

Output:
  Data/raw/multispecies/celegans_positives.fasta
"""

import argparse
import os
from .species_cdna_fetcher import (
    fetch_cdna_by_ensembl_gene, fetch_cdna_by_symbol, write_positives_fasta
)


def gene_list_lee_putnam_2020() -> list[tuple[str, str]]:
    """
    P-granule-enriched mRNAs (Lee, Putnam, et al. 2020, Cell Reports).
    Each entry is (identifier, kind) where kind is 'wbgene' or 'symbol'.

    Source: supplementary data file (xlsx) from Cell Reports paper.
    TODO: paste extracted gene list here.
    """
    # Seed set of well-known C. elegans P-granule mRNAs (manual from literature)
    return [
        ("pgl-1",   "symbol"),
        ("pgl-3",   "symbol"),
        ("meg-3",   "symbol"),
        ("meg-4",   "symbol"),
        ("car-1",   "symbol"),
        ("vbh-1",   "symbol"),
        ("glh-1",   "symbol"),
        ("glh-4",   "symbol"),
        ("nos-2",   "symbol"),
        ("nos-3",   "symbol"),
        ("oma-1",   "symbol"),
        ("oma-2",   "symbol"),
        ("mex-3",   "symbol"),
        ("mex-5",   "symbol"),
        ("mex-6",   "symbol"),
        # TODO: append the supplementary table contents
    ]


def gene_list_knutson_2017() -> list[tuple[str, str]]:
    """Knutson et al. 2017 -- PGL-1 binding partners."""
    return [
        # TODO: paste from supplementary
    ]


def main(out_dir: str = "Data/raw/multispecies") -> None:
    sources = {
        "Lee2020":     gene_list_lee_putnam_2020(),
        "Knutson2017": gene_list_knutson_2017(),
    }

    # Deduplicate while preserving (id, kind, source)
    seen = set()
    pairs = []
    for src, lst in sources.items():
        for (ident, kind) in lst:
            key = (ident.lower(), kind)
            if key not in seen:
                seen.add(key)
                pairs.append((ident, kind, src))

    print(f"C. elegans LLPS gene-list size: {len(pairs)} (across {len(sources)} papers)")

    records = []
    for i, (ident, kind, src) in enumerate(pairs, 1):
        if kind == "wbgene":
            seq, tx = fetch_cdna_by_ensembl_gene(ident, "caenorhabditis_elegans")
        else:
            seq, tx = fetch_cdna_by_symbol(ident, "caenorhabditis_elegans")
        if seq:
            records.append((ident, tx, seq, src))
        if i % 25 == 0:
            print(f"  [{i}/{len(pairs)}] fetched {len(records)} so far", flush=True)

    out_path = os.path.join(out_dir, "celegans_positives.fasta")
    n = write_positives_fasta(records, out_path, "celegans")
    print(f"Wrote {n} C. elegans LLPS-positive sequences -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw/multispecies")
    args = parser.parse_args()
    main(args.out_dir)
