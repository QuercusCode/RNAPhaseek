"""
Drosophila melanogaster LLPS-positive RNA collection.

Canonical literature for Drosophila germ granule / polar granule mRNAs:

  1. Trcek T. et al. (2020) Nat Comm -- "Sequence-independent self-assembly
     of germ granule mRNAs into homotypic clusters"
     -> Supplementary tables list germ-granule-enriched mRNAs
     -> GEO accession GSE154236

  2. Eichler CE. et al. (2020) Genes Dev -- germ-granule-associated mRNAs
     -> Smaller curated set

  3. Niepielko MG. et al. (2018) Curr Biol -- nanos / pgc / gcl mRNAs in
     germ granules

  4. Lehmann lab classic papers (Trcek, Vinter et al.) -- germ-cell-specific
     mRNAs known to phase-separate

Output:
  Data/raw/multispecies/drosophila_positives.fasta
"""

import argparse
import os
from .species_cdna_fetcher import (
    fetch_cdna_by_ensembl_gene, fetch_cdna_by_symbol, write_positives_fasta
)


def gene_list_trcek_2020() -> list[tuple[str, str]]:
    """
    Germ-granule mRNAs (Trcek et al. 2020, Nat Comm).
    Source: GEO GSE154236 + supplementary tables.

    TODO: paste extracted gene list from the paper's supplementary table.
    """
    # Seed set of well-known Drosophila germ-granule mRNAs (manual curation)
    return [
        ("nos",   "symbol"),     # nanos
        ("pgc",   "symbol"),     # polar granule component
        ("gcl",   "symbol"),     # germ cell-less
        ("osk",   "symbol"),     # oskar
        ("vas",   "symbol"),     # vasa
        ("tud",   "symbol"),     # tudor
        ("aub",   "symbol"),     # aubergine
        ("piwi",  "symbol"),
        ("Hrb27C","symbol"),     # heterogeneous nuclear RNP at 27C
        ("Cup",   "symbol"),     # Cup eIF4E-binding
        # TODO: append the supplementary table contents
    ]


def gene_list_eichler_2020() -> list[tuple[str, str]]:
    """Eichler et al. 2020, Genes Dev."""
    return [
        # TODO: paste curated entries
    ]


def main(out_dir: str = "Data/raw/multispecies") -> None:
    sources = {
        "Trcek2020":   gene_list_trcek_2020(),
        "Eichler2020": gene_list_eichler_2020(),
    }

    seen = set()
    pairs = []
    for src, lst in sources.items():
        for (ident, kind) in lst:
            key = (ident.lower(), kind)
            if key not in seen:
                seen.add(key)
                pairs.append((ident, kind, src))

    print(f"Drosophila LLPS gene-list size: {len(pairs)} (across {len(sources)} papers)")

    records = []
    for i, (ident, kind, src) in enumerate(pairs, 1):
        if kind == "fbgn":
            seq, tx = fetch_cdna_by_ensembl_gene(ident, "drosophila_melanogaster")
        else:
            seq, tx = fetch_cdna_by_symbol(ident, "drosophila_melanogaster")
        if seq:
            records.append((ident, tx, seq, src))
        if i % 10 == 0:
            print(f"  [{i}/{len(pairs)}] fetched {len(records)} so far", flush=True)

    out_path = os.path.join(out_dir, "drosophila_positives.fasta")
    n = write_positives_fasta(records, out_path, "drosophila")
    print(f"Wrote {n} Drosophila LLPS-positive sequences -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw/multispecies")
    args = parser.parse_args()
    main(args.out_dir)
