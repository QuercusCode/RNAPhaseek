# Multi-species RNAPhaseek data collection

This module extends the single-species (mostly human) training pool with
positives + species-matched negatives for mouse, yeast, *C. elegans*,
Drosophila, viruses, and (optionally) plants. The goal is a foundation-model
predictor of RNA LLPS that generalizes across the major eukaryotic
condensate categories rather than just human mRNA stress granules.

## Pipeline overview

```
   1. download_reference_transcriptomes.py
        -> Data/raw/multispecies/refs/{species}_cdna.fa.gz
        (mouse, yeast, worm, fly; for the species-matched negative pool)

   2. collect_<species>_llps.py
        -> Data/raw/multispecies/{species}_positives.fasta
        (per-species LLPS-positive RNAs from canonical literature)

   3. prepare_multispecies_negatives.py
        -> Data/raw/multispecies/negatives/{species}_negatives.fasta
        (GC + length-matched negatives sampled from each species' reference)

   4. unified_merge (TODO)
        -> Data/raw/multispecies/unified_all_{positives,negatives}.fasta
        (one big pool, CD-HIT-EST-deduplicated, with species labels)
```

## Per-species sources

### Mouse — `download_smoops_mouse_native.py`

Re-extracts the smOOPs (Ivanov 2025, *Cell Genomics*) dataset directly as
mouse cDNA, without the human-ortholog mapping the original downloader
performed. Yields ~1,800 mouse-native LLPS-positive mRNAs.

### Yeast — `collect_yeast_llps.py`

Canonical papers documented in the scaffold's gene_list_<paper>() stubs:

- Hubstenberger A. et al. (2017) *Cell* — P-body mRNAs
- Mitchell SF. et al. (2013) *Nat Struct Mol Biol* — yeast mRNPs
- Buchan JR. et al. (2008) *JCB* — yeast stress granules
- Cherry P. et al. (2018) — heat-shock SG (GEO GSE107797)
- Begovich K. & Wilhelm JE. (2020) — glucose-starvation SG

Sequence lookup against SGD `orf_coding_all.fasta` (in-memory index of both
systematic IDs and standard names).

### C. elegans — `collect_celegans_llps.py`

Canonical papers:

- Lee CYS. & Putnam A. et al. (2020) *Cell Reports* — P-granule mRNAs
- Knutson AK. et al. (2017) *Mol Cell* — PGL-1 binding partners
- Updike DL. & Strome S. (2010) — earlier P-granule transcriptome
- Wang JT. et al. (2014) — MEG-3 / PGL-1 / P-granule mRNAs

Sequence lookup via Ensembl REST `/lookup/symbol/caenorhabditis_elegans/<name>`
and `/sequence/id/<transcript>`.

### Drosophila — `collect_drosophila_llps.py`

Canonical papers:

- Trcek T. et al. (2020) *Nat Comm* — germ granule mRNAs (GEO GSE154236)
- Eichler CE. et al. (2020) *Genes Dev* — germ-granule-associated mRNAs
- Niepielko MG. et al. (2018) *Curr Biol* — nanos / pgc / gcl

Sequence lookup via Ensembl REST `/lookup/symbol/drosophila_melanogaster/<name>`.

### Viruses — `collect_viral_llps.py`

Hardcoded NCBI accessions for canonical viral condensate systems:

- SARS-CoV-2 (N protein RNP)  · NC_045512.2
- HCV H77 (NS5A condensate)   · NC_038882.1
- RSV A (inclusion bodies)    · NC_038235.1
- VSV Indiana                 · NC_001560.1
- HIV-1 HXB2                  · K03455.1
- Rotavirus A segments 7, 8   · DQ490542.1, DQ490543.1
- Reovirus T3 segment S2      · AF378003.1
- Influenza A PR8 segment 8   · NC_002020.1

## Extending with new supplementary tables

When you obtain a paper's supplementary gene-list xlsx/csv/tsv:

1. Save the file under `Data/raw/multispecies/papers/<paper>.xlsx`.
2. Use `parse_supplementary.py` to extract IDs:
   ```
   python -m Functions.data_collection_multispecies.parse_supplementary \\
       --file Data/raw/multispecies/papers/hubstenberger_2017.xlsx \\
       --species yeast
   ```
3. Paste the printed IDs into the corresponding `gene_list_<paper>()`
   function in the species' `collect_<species>_llps.py`.
4. Re-run the collector:
   ```
   python -m Functions.data_collection_multispecies.collect_yeast_llps
   ```
5. Re-run negatives:
   ```
   python -m Functions.data_collection_multispecies.prepare_multispecies_negatives
   ```

## Header conventions

All multispecies FASTAs follow a uniform pipe-delimited header so a downstream
species detector can parse the species label from any record:

```
>llps_{species}|{primary_id}|{transcript_id}|{source_tag}
>neg_{species}|{transcript_id}|{gene_symbol}|len={L}|gc={gc}
```

Where `species` ∈ {human, mouse, yeast, celegans, drosophila, virus,
arabidopsis, unknown}.

## Output files (current)

```
Data/raw/multispecies/
├── refs/                                   (~80 MB)
│   ├── mus_musculus_cdna.fa.gz                  (48 MB)
│   ├── caenorhabditis_elegans_cdna.fa.gz        (13 MB)
│   ├── drosophila_melanogaster_cdna.fa.gz       (17 MB)
│   └── saccharomyces_cerevisiae_cdna.fa.gz      (3.7 MB)
├── smoops_mouse_positives.fasta                  1,825 seqs
├── yeast_positives.fasta                            (seed set; expandable)
├── celegans_positives.fasta                         (seed set; expandable)
├── drosophila_positives.fasta                       (seed set; expandable)
├── viral_positives.fasta                       9 viral RNAs
└── negatives/
    ├── mouse_negatives.fasta                    1,678 matched
    ├── yeast_negatives.fasta                       (matches positives)
    ├── celegans_negatives.fasta                    (matches positives)
    └── drosophila_negatives.fasta                  (matches positives)
```
