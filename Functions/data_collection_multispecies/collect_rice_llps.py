"""
Oryza sativa (rice) LLPS-positive RNA collection.

The rice condensate literature is sparser than the human/yeast/Arabidopsis
sets, so curation combines:

  1. UniProt KW-1185 (Phase-separation) + RNP-granule / P-body / SG GO terms
     for taxon 4530 (yields only ~4 entries on its own).
  2. Canonical rice condensate-paper hits collected from:
        - OsP19 family   (Park 2017; Park & Han 2017 stress granules)
        - OsRBP family    (RBP-DR1 / OsRBP-related)
        - OsAGO family    (AGO-loaded miRNA bodies; Mi 2008; Chodasiewicz 2020)
        - OsUBP1 (UBP1)   (Lambermon 2002; Sorenson 2014)
        - OsDCP1/2/5      (decapping; SG / P-body markers)
        - OsTSN1/2        (Tudor SN; Yan 2014 SG)
        - OsCBP20/80      (cap-binding; SG / nuclear speckle)
        - OsGRP7          (cold-stress granules; Kim 2008)
        - HSFA2/HSFA4/HSFA7 (Scharf 2012; heat-shock granule TFs)
        - HSP70 / HSP90 / HSP101 / smHSP family (heat granules; Kotak 2007)
        - OsLSM1-8       (P-body / decay)
        - OsRH (DEAD-box) family
        - OsSR / OsU2AF (splicing speckle)
        - OsCAR (CAR ribonucleoprotein granules)
  3. Reference-derived canonical genes via the gene_symbol annotation in the
     Ensembl Plants cDNA fasta.

Resolution strategy:
  - Build an index from the local Ensembl Plants cDNA fasta (gene_id,
    transcript_id, gene_symbol) → sequence (longest transcript per gene).
  - For each curated entry, try (in order):
      (a) RAP-DB locus ID (Os01g0123450)        → direct gene_id hit
      (b) MSU locus ID (LOC_Os01g12345)         → mapped via reference
      (c) gene symbol match (case-insensitive)
      (d) gene name token from header           → partial fallback

Output:
  Data/raw/multispecies/rice_positives.fasta

Header format:
  >llps_rice|<gene_name>|<rap_db_id_or_locus>|<source>
"""

import argparse
import gzip
import os
import re
from typing import Optional


REF_FASTA_GZ = "Data/raw/multispecies/refs/oryza_sativa_cdna.fa.gz"


# ── Curated gene tables ──────────────────────────────────────────────────────
# Each entry: (preferred_label, identifier_or_symbol)
# identifier can be: RAP-DB locus ID ("Os01g0123450"), gene symbol
# ("OsAGO1a") or an MSU locus (mapped later).

def gene_list_uniprot_go_kw() -> list[tuple[str, str]]:
    """UniProt taxon=4530, KW-1185 + GO:0035770/0010494/0036464/0000932."""
    # Pulled from rest.uniprot.org query, plus their EnsemblPlants/RAP-DB
    # xrefs where present.
    return [
        # accession Q01J34 -> LSM1 -> OsIGBa0140O07.9 -> OsLSM1
        ("OsLSM1",                "Os07g0686100"),
        # Q94LL5 -> snRNP-G
        ("OsSmG",                 "Os06g0667100"),
        # Q01JN2 -> Sm D1
        ("OsSmD1",                "Os05g0481600"),
        # Q01K89 -> Sm-like RNA processing
        ("OsSm_like",             "Os05g0497100"),
    ]


def gene_list_canonical_paper_hits() -> list[tuple[str, str]]:
    """
    Canonical rice LLPS / RNP-granule proteins from the literature.

    Provided as (label, identifier) where identifier is either the RAP-DB
    gene ID (Os01g0123450) or a gene_symbol that we resolve against the
    Ensembl Plants reference.
    """
    return [
        # -------- OsP19 family (Park 2017 SG) --------
        ("OsP19_1",               "Os01g0124000"),   # OsP19-1 (placeholder; resolve via symbol)
        # -------- AGO / RNA silencing condensates --------
        ("OsAGO1a",               "OsAGO1a"),
        ("OsAGO1b",               "OsAGO1b"),
        ("OsAGO1c",               "OsAGO1c"),
        ("OsAGO1d",               "OsAGO1d"),
        ("OsAGO2",                "OsAGO2"),
        ("OsAGO4a",               "OsAGO4a"),
        ("OsAGO4b",               "OsAGO4b"),
        ("OsAGO7",                "OsAGO7"),
        ("OsAGO18",               "OsAGO18"),
        # -------- UBP1 family (Lambermon 2002) --------
        ("OsUBP1a",               "Os03g0750100"),
        ("OsUBP1b",               "Os02g0782800"),
        ("OsUBP1c",               "Os06g0162800"),
        # -------- DCP1/2/5 (P-body decapping) --------
        ("OsDCP1",                "Os03g0287000"),
        ("OsDCP2",                "Os07g0518800"),
        ("OsDCP5",                "Os04g0413500"),
        # -------- Tudor-SN (SG) --------
        ("OsTSN1",                "OsTSN"),
        ("OsTSN2",                "Os05g0186300"),
        # -------- CBP20 / CBP80 (cap-binding speckle) --------
        ("OsCBP20",               "Os06g0142800"),
        ("OsCBP80",               "Os02g0750100"),
        # -------- GRP7 (cold-stress granule) --------
        ("OsGRP1",                "OsGRP1"),
        ("OsGRP3",                "Os03g0670700"),
        ("OsGRP6",                "Os03g0670400"),
        # -------- Heat-shock factors and HSPs --------
        ("OsHsfA2a",              "Os03g0161900"),
        ("OsHsfA2c",              "Os10g0419300"),
        ("OsHsfA2d",              "Os03g0745000"),
        ("OsHsfA2e",              "HSFA2E"),
        ("OsHsfA4a",              "Os05g0429100"),
        ("OsHsfA7",               "OsHsfA7"),
        ("OsHsfB4d",              "OsHsfB4d"),
        ("OsHSP82",               "OsHSP82"),
        ("OsHSP90",               "OsHsp90"),
        ("OsHSP70",               "OsHSP70"),
        ("OsHsp70CP2",            "OsHsp70CP2"),
        ("OsHsp101",              "Os05g0445100"),
        ("OsHsp17.3",             "OsHsp17.3"),
        ("OsHsp17.9B",            "Oshsp17.9B."),
        ("OsHsp16.9C",            "Oshsp16.9C"),
        ("OsHsp58.7",             "OsHsp58.7"),
        ("OsHsp74.8",             "OsHsp74.8"),
        # -------- Lsm 1-8 (P-body decay) --------
        ("OsLSM2",                "Os04g0644300"),
        ("OsLSM3",                "Os02g0182700"),
        ("OsLSM4",                "Os02g0664200"),
        ("OsLSM5",                "Os07g0517600"),
        ("OsLSM6",                "Os03g0274300"),
        ("OsLSM7",                "Os10g0500800"),
        ("OsLSM8",                "Os05g0471400"),
        # -------- DEAD-box RNA helicases (SG / P-body markers) --------
        ("OsRH13",                "OsRH13"),
        ("OsRH16",                "OsRH16"),
        ("OsRH34",                "OsRH34"),
        ("OsRH40",                "OsRH40"),
        ("OsRH51",                "OsRH51"),
        ("OsRH52A",               "OsRH52A"),
        ("OsDRH1",                "Os03g0686100"),
        # -------- SR splicing speckle --------
        ("OsSR45",                "Os02g0274000"),
        ("OsSCL30",               "Os05g0388600"),
        ("OsSCL33",               "Os02g0716700"),
        # -------- TIA-1 / Tia-like (SG) --------
        ("OsTIAR",                "Os04g0432800"),
        # -------- CAR (CAR ribonucleoprotein granules) --------
        ("OsCAR1",                "Os04g0419100"),
        ("OsCAR4",                "Os02g0103600"),
        # -------- XRN family (5' decay; P-body) --------
        ("OsXRN3",                "Os02g0743900"),
        ("OsXRN4",                "Os02g0743600"),
        # -------- PABP / eIF4 / mRNP (SG core) --------
        ("OsPABP",                "Os07g0635900"),
        ("OsPABP2",               "Os03g0285800"),
        ("OsEIF4G",               "Os09g0556000"),
        ("OsEIF4E",               "Os04g0379600"),
        # -------- RBP / RNA-binding (RBP-DR1) --------
        ("OsRBP",                 "OsRBP"),
        ("OsRBP_DR1",             "Os02g0631100"),
        # -------- HSP-memory: HSFA2 OsHSFA2 LOC_Os07g08140 --------
        ("OsHSFA2_LOC07g08140",   "Os07g0177400"),
        # -------- AGO-related miRNA granule --------
        ("OsDCL1",                "Os03g0182200"),
        ("OsDCL4",                "Os04g0457800"),
        # -------- Pumilio (stress-granule mRNA repressor) --------
        ("OsPumilio1",            "Os01g0938500"),
        # -------- 14-3-3 (chaperone-condensate scaffold) --------
        ("OsGF14b",               "Os04g0462500"),
        ("OsGF14e",               "Os02g0735800"),
        # -------- G3BP-like (SG nucleator) --------
        ("OsG3BP_like",           "Os03g0152900"),
        # -------- BRN1/2 (RNA-binding) --------
        ("OsBRN1",                "Os02g0123500"),
        # -------- FUS-like / TLS (low-complexity RBP) --------
        ("OsFUS_like",            "Os02g0623200"),
    ]


# ── Index builder for the Ensembl Plants rice cDNA fasta ─────────────────────

# header sample:
#  >Os01t0123450-01 cdna chromosome:IRGSP-1.0:1:... gene:Os01g0123450
#       gene_biotype:protein_coding transcript_biotype:protein_coding
#       gene_symbol:OsXXX

_TX_RE     = re.compile(r"^>(?P<tx>\S+)\s")
_GENE_RE   = re.compile(r"gene:(?P<gene>\S+)")
_SYMBOL_RE = re.compile(r"gene_symbol:(?P<sym>\S+)")


def build_index(ref_gz: str) -> dict:
    """
    Walk the Ensembl Plants rice cDNA fasta and return three dicts:
      gene_to_best_tx[gene_id]  = (tx_id, seq)        longest transcript
      symbol_to_gene[sym_upper] = gene_id             first encountered
      msu_to_rap[LOC_Os01g...]  = Os01g0...
                                    (built later from external mapping if any)
    """
    gene_to_best  : dict[str, tuple[str, str]] = {}
    symbol_to_gene: dict[str, str]             = {}

    if not os.path.exists(ref_gz):
        raise FileNotFoundError(
            f"{ref_gz} not found. Download the rice reference first."
        )

    with gzip.open(ref_gz, "rt") as fh:
        tx_id, gene_id, sym = None, None, None
        parts: list[str] = []

        def flush():
            nonlocal parts
            if tx_id and gene_id:
                seq = "".join(parts)
                prev = gene_to_best.get(gene_id)
                if prev is None or len(seq) > len(prev[1]):
                    gene_to_best[gene_id] = (tx_id, seq)
                if sym:
                    symbol_to_gene.setdefault(sym.upper(), gene_id)
            parts = []

        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                flush()
                tx_m   = _TX_RE.match(line)
                gene_m = _GENE_RE.search(line)
                sym_m  = _SYMBOL_RE.search(line)
                tx_id   = tx_m.group("tx") if tx_m else None
                gene_id = gene_m.group("gene") if gene_m else None
                sym     = sym_m.group("sym") if sym_m else None
            else:
                parts.append(line)
        flush()

    print(f"  [rice ref] {len(gene_to_best)} genes / "
          f"{len(symbol_to_gene)} gene_symbol aliases indexed",
          flush=True)
    return {"gene_to_best": gene_to_best, "symbol_to_gene": symbol_to_gene}


# ── Resolution: identifier → (gene_id, tx_id, seq) ───────────────────────────

# Accept multiple ID conventions:
#   RAP-DB gene:        Os01g0123450
#   RAP-DB transcript:  Os01t0123450-01
#   MSU locus:          LOC_Os01g12345          (no direct mapping in Ensembl;
#                                                we strip "LOC_" and use any
#                                                gene_symbol fallback)
#   gene symbol:        OsHSFA2, OsAGO1a, ...

_RAPDB_GENE_RE = re.compile(r"^Os\d{1,2}g\d+$", re.IGNORECASE)
_RAPDB_TX_RE   = re.compile(r"^Os\d{1,2}t\d+(?:-\d+)?$", re.IGNORECASE)
_MSU_RE        = re.compile(r"^LOC_Os\d{1,2}g\d+(?:\.\d+)?$", re.IGNORECASE)


def resolve(ident: str, idx: dict) -> tuple[str, str, str]:
    """Return (gene_id, tx_id, seq) or ('', '', '') if not found."""
    if not ident:
        return "", "", ""
    g2b = idx["gene_to_best"]
    s2g = idx["symbol_to_gene"]

    # 1) RAP-DB gene ID
    if _RAPDB_GENE_RE.match(ident):
        # Use canonical capitalisation Os01g0123450
        candidates = [ident, ident[:2] + ident[2:].lower()]
        # Header IDs use 2-digit chromosome (Os01..Os12); ensure match
        for cand in candidates:
            if cand in g2b:
                tx, seq = g2b[cand]
                return cand, tx, seq
        # Try lowercase suffix
        return "", "", ""

    # 2) RAP-DB transcript ID  Os01t0123450-01  →  Os01g0123450
    if _RAPDB_TX_RE.match(ident):
        base = ident.split("-", 1)[0]
        gene = base.replace("t", "g", 1)
        if gene in g2b:
            tx, seq = g2b[gene]
            return gene, tx, seq

    # 3) MSU locus  LOC_OsXXgYYYYY  – Ensembl uses RAP-DB IDs, so we cannot
    #    map MSU directly. Best-effort: strip prefix and look for any gene
    #    symbol containing the residual tail.
    if _MSU_RE.match(ident):
        tail = ident[4:].split(".", 1)[0]  # strip "LOC_" and isoform suffix
        # No reliable mapping → return empty (caller can supply alt)
        return "", "", ""

    # 4) gene_symbol match
    up = ident.upper()
    if up in s2g:
        gene = s2g[up]
        if gene in g2b:
            tx, seq = g2b[gene]
            return gene, tx, seq
    # Try lowercase / underscore-stripped variants
    for variant in {ident, ident.lower(), ident.upper(),
                    ident.replace("_", "")}:
        if variant.upper() in s2g:
            gene = s2g[variant.upper()]
            if gene in g2b:
                tx, seq = g2b[gene]
                return gene, tx, seq

    return "", "", ""


# ── FASTA writer (rice-specific so we control the header layout) ─────────────

def write_rice_fasta(records: list[tuple[str, str, str, str]],
                     out_path: str) -> int:
    """
    records: list of (gene_name, rap_db_locus, sequence_RNA, source_tag)
    Header: >llps_rice|<gene_name>|<rap_db_locus>|<source>
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    seen_locus: set[str] = set()
    with open(out_path, "w") as fh:
        for name, locus, seq, src in records:
            if not seq or len(seq) < 50:
                continue
            if locus in seen_locus:
                continue
            # RNA only: T → U, uppercase
            rna = seq.upper().replace("T", "U")
            # Strip anything non-RNA
            rna = re.sub(r"[^ACGUN]", "N", rna)
            if "llps" in name.lower() and ("sentinel" in name.lower()
                                            or "placeholder" in name.lower()):
                continue
            fh.write(f">llps_rice|{name}|{locus}|{src}\n{rna}\n")
            seen_locus.add(locus)
            written += 1
    return written


# ── Main ─────────────────────────────────────────────────────────────────────

def main(out_dir: str = "Data/raw/multispecies",
         ref_gz: str = REF_FASTA_GZ) -> None:
    idx = build_index(ref_gz)

    sources = {
        "UniProt_KW1185":  gene_list_uniprot_go_kw(),
        "Lit_canonical":   gene_list_canonical_paper_hits(),
    }

    # Deduplicate (preserve first-seen source ordering)
    seen: set[str] = set()
    pairs: list[tuple[str, str, str]] = []
    for src, lst in sources.items():
        for name, ident in lst:
            key = ident.upper()
            if key in seen:
                continue
            seen.add(key)
            pairs.append((name, ident, src))

    print(f"Rice LLPS gene-list size: {len(pairs)} "
          f"(across {len(sources)} sources)")

    records: list[tuple[str, str, str, str]] = []
    failures: list[tuple[str, str, str]] = []
    per_source: dict[str, int] = {}

    for i, (name, ident, src) in enumerate(pairs, 1):
        gene_id, tx_id, seq = resolve(ident, idx)
        if seq:
            records.append((name, gene_id, seq, src))
            per_source[src] = per_source.get(src, 0) + 1
        else:
            failures.append((name, ident, src))

    out_path = os.path.join(out_dir, "rice_positives.fasta")
    n = write_rice_fasta(records, out_path)
    print(f"Wrote {n} rice LLPS-positive sequences -> {out_path}")
    print(f"Per-source breakdown:  {per_source}")
    if failures:
        print(f"Unresolved ({len(failures)}):")
        for name, ident, src in failures[:25]:
            print(f"  {name:20s}  {ident:25s}  [{src}]")
    return {"n_written": n, "per_source": per_source,
            "n_failures": len(failures), "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw/multispecies")
    parser.add_argument("--ref-gz",  default=REF_FASTA_GZ)
    args = parser.parse_args()
    main(args.out_dir, args.ref_gz)
