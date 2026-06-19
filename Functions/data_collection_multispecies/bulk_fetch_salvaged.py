"""
Bulk-fetch cDNA for gene-ID lists salvaged from the workflow extractions.

Inputs:
  Data/raw/multispecies/salvaged/{species}_gene_list.txt

Outputs (append to existing per-species positives):
  Data/raw/multispecies/{species}_positives.fasta
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path
from .species_cdna_fetcher import (
    fetch_yeast_orf_from_sgd, fetch_cdna_by_symbol, fetch_cdna_by_ensembl_gene,
    write_positives_fasta,
)


def read_id_list(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    # Dedup, preserve order
    seen = set(); out = []
    for x in ids:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def existing_positive_ids(fasta: str) -> set[str]:
    s = set()
    if not os.path.exists(fasta):
        return s
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                toks = line[1:].strip().split("|")
                if len(toks) >= 2:
                    s.add(toks[1].upper().split(".")[0])
    return s


def run_yeast(out_dir: Path, salvaged: list[str], source_tag: str = "WorkflowSalvage"):
    """Fast in-memory lookup against SGD ORF index."""
    out_fasta = out_dir / "yeast_positives.fasta"
    existing = existing_positive_ids(str(out_fasta))
    print(f"\n=== Yeast bulk fetch (SGD direct) ===")
    print(f"  to-resolve: {len(salvaged)}  |  existing in FASTA: {len(existing)}")

    records = []
    n_found, n_skipped, n_already = 0, 0, 0
    for i, gid in enumerate(salvaged, 1):
        if gid.upper() in existing:
            n_already += 1; continue
        seq, sys_id = fetch_yeast_orf_from_sgd(gid)
        if seq and sys_id:
            records.append((gid, sys_id, seq, source_tag))
            n_found += 1
        else:
            n_skipped += 1
        if i % 2000 == 0:
            print(f"  [{i}/{len(salvaged)}] found={n_found} skipped={n_skipped}", flush=True)
    print(f"  resolved {n_found}/{len(salvaged)} (skipped {n_skipped}, already-have {n_already})")

    # Append to existing FASTA
    written = 0
    with open(out_fasta, "a") as fh:
        for gid, tx, seq, src in records:
            if len(seq) < 50: continue
            fh.write(f">llps_yeast|{gid}|{tx}|{src}\n{seq}\n")
            written += 1
    print(f"  appended {written} new -> {out_fasta}")
    return written


def run_celegans(out_dir: Path, salvaged: list[str], rate: float = 0.33,
                  source_tag: str = "WorkflowSalvage", cap: int = None):
    """Ensembl REST lookup for WBGene IDs. Slow (rate-limited)."""
    out_fasta = out_dir / "celegans_positives.fasta"
    existing = existing_positive_ids(str(out_fasta))
    pool = [x for x in salvaged if x.upper() not in existing]
    if cap:
        pool = pool[:cap]
    print(f"\n=== C. elegans bulk fetch (Ensembl REST) ===")
    print(f"  to-resolve: {len(pool)}  (capped from {len(salvaged)})")
    print(f"  ETA: ~{len(pool)*rate/60:.0f} min at ~3 req/sec")

    records, n_found = [], 0
    for i, gid in enumerate(pool, 1):
        # WBGene IDs vs other formats
        if gid.startswith("WBGene"):
            seq, tx = fetch_cdna_by_ensembl_gene(gid, "caenorhabditis_elegans")
        else:
            seq, tx = fetch_cdna_by_symbol(gid, "caenorhabditis_elegans")
        if seq and len(seq) >= 50:
            records.append((gid, tx, seq, source_tag))
            n_found += 1
        if i % 200 == 0:
            print(f"  [{i}/{len(pool)}] found={n_found}", flush=True)

    with open(out_fasta, "a") as fh:
        for gid, tx, seq, src in records:
            fh.write(f">llps_celegans|{gid}|{tx}|{src}\n{seq}\n")
    print(f"  appended {len(records)} new -> {out_fasta}")
    return len(records)


def run_drosophila(out_dir: Path, salvaged: list[str], source_tag: str = "WorkflowSalvage"):
    """Ensembl REST lookup; mixed CG IDs / FBgn IDs / symbols."""
    out_fasta = out_dir / "drosophila_positives.fasta"
    existing = existing_positive_ids(str(out_fasta))
    pool = [x for x in salvaged if x.upper() not in existing]
    print(f"\n=== Drosophila bulk fetch (Ensembl REST) ===")
    print(f"  to-resolve: {len(pool)}")

    records, n_found = [], 0
    for i, gid in enumerate(pool, 1):
        if gid.startswith("FBgn"):
            seq, tx = fetch_cdna_by_ensembl_gene(gid, "drosophila_melanogaster")
        elif gid.startswith("CG"):
            # CG IDs are gene-level; try as symbol
            seq, tx = fetch_cdna_by_symbol(gid, "drosophila_melanogaster")
        else:
            seq, tx = fetch_cdna_by_symbol(gid, "drosophila_melanogaster")
        if seq and len(seq) >= 50:
            records.append((gid, tx, seq, source_tag))
            n_found += 1
        if i % 50 == 0:
            print(f"  [{i}/{len(pool)}] found={n_found}", flush=True)

    with open(out_fasta, "a") as fh:
        for gid, tx, seq, src in records:
            fh.write(f">llps_drosophila|{gid}|{tx}|{src}\n{seq}\n")
    print(f"  appended {len(records)} new -> {out_fasta}")
    return len(records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--species", choices=["yeast","celegans","drosophila","all"], default="all")
    p.add_argument("--celegans-cap", type=int, default=4500,
                   help="C. elegans cap so we don't fetch 45k Ensembl IDs (use only valid WBGenes)")
    args = p.parse_args()

    out_dir = Path("Data/raw/multispecies")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    if args.species in ("yeast","all"):
        ids = read_id_list("Data/raw/multispecies/salvaged/yeast_gene_list.txt")
        summary["yeast"] = run_yeast(out_dir, ids)

    if args.species in ("drosophila","all"):
        ids = read_id_list("Data/raw/multispecies/salvaged/drosophila_gene_list.txt")
        summary["drosophila"] = run_drosophila(out_dir, ids)

    if args.species in ("celegans","all"):
        ids = read_id_list("Data/raw/multispecies/salvaged/celegans_gene_list.txt")
        # Filter to genuine WBGene IDs to avoid fetching the entire transcriptome
        # (Knutson's 45k extraction probably included the full WBGene index, not just LLPS+).
        wbgenes = [x for x in ids if re.match(r"^WBGene\d{8}$", x)]
        others  = [x for x in ids if not re.match(r"^WBGene\d{8}$", x)]
        print(f"\nC. elegans filtering: {len(wbgenes)} WBGenes + {len(others)} other identifiers")
        # Cap WBGenes to args.celegans_cap so we don't fetch 45k
        wbgenes = wbgenes[:args.celegans_cap]
        summary["celegans"] = run_celegans(out_dir, wbgenes)

    print("\n=== Summary ===")
    for sp, n in summary.items():
        print(f"  {sp:<12} +{n}")


if __name__ == "__main__":
    main()
