"""
fetch_sequences.py
==================
Fetch RNA sequences for the RPS 2.0 Reviewed dataset from NCBI.

Strategy:
  1. Load unique (gene_symbol, organism) pairs from the RPS Reviewed CSV.
  2. Skip "Undefined" organism entries (mostly synthetic — handled separately).
  3. For each pair, query NCBI Nucleotide with:
       "{gene}[Gene Name] AND {taxon}[Organism] AND (NR_* OR NM_*)"
     Prefer NR_ (ncRNA) > NM_ (mRNA) > XR_ > XM_
  4. Fetch up to MAX_TRANSCRIPTS per gene, filter 20–12000 nt.
  5. Write positives to Data/raw/rps2_reviewed_rna.fasta
  6. Separately handle synthetic sequences from descriptions.

Run:
    python Functions/fetch_sequences.py --email your@email.com
"""

import argparse
import csv
import os
import re
import sys
import time
import json
from collections import defaultdict
from pathlib import Path

from Bio import Entrez, SeqIO

# ─────────────────────────── config ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "Data" / "raw"
RPS_CSV      = DATA_RAW / "RPS_reviewed" / "RPS_reviewed_detail_table_0903.csv"
OUT_FASTA    = DATA_RAW / "rps2_reviewed_rna.fasta"
SYNTH_FASTA  = DATA_RAW / "synthetic_rna.fasta"
PROGRESS_JSON = DATA_RAW / "fetch_progress.json"

MAX_TRANSCRIPTS  = 3      # max transcripts per gene/organism pair
SEQ_MIN, SEQ_MAX = 20, 12_000
BATCH_SIZE       = 200    # IDs per efetch call
SLEEP_SHORT      = 0.35   # seconds between requests (NCBI limit: 3/sec)
SLEEP_LONG       = 2.0    # after errors

# ncRNA accession types in preference order
PREFERRED_PREFIXES = ["NR_", "NM_", "XR_", "XM_", "ENST"]

# NCBI organism name → taxon query string
ORGANISM_TAXON = {
    "Homo sapiens":               "Homo sapiens[Organism]",
    "Mus musculus":               "Mus musculus[Organism]",
    "Saccharomyces cerevisiae":   "Saccharomyces cerevisiae[Organism]",
    "Caenorhabditis elegans":     "Caenorhabditis elegans[Organism]",
    "SARS-CoV-2":                 "Severe acute respiratory syndrome coronavirus 2[Organism]",
    "Drosophila melanogaster":    "Drosophila melanogaster[Organism]",
    "Xenopus laevis":             "Xenopus laevis[Organism]",
    "Danio rerio":                "Danio rerio[Organism]",
    "Escherichia coli":           "Escherichia coli[Organism]",
    "Rotavirus A":                "Rotavirus A[Organism]",
    "Human papillomavirus 16":    "Human papillomavirus type 16[Organism]",
    "Arabidopsis thaliana":       "Arabidopsis thaliana[Organism]",
    "Mammalian orthoreovirus":    "Mammalian orthoreovirus[Organism]",
}

# ─────────────────────────── helpers ──────────────────────────

def normalise_seq(seq: str) -> str:
    """Uppercase, T→U, keep only AUGCN."""
    seq = seq.upper().replace("T", "U")
    seq = re.sub(r"[^AUGCN]", "", seq)
    return seq

def acc_rank(accession: str) -> int:
    """Lower = more preferred accession type."""
    for i, pref in enumerate(PREFERRED_PREFIXES):
        if accession.startswith(pref):
            return i
    return len(PREFERRED_PREFIXES)

def search_gene(gene: str, taxon: str, max_results: int = 20) -> list[str]:
    """Return list of accession strings, sorted by preference."""
    query = (
        f'("{gene}"[Gene Name] OR "{gene}"[Symbol]) '
        f'AND {taxon} '
        f'AND (biomol_ncRNA[PROP] OR biomol_mRNA[PROP])'
    )
    try:
        handle = Entrez.esearch(db="nucleotide", term=query,
                                retmax=max_results, idtype="acc")
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        time.sleep(SLEEP_SHORT)
        if not ids:
            # Fallback: broader search
            query2 = f'"{gene}"[Gene Name] AND {taxon}'
            handle2 = Entrez.esearch(db="nucleotide", term=query2,
                                     retmax=max_results, idtype="acc")
            record2 = Entrez.read(handle2)
            handle2.close()
            ids = record2.get("IdList", [])
            time.sleep(SLEEP_SHORT)
        # Sort by preference
        ids_sorted = sorted(ids, key=acc_rank)
        return ids_sorted
    except Exception as e:
        print(f"    [search error] {gene}/{taxon}: {e}")
        time.sleep(SLEEP_LONG)
        return []

def fetch_records(acc_ids: list[str]) -> list[SeqIO.SeqRecord]:
    """Batch-fetch SeqRecords for a list of accession IDs."""
    if not acc_ids:
        return []
    records = []
    try:
        handle = Entrez.efetch(db="nucleotide", id=",".join(acc_ids),
                               rettype="fasta", retmode="text")
        for rec in SeqIO.parse(handle, "fasta"):
            records.append(rec)
        handle.close()
        time.sleep(SLEEP_SHORT)
    except Exception as e:
        print(f"    [fetch error] {acc_ids[:3]}...: {e}")
        time.sleep(SLEEP_LONG)
    return records

def pick_best_transcripts(records: list[SeqIO.SeqRecord],
                          max_n: int = MAX_TRANSCRIPTS) -> list[SeqIO.SeqRecord]:
    """Keep ≤ max_n records, preferring preferred accession types and longer seqs."""
    valid = [r for r in records if SEQ_MIN <= len(r.seq) <= SEQ_MAX]
    if not valid:
        return []
    # Sort: preferred type first, then by length descending
    valid.sort(key=lambda r: (acc_rank(r.id), -len(r.seq)))
    return valid[:max_n]

# ─────────────────────── synthetic sequences ─────────────────

# Well-known synthetic sequences that can be extracted from descriptions
KNOWN_SYNTHETICS = {
    "polyU":   "U" * 200,
    "polyA":   "A" * 200,
    "polyC":   "C" * 200,
    "polyG":   "G" * 200,
    "polyAU":  ("AU") * 100,
    "polyUG":  ("UG") * 100,
    "polyGA":  ("GA") * 100,
    "polyCA":  ("CA") * 100,
}

# Regex to extract explicit sequences from descriptions
SEQ_REGEX = re.compile(r'\b([AUGCTN]{12,})\b')
CAG_REPEAT = re.compile(r'(\d+)[xX]CAG', re.IGNORECASE)
GU_REPEAT  = re.compile(r'A\(GU\)(\d+)', re.IGNORECASE)
U_REPEAT   = re.compile(r'(\d+)[xX]\s*U', re.IGNORECASE)

def parse_synthetic_seq(gene: str, description: str) -> str | None:
    """Try to extract or reconstruct a synthetic sequence from its description."""
    gene_low = gene.lower()

    # Check known homopolymers
    if gene_low in KNOWN_SYNTHETICS:
        return KNOWN_SYNTHETICS[gene_low]
    for key, seq in KNOWN_SYNTHETICS.items():
        if gene_low.startswith(key):
            return seq

    # CAG repeat
    m = CAG_REPEAT.search(description)
    if m:
        n = min(int(m.group(1)), 200)
        return "CAG" * n

    # GU repeat: A(GU)N
    m = GU_REPEAT.search(description)
    if m:
        n = min(int(m.group(1)), 100)
        return "A" + "GU" * n

    # U repeat
    m = U_REPEAT.search(description)
    if m:
        n = min(int(m.group(1)), 300)
        return "U" * n

    # Inline sequence
    m = SEQ_REGEX.search(description)
    if m:
        cand = m.group(1).upper().replace("T", "U")
        if len(cand) >= 12:
            return cand

    # Syn_1 type: check if description has an explicit short seq
    seq_match = re.search(r"sequence of ([AUGCTN]{6,})", description, re.IGNORECASE)
    if seq_match:
        return seq_match.group(1).upper().replace("T","U")

    return None

# ─────────────────────────── main ─────────────────────────────

def load_rps_pairs(csv_path: Path):
    """Return list of (gene_symbol, organism, description) unique pairs."""
    seen = set()
    pairs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene = row["gene_symbol"].strip()
            org  = row["organism"].strip()
            desc = row.get("rps_description", "").strip()
            key  = (gene, org)
            if key not in seen:
                seen.add(key)
                pairs.append((gene, org, desc))
    return pairs

def load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def save_progress(path: Path, prog: dict):
    with open(path, "w") as f:
        json.dump(prog, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Fetch RPS2 RNA sequences from NCBI")
    parser.add_argument("--email", required=True,
                        help="Email for NCBI Entrez (required by NCBI)")
    parser.add_argument("--max-per-gene", type=int, default=MAX_TRANSCRIPTS)
    parser.add_argument("--resume", action="store_true",
                        help="Skip genes already in progress JSON")
    args = parser.parse_args()

    Entrez.email = args.email
    Entrez.tool  = "RNAPhaseek_data_collector"

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    pairs = load_rps_pairs(RPS_CSV)
    print(f"Loaded {len(pairs)} unique (gene, organism) pairs")

    # Split into known-organism and synthetic
    real_pairs  = [(g, o, d) for g, o, d in pairs if o != "Undefined" and o in ORGANISM_TAXON]
    synth_pairs = [(g, o, d) for g, o, d in pairs if o == "Undefined"]
    unknown_org = [(g, o, d) for g, o, d in pairs if o != "Undefined" and o not in ORGANISM_TAXON]

    print(f"  Real organism pairs  : {len(real_pairs)}")
    print(f"  Synthetic (Undefined): {len(synth_pairs)}")
    print(f"  Unknown organism     : {len(unknown_org)}")
    if unknown_org:
        print(f"    → Skipped: {[o for _,o,_ in unknown_org[:5]]}")

    progress = load_progress(PROGRESS_JSON) if args.resume else {}

    # ── Fetch real sequences ──────────────────────────────────
    written = 0
    skipped = 0
    failed  = 0
    total   = len(real_pairs)

    mode = "a" if args.resume else "w"
    out_f = open(OUT_FASTA, mode, encoding="utf-8")

    for idx, (gene, org, desc) in enumerate(real_pairs):
        key = f"{gene}|{org}"
        if args.resume and key in progress:
            skipped += 1
            continue

        taxon = ORGANISM_TAXON[org]
        print(f"[{idx+1}/{total}] {gene} / {org} ...", end=" ", flush=True)

        ids = search_gene(gene, taxon, max_results=30)
        if not ids:
            print("no IDs")
            progress[key] = []
            failed += 1
            if (idx + 1) % 20 == 0:
                save_progress(PROGRESS_JSON, progress)
            continue

        # Fetch candidates (take top 15 by acc_rank)
        top_ids = sorted(ids[:15], key=acc_rank)[:15]
        records = fetch_records(top_ids)
        best    = pick_best_transcripts(records, max_n=args.max_per_gene)

        if not best:
            print(f"no valid seqs (fetched {len(records)}, all out of range)")
            progress[key] = []
            failed += 1
        else:
            acc_list = []
            for rec in best:
                seq_norm = normalise_seq(str(rec.seq))
                if len(seq_norm) < SEQ_MIN:
                    continue
                header = (f">rps2|{gene}|{org.replace(' ','_')}|"
                          f"{rec.id}|len={len(seq_norm)}")
                out_f.write(f"{header}\n{seq_norm}\n")
                acc_list.append(rec.id)
                written += 1
            progress[key] = acc_list
            print(f"✓ {len(best)} seqs ({[r.id for r in best]})")

        if (idx + 1) % 20 == 0:
            save_progress(PROGRESS_JSON, progress)
            print(f"  --- checkpoint: {written} seqs written so far ---")

    out_f.close()
    save_progress(PROGRESS_JSON, progress)
    print(f"\nReal sequences: {written} written, {failed} genes failed/not found")

    # ── Synthetic sequences ──────────────────────────────────
    syn_written = 0
    with open(SYNTH_FASTA, "w", encoding="utf-8") as sf:
        for gene, org, desc in synth_pairs:
            seq = parse_synthetic_seq(gene, desc)
            if seq:
                seq = normalise_seq(seq)
                if SEQ_MIN <= len(seq) <= SEQ_MAX:
                    header = f">synthetic|{gene}|len={len(seq)}"
                    sf.write(f"{header}\n{seq}\n")
                    syn_written += 1

    print(f"Synthetic sequences: {syn_written} written to {SYNTH_FASTA}")
    print(f"\nDone. Main output: {OUT_FASTA}")

if __name__ == "__main__":
    main()
