"""
Arabidopsis thaliana LLPS-positive RNA collection.

Strategy:
  1. Query UniProt for proteins annotated with LLPS-related GO terms restricted
     to Arabidopsis (taxonomy id 3702):
       - GO:0035770  cytoplasmic ribonucleoprotein granule
       - GO:0010494  stress granule
       - GO:0036464  cytoplasmic ribonucleoprotein granule
     plus a curated set of canonical literature hits when UniProt coverage is
     incomplete or the locus comes from a non-Col-0 accession.

  2. Resolve TAIR locus IDs (AT1G12345 form) against a local copy of the TAIR10
     cDNA FASTA (Ensembl Plants release-58). The principal `.1` splice isoform is
     preferred; if absent, the longest available isoform is picked.

  3. Emit RNA (T -> U) sequences with the standard multispecies header.

Output:
  Data/raw/multispecies/arabidopsis_positives.fasta

Header format:
  >llps_arabidopsis|<gene_name>|<TAIR_locus.transcript>|<source>
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import OrderedDict


TAIR_LOCUS_RE = re.compile(r"\bAT[1-5MC]G\d{5}\b", re.IGNORECASE)
DEFAULT_CDNA = "Data/raw/multispecies/refs/arabidopsis_thaliana_cdna.fa.gz"
DEFAULT_OUT  = "Data/raw/multispecies/arabidopsis_positives.fasta"

# ── Curated literature hits (canonical Arabidopsis LLPS factors) ──────────────
# Locus  -> (gene_name, source tag)
LIT_CANONICAL: list[tuple[str, str, str]] = [
    ("AT4G16280", "FCA",      "Lit_canonical"),   # flowering time, prion-like
    ("AT3G10390", "FLD",      "Lit_canonical"),
    ("AT1G04510", "AtPRP19",  "Lit_canonical"),
    ("AT2G27100", "SERRATE",  "Lit_canonical"),
    ("AT1G48410", "AGO1",     "Lit_canonical"),
    ("AT1G08370", "DCP1",     "Lit_canonical"),
    ("AT5G13570", "DCP2",     "Lit_canonical"),
    ("AT1G26110", "DCP5",     "Lit_canonical"),
    ("AT3G13300", "VCS",      "Lit_canonical"),   # VARICOSE
    ("AT1G54080", "AtUBP1B",  "Lit_canonical"),
    ("AT5G07350", "TSN1",     "Lit_canonical"),
    ("AT5G61780", "TSN2",     "Lit_canonical"),
    ("AT2G21660", "GRP7",     "Lit_canonical"),
    ("AT4G31120", "AtPRMT5",  "Lit_canonical"),
    ("AT4G24770", "AtALBA4",  "Lit_canonical"),
    # FLC autonomous-pathway condensate target
    ("AT5G10140", "FLC",      "Lit_canonical"),
    # RBP47 family (Sorenson & Bailey-Serres SG paper)
    ("AT1G47490", "RBP47A",   "Lit_canonical"),
    ("AT3G19130", "RBP47B",   "Lit_canonical"),
    ("AT1G49600", "RBP47C",   "Lit_canonical"),
    # Cap-binding complex
    ("AT5G44200", "AtCBP20",  "Lit_canonical"),
    ("AT2G13540", "AtCBP80",  "Lit_canonical"),
    # SG marker UBP1 family
    ("AT3G14100", "AtUBP1A",  "Lit_canonical"),
    ("AT3G14080", "AtUBP1C",  "Lit_canonical"),
    # Additional canonical SG/P-body markers (Sorenson Plant Cell 2019)
    ("AT3G25500", "AtFMN",    "Lit_canonical"),   # FRIGIDA-ESSENTIAL 1 (formamidopyrimidine homolog)
    ("AT4G25500", "AtRS40",   "Lit_canonical"),   # arginine/serine-rich splicing
    ("AT4G14300", "AtPABN1",  "Lit_canonical"),
    ("AT5G65260", "RBGB1",    "Lit_canonical"),
    ("AT5G16780", "ATERF",    "Lit_canonical"),
    ("AT4G39260", "AtGRP8",   "Lit_canonical"),
    ("AT1G18450", "AtABF1",   "Lit_canonical"),
    ("AT4G37280", "MRG1",     "Lit_canonical"),
    ("AT1G65610", "AtKHZ1",   "Lit_canonical"),
    ("AT2G16800", "AtKHZ2",   "Lit_canonical"),
]

# ── UniProt query ────────────────────────────────────────────────────────────

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_QUERY = (
    "organism_id:3702 AND (go:0035770 OR go:0010494 OR go:0036464)"
)
UNIPROT_FIELDS = "accession,gene_primary,gene_names,xref_tair,xref_ensemblplants"


def _http_get(url: str, headers: dict | None = None, timeout: int = 60) -> tuple[bytes, "object"]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # Return raw HTTPMessage so we can do case-insensitive Link lookup
        return r.read(), r.info()


def _parse_next_link(link_header: str) -> str | None:
    """Parse Link: <url>; rel=\"next\" header (HTTP/1.1 RFC5988)."""
    if not link_header:
        return None
    # The Link header may contain commas inside URL params; split only on `,\s*<`
    parts = re.split(r",\s*(?=<)", link_header)
    for p in parts:
        m = re.match(r'<([^>]+)>\s*;\s*rel="?next"?', p.strip())
        if m:
            return m.group(1)
    return None


def fetch_uniprot_tsv() -> list[dict]:
    """Return list of {accession, gene_primary, gene_names, tair, ensembl_plants}."""
    base_params = {
        "query":  UNIPROT_QUERY,
        "fields": UNIPROT_FIELDS,
        "format": "tsv",
        "size":   "500",
    }
    url = f"{UNIPROT_URL}?{urllib.parse.urlencode(base_params)}"
    rows: list[dict] = []
    while url:
        body, hdrs = _http_get(url)
        text = body.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            break
        # header row only on first call
        start = 0
        if lines[0].startswith("Entry"):
            start = 1
        for line in lines[start:]:
            if not line.strip():
                continue
            cells = line.split("\t")
            # pad if short
            cells = (cells + [""] * 5)[:5]
            rows.append({
                "accession":     cells[0],
                "gene_primary":  cells[1],
                "gene_names":    cells[2],
                "tair":          cells[3],
                "ensembl_plants": cells[4],
            })
        # HTTPMessage exposes case-insensitive .get
        link = hdrs.get("Link") or hdrs.get("link") or ""
        url = _parse_next_link(link)
        if url:
            time.sleep(0.2)
    return rows


def extract_tair_locus(row: dict) -> str | None:
    """Pull a TAIR locus (AT1G12345 form) from any field."""
    for field in ("tair", "gene_names", "gene_primary"):
        text = row.get(field) or ""
        m = TAIR_LOCUS_RE.search(text)
        if m:
            return m.group(0).upper()
    return None


# ── cDNA index (TAIR10) ──────────────────────────────────────────────────────

def build_cdna_index(fasta_gz: str) -> dict[str, list[tuple[str, str]]]:
    """
    Map TAIR locus -> list of (transcript_id, sequence) tuples, sorted by
    isoform suffix (so .1 sorts first).
    Sequences are RNA (T -> U) upper-case.
    """
    locus_to_tx: dict[str, list[tuple[str, str]]] = {}
    current_tx, current_locus, parts = None, None, []

    def flush():
        nonlocal current_tx, current_locus, parts
        if current_tx and current_locus and parts:
            seq = "".join(parts).upper().replace("T", "U")
            locus_to_tx.setdefault(current_locus, []).append((current_tx, seq))
        current_tx, current_locus, parts = None, None, []

    with gzip.open(fasta_gz, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                # Ensembl Plants header:
                # >AT1G76820.1 cdna chromosome:... gene:AT1G76820 ...
                head = line[1:]
                first = head.split(maxsplit=1)[0]   # e.g. AT1G76820.1
                current_tx = first
                # Locus = stripped of isoform suffix
                base = first.split(".", 1)[0]
                current_locus = base.upper()
                parts = []
            else:
                parts.append(line)
        flush()

    # sort isoforms by suffix number (.1 first)
    def iso_key(tx: str) -> tuple[int, str]:
        m = re.search(r"\.(\d+)$", tx)
        return (int(m.group(1)) if m else 99, tx)

    for locus, items in locus_to_tx.items():
        items.sort(key=lambda kv: iso_key(kv[0]))
    return locus_to_tx


def pick_transcript(locus: str, idx: dict[str, list[tuple[str, str]]]
                    ) -> tuple[str, str]:
    """Return (transcript_id, RNA sequence) for a TAIR locus (.1 if present)."""
    items = idx.get(locus.upper())
    if not items:
        return "", ""
    # Prefer the longest among .1-suffixed; otherwise the longest overall
    suf1 = [(tx, seq) for tx, seq in items if tx.endswith(".1")]
    pool = suf1 if suf1 else items
    tx, seq = max(pool, key=lambda kv: len(kv[1]))
    return tx, seq


# ── Quality gate ─────────────────────────────────────────────────────────────

VALID_NT = set("ACGUN")


def quality_ok(seq: str, gene_name: str, locus: str) -> bool:
    if not seq or len(seq) < 50:
        return False
    if gene_name.upper() == locus.upper():
        return False  # sentinel guard
    # Reject protein 1-letter codes - allow only A/C/G/U/N
    bad = set(seq.upper()) - VALID_NT
    if bad:
        return False
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main(out_path: str = DEFAULT_OUT,
         cdna_gz: str = DEFAULT_CDNA,
         skip_uniprot: bool = False) -> None:

    print(f"[1/4] Building TAIR10 cDNA index from {cdna_gz} ...", flush=True)
    idx = build_cdna_index(cdna_gz)
    print(f"      {len(idx)} unique loci indexed", flush=True)

    # locus -> (gene_name, source, original_uniprot_acc | "")
    # First-source-wins ordering so UniProt rows aren't displaced by lit list.
    locus_to_entry: "OrderedDict[str, tuple[str, str]]" = OrderedDict()

    if not skip_uniprot:
        print("[2/4] Querying UniProt ...", flush=True)
        try:
            rows = fetch_uniprot_tsv()
        except Exception as e:
            print(f"      UniProt query failed: {e}", file=sys.stderr)
            rows = []
        print(f"      retrieved {len(rows)} UniProt rows", flush=True)

        for row in rows:
            locus = extract_tair_locus(row)
            if not locus:
                continue
            gene = (row.get("gene_primary") or "").strip()
            if not gene:
                # fall back to first token in gene_names that isn't a TAIR locus
                # (and not a lowercased "atNgNNNNN" duplicate)
                for tok in (row.get("gene_names") or "").split():
                    if not TAIR_LOCUS_RE.match(tok):
                        gene = tok
                        break
            if not gene:
                # Uncharacterized locus -- use UniProt accession so we have a
                # human-readable, non-locus identifier (avoids sentinel guard).
                acc = (row.get("accession") or "").strip()
                gene = f"UNIPROT_{acc}" if acc else ""
            if not gene:
                continue  # nothing usable
            if locus not in locus_to_entry:
                locus_to_entry[locus] = (gene, "UniProt_GO")
    else:
        print("[2/4] Skipping UniProt (per --skip-uniprot)", flush=True)

    print("[3/4] Adding canonical literature hits ...", flush=True)
    for locus, gene, src in LIT_CANONICAL:
        locus = locus.upper()
        if locus not in locus_to_entry:
            locus_to_entry[locus] = (gene, src)
    print(f"      total unique loci queued: {len(locus_to_entry)}", flush=True)

    print("[4/4] Resolving loci against cDNA index ...", flush=True)
    records: list[tuple[str, str, str, str]] = []
    per_source: dict[str, int] = {}
    n_unresolved = 0
    failures: list[str] = []

    for locus, (gene, src) in locus_to_entry.items():
        tx, seq = pick_transcript(locus, idx)
        if not seq:
            n_unresolved += 1
            failures.append(f"{locus} ({gene})")
            continue
        if not quality_ok(seq, gene, locus):
            failures.append(f"{locus} ({gene}) -- quality gate")
            continue
        # Output identifier:  use the transcript id (e.g. AT1G12345.1) which
        # carries the TAIR locus and the canonical isoform suffix.  This keeps
        # the FASTA header at 4 pipe-separated fields, matching the other
        # multispecies files (>llps_<species>|gene|transcript|source).
        records.append((gene, tx, seq, src))
        per_source[src] = per_source.get(src, 0) + 1

    # Write FASTA
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w") as fh:
        for gene, tx, seq, src in records:
            fh.write(f">llps_arabidopsis|{gene}|{tx}|{src}\n{seq}\n")
            written += 1

    # Report
    print()
    print("=" * 60)
    print("Arabidopsis thaliana LLPS-positive curation summary")
    print("=" * 60)
    print(f"  unique loci queued        : {len(locus_to_entry)}")
    print(f"  successfully resolved     : {written}")
    print(f"  unresolved (no cDNA hit)  : {n_unresolved}")
    print(f"  per-source breakdown      :")
    for src, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"      {src:20s}  {n}")
    print(f"  output -> {out_path}")
    if failures:
        print(f"  failures ({len(failures)}): first 10 -> {failures[:10]}")

    # First 3 headers (preview)
    print()
    print("First 3 headers:")
    with open(out_path) as fh:
        shown = 0
        for line in fh:
            if line.startswith(">"):
                print(f"  {line.rstrip()}")
                shown += 1
                if shown == 3:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--cdna", default=DEFAULT_CDNA)
    parser.add_argument("--skip-uniprot", action="store_true")
    args = parser.parse_args()
    main(args.out, args.cdna, args.skip_uniprot)
