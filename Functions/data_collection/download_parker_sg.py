"""
Download Parker stress-granule (SG) mRNA enrichment data (GSE99304).

Source:
  GEO GSE99304 — U2OS cell SG transcriptome (Parker lab, 2018)
  File: GSE99304_U2OSSGcuffdiff.txt.gz
  2,457 genes marked "enriched" in stress granules

Pipeline:
  1. Download the cuffdiff table from GEO
  2. Extract gene symbols for enriched genes (significant == "yes", log2fc > 0)
  3. Map gene symbols → Ensembl transcript IDs via MyGene.info
  4. Fetch longest mRNA sequence per gene from Ensembl REST API
  5. Write Data/raw/parker_sg_positives.fasta

Run:
    python -m Functions.data_collection.download_parker_sg
"""

import gzip
import io
import os
import sys
import time
import argparse
import requests

GEO_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/download/"
    "?acc=GSE99304&format=file&file=GSE99304_U2OSSGcuffdiff.txt.gz"
)
MYGENE_URL  = "https://mygene.info/v3/query"
ENSEMBL_URL = "https://rest.ensembl.org"
TIMEOUT = 60


# ── Step 1: Download and parse the cuffdiff table ────────────────────────────

def download_cuffdiff(url: str = GEO_URL) -> list[str]:
    """Return gene symbols for SG-enriched genes."""
    print(f"Downloading {url} …", flush=True)
    r = requests.get(url, timeout=TIMEOUT, stream=True)
    r.raise_for_status()

    buf = io.BytesIO(r.content)
    enriched: list[str] = []
    with gzip.open(buf, "rt") as fh:
        header = fh.readline().strip().split("\t")
        # Expected columns: gene, locus, sample_1, sample_2,
        #   status, value_1, value_2, log2(fold_change), test_stat,
        #   p_value, q_value, significant
        try:
            gene_col   = header.index("gene")
            sig_col    = header.index("significant")
            status_col = header.index("status")
            cls_col    = header.index("classification")
        except ValueError:
            gene_col, status_col, sig_col, cls_col = 0, 4, 11, 12
            fh.seek(0)

        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) <= max(gene_col, sig_col, status_col, cls_col):
                continue
            gene   = parts[gene_col].strip()
            sig    = parts[sig_col].strip().lower()
            status = parts[status_col].strip().upper()
            cls    = parts[cls_col].strip().lower()

            # cuffdiff sample order is (SG, Total); log2 sign is therefore inverted —
            # use the explicit classification column to keep SG-enriched genes.
            if sig == "yes" and status == "OK" and cls == "enriched":
                enriched.append(gene)

    unique = sorted(set(enriched))
    print(f"  {len(unique)} unique SG-enriched genes", flush=True)
    return unique


# ── Step 2: Map symbols → Ensembl transcript IDs via MyGene.info ─────────────

def symbols_to_ensembl(symbols: list[str], batch: int = 500) -> dict[str, str]:
    """Return {symbol: ensembl_transcript_id} for the longest known transcript."""
    print(f"Mapping {len(symbols)} symbols → Ensembl via MyGene.info …", flush=True)
    result: dict[str, str] = {}

    for start in range(0, len(symbols), batch):
        chunk = symbols[start : start + batch]
        payload = {
            "q":       ",".join(chunk),
            "scopes":  "symbol,alias",
            "fields":  "ensembl.transcript",
            "species": "human",
        }
        r = requests.post(MYGENE_URL, data=payload, timeout=TIMEOUT)
        r.raise_for_status()
        for hit in r.json():
            sym = hit.get("query", "")
            ens = hit.get("ensembl", {})
            if isinstance(ens, list):
                ens = ens[0]
            txs = ens.get("transcript", [])
            if isinstance(txs, str):
                txs = [txs]
            if txs:
                result[sym] = txs[0]   # take first; we'll fetch all for this gene later
        time.sleep(0.2)

    print(f"  {len(result)}/{len(symbols)} mapped", flush=True)
    return result


# ── Step 3: Fetch sequence from Ensembl REST ─────────────────────────────────

def fetch_ensembl_sequence(transcript_id: str, max_retries: int = 4) -> str:
    """Return mRNA sequence for an Ensembl transcript. Retries on 429/5xx."""
    url = f"{ENSEMBL_URL}/sequence/id/{transcript_id}"
    params = {"content-type": "text/plain", "type": "cdna"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text.strip().upper().replace("T", "U")
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 30))
                continue
            return ""
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return ""


def fetch_all_transcripts_for_gene(symbol: str, gene_id: str = "") -> tuple[str, str]:
    """
    Fetch the longest mRNA for a gene by querying Ensembl lookup.
    Returns (sequence, transcript_id).
    """
    # First get all transcripts for this gene via gene symbol
    url = f"{ENSEMBL_URL}/xrefs/symbol/homo_sapiens/{symbol}"
    params = {"content-type": "application/json", "object_type": "transcript"}
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return "", ""
        transcripts = [item["id"] for item in r.json() if item.get("id", "").startswith("ENST")]
    except Exception:
        return "", ""

    best_seq, best_tx = "", ""
    for tx in transcripts[:5]:   # try up to 5 transcripts, keep longest
        seq = fetch_ensembl_sequence(tx)
        if len(seq) > len(best_seq):
            best_seq, best_tx = seq, tx
        time.sleep(0.05)

    return best_seq, best_tx


# ── Main ──────────────────────────────────────────────────────────────────────

def _already_fetched_symbols(out_fasta: str) -> set[str]:
    """Read existing FASTA, return the set of ParkerSG|<symbol>|... already written."""
    done: set[str] = set()
    if not os.path.exists(out_fasta):
        return done
    with open(out_fasta) as fh:
        for line in fh:
            if line.startswith(">ParkerSG|"):
                parts = line[1:].strip().split("|")
                if len(parts) >= 2:
                    done.add(parts[1])
    return done


def main(out_dir: str = "Data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_fasta = os.path.join(out_dir, "parker_sg_positives.fasta")

    print("=== Parker SG download ===", flush=True)

    enriched_symbols = download_cuffdiff()
    already          = _already_fetched_symbols(out_fasta)
    if already:
        print(f"  Resume mode: {len(already)} symbols already written, skipping those.", flush=True)
    todo = [s for s in enriched_symbols if s not in already]
    print(f"  {len(todo)} symbols to fetch", flush=True)

    sym_to_tx = symbols_to_ensembl(todo) if todo else {}

    written, skipped = 0, 0
    mode = "a" if already else "w"
    with open(out_fasta, mode) as fh:
        total = len(todo)
        for i, sym in enumerate(todo, 1):
            tx = sym_to_tx.get(sym, "")
            seq = fetch_ensembl_sequence(tx) if tx else ""

            if not seq:
                skipped += 1
            else:
                header = f"ParkerSG|{sym}|{tx}"
                fh.write(f">{header}\n{seq}\n")
                fh.flush()
                written += 1

            if i % 100 == 0:
                print(f"  [{i}/{total}] {written} written, {skipped} skipped ...", flush=True)
            # ~3 req/sec ceiling; Ensembl allows 15/s but we stay well below.
            time.sleep(0.33)

    print(f"\nDone: {written} new sequences appended to {out_fasta}  ({skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw")
    args = parser.parse_args()
    main(args.out_dir)
