"""
Re-download smOOPs (Ivanov et al. 2025, Cell Genomics) keeping NATIVE mouse
sequences instead of mapping to human orthologs.

The original Functions/data_collection/download_smoops.py mapped 1,828 mouse
ENSMUSG IDs -> 1,306 human ENSG orthologs and fetched human mRNAs. That
discarded all native mouse sequence information.

This script keeps the mouse sequences directly: fetches the longest cDNA per
mouse Ensembl gene via Ensembl REST /lookup/id/{ENSMUSG...} + /sequence/id/.

Output: Data/raw/multispecies/smoops_mouse_positives.fasta
"""

import argparse
import io
import os
import re
import time
import requests

XLSX_URL    = "https://ars.els-cdn.com/content/image/1-s2.0-S2666979X25003210-mmc2.xlsx"
ENSEMBL_URL = "https://rest.ensembl.org"
TIMEOUT     = 60
MOUSE       = "mus_musculus"


def parse_smoops_xlsx(content: bytes) -> list[str]:
    """Return list of mouse Ensembl gene IDs flagged smOOPs+ in any condition."""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("pip install openpyxl")

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise ValueError("Empty xlsx")

    # Find the real header row (row 0 is a description paragraph).
    header_row_idx = 0
    for i, row in enumerate(rows[:5]):
        non_empty = [str(c).strip() for c in row if c is not None and str(c).strip()]
        if len(non_empty) >= 5 and any("gene_id" in c.lower() or "ensmusg" in c.lower() for c in non_empty):
            header_row_idx = i
            break
    header = [str(c).strip() if c else "" for c in rows[header_row_idx]]
    print(f"  header row index: {header_row_idx}", flush=True)
    print(f"  columns: {header[:10]} ...", flush=True)
    data_rows = rows[header_row_idx + 1 :]

    ensmusg_col = next(
        (i for i, h in enumerate(header)
         if "ensmusg" in h.lower() or "ensembl" in h.lower() or "gene_id" in h.lower()),
        None,
    )
    smoops_cols = [
        i for i, h in enumerate(header)
        if "smoops" in h.lower() or "npsc" in h.lower() or "ppsc" in h.lower() or "dpsc" in h.lower()
    ]
    if ensmusg_col is None:
        raise ValueError(f"Cannot find gene-ID column. header={header}")
    if not smoops_cols:
        raise ValueError(f"Cannot find smOOPs flag columns. header={header}")

    positives = []
    for row in data_rows:
        gid = str(row[ensmusg_col] or "").strip().split(".")[0]
        if not gid.startswith("ENSMUSG"):
            continue
        flags = [row[c] for c in smoops_cols if c < len(row)]
        if any(str(f or "").strip() in ("1", "1.0", "True", "TRUE", "yes", "YES") for f in flags):
            positives.append(gid)
    unique = sorted(set(positives))
    print(f"  {len(unique)} smOOPs-positive mouse genes", flush=True)
    return unique


def fetch_mouse_cdna(ensmusg_id: str, max_retries: int = 4) -> tuple[str, str]:
    """Return (cdna_sequence, transcript_id) for the longest mouse transcript of this gene."""
    url = f"{ENSEMBL_URL}/lookup/id/{ensmusg_id}"
    params = {"expand": 1}
    transcripts = []
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params,
                             headers={"Accept": "application/json"}, timeout=TIMEOUT)
            if r.status_code == 200:
                transcripts = [t["id"] for t in r.json().get("Transcript", [])
                               if t.get("id", "").startswith("ENSMUST")]
                break
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 30))
                continue
            return "", ""
        except requests.RequestException:
            time.sleep(2 ** attempt)
    if not transcripts:
        return "", ""

    best_seq, best_tx = "", ""
    for tx in transcripts[:5]:                  # longest among first 5
        for attempt in range(max_retries):
            try:
                r = requests.get(f"{ENSEMBL_URL}/sequence/id/{tx}",
                                 params={"type": "cdna"},
                                 headers={"Accept": "text/plain"}, timeout=TIMEOUT)
                if r.status_code == 200:
                    seq = r.text.strip().upper().replace("T", "U")
                    if len(seq) > len(best_seq):
                        best_seq, best_tx = seq, tx
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = float(r.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(min(wait, 30))
                    continue
                break
            except requests.RequestException:
                time.sleep(2 ** attempt)
        time.sleep(0.33)                        # ~3 req/sec ceiling
    return best_seq, best_tx


def _already_fetched(out_fasta: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(out_fasta):
        return done
    with open(out_fasta) as f:
        for line in f:
            if line.startswith(">smOOPs_mouse|"):
                parts = line[1:].strip().split("|")
                if len(parts) >= 2:
                    done.add(parts[1])
    return done


def main(out_dir: str = "Data/raw/multispecies") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_fasta = os.path.join(out_dir, "smoops_mouse_positives.fasta")
    print("=== smOOPs (mouse-native) download ===", flush=True)

    print(f"Downloading xlsx from {XLSX_URL} ...", flush=True)
    r = requests.get(XLSX_URL, timeout=TIMEOUT)
    r.raise_for_status()
    mouse_ids = parse_smoops_xlsx(r.content)

    done = _already_fetched(out_fasta)
    if done:
        print(f"  resume: {len(done)} already fetched", flush=True)
    todo = [g for g in mouse_ids if g not in done]
    print(f"  fetching mouse cDNA for {len(todo)} genes (~3 req/sec, ETA ~{len(todo)*0.4/60:.0f} min) ...",
          flush=True)

    written, skipped = 0, 0
    mode = "a" if done else "w"
    with open(out_fasta, mode) as fh:
        for i, gid in enumerate(todo, 1):
            seq, tx = fetch_mouse_cdna(gid)
            if not seq:
                skipped += 1
            else:
                fh.write(f">smOOPs_mouse|{gid}|{tx}|Mus_musculus\n{seq}\n")
                fh.flush()
                written += 1
            if i % 100 == 0:
                print(f"  [{i}/{len(todo)}] {written} written, {skipped} skipped ...", flush=True)

    print(f"\nDone: {written} new mouse cDNA sequences appended to {out_fasta} ({skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw/multispecies")
    args = parser.parse_args()
    main(args.out_dir)
