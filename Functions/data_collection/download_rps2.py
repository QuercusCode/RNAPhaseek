"""
Download RPS 2.0 reviewed RNA LLPS entries.

API endpoint: https://gateway.rjmart.cn/bigdata/table/page
- Fetches all reviewed entries (reviewed != 0) — ~517 RNAs
- Retrieves the stored RNA sequence for each entry
- Converts DNA alphabet (T) to RNA (U)
- Writes Data/raw/rps2_positives.fasta

Run:
    python -m Functions.data_collection.download_rps2
"""

import os
import sys
import time
import argparse
import requests

API = "https://gateway.rjmart.cn/bigdata/table/page"
HEADERS = {"Content-Type": "application/json"}
TIMEOUT = 30


def _post(payload: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(API, json=payload, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{retries}] {e}  — waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"API failed after {retries} attempts")


def _rows_as_dicts(response: dict) -> list[dict]:
    """Convert the API's columnNames + columnValues arrays into a list of dicts."""
    d = response.get("data") or {}
    names = d.get("columnNames") or []
    values = d.get("columnValues") or []
    return [dict(zip(names, row)) for row in values]


def fetch_reviewed_entries(page_size: int = 250) -> list[dict]:
    """Return all reviewed RPS 2.0 browse-table rows."""
    payload = {
        "project": "zhongda",
        "search": {
            "source": "rps_browse_table",
            "where": {"and": [{"field": "reviewed", "operator": "!=", "value": 0}]},
            "orders": [],
        },
        "pageNo": 1,
        "pageSize": page_size,
    }
    data = _post(payload)
    rows = _rows_as_dicts(data)
    total = data.get("total", len(rows))
    print(f"  Found {total} reviewed entries (fetched {len(rows)})", flush=True)

    all_rows = list(rows)
    page = 2
    while len(all_rows) < total:
        payload["pageNo"] = page
        chunk = _rows_as_dicts(_post(payload))
        if not chunk:
            break
        all_rows.extend(chunk)
        page += 1

    return all_rows


def fetch_sequence(rps_id: str) -> str:
    """Fetch the stored sequence for one RNA by its rpsId."""
    payload = {
        "project": "zhongda",
        "search": {
            "source": "rps_rna_sequence",
            "where": {"and": [{"field": "rpsId", "operator": "=", "value": rps_id}]},
            "orders": [],
        },
        "pageNo": 1,
        "pageSize": 1,
    }
    try:
        rows = _rows_as_dicts(_post(payload))
        if rows:
            return rows[0].get("sequence", "")
    except Exception:
        pass
    return ""


def to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def main(out_dir: str = "Data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_fasta = os.path.join(out_dir, "rps2_positives.fasta")

    print("=== RPS 2.0 download ===", flush=True)
    entries = fetch_reviewed_entries()

    written, skipped = 0, 0
    with open(out_fasta, "w") as fh:
        for i, row in enumerate(entries, 1):
            rps_id   = str(row.get("rpsId") or "")
            gene_sym = str(row.get("geneSymbol") or rps_id)
            organism = str(row.get("organism") or "").replace(" ", "_")
            tx_id    = str(row.get("transcriptId") or "")

            seq = fetch_sequence(rps_id) if rps_id else ""

            if not seq:
                if i % 50 == 0 or i <= 5:
                    print(f"  [{i}/{len(entries)}] SKIP {gene_sym} ({rps_id}) - no sequence", flush=True)
                skipped += 1
                continue

            rna_seq = to_rna(seq)
            header  = f"RPS2|{rps_id}|{gene_sym}|{tx_id}|{organism}".replace(" ", "_")
            fh.write(f">{header}\n{rna_seq}\n")
            written += 1

            if i % 50 == 0:
                print(f"  {i}/{len(entries)} processed …", flush=True)
            time.sleep(0.1)   # be polite to the API

    print(f"\nDone: {written} sequences written to {out_fasta}  ({skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw")
    args = parser.parse_args()
    main(args.out_dir)
