"""
Download G4RNA screener database entries (G-quadruplex-forming RNAs).

G4RNA: http://scottgroup.med.usherbrooke.ca/G4RNA/  (intermittent)
  - 334 G4-forming RNA sequences with biophysical annotations
  - Contact: jean-michel.garant@usherbrooke.ca for bulk data if server is down

Strategy:
  1. Try the known CSV/XLS export endpoint
  2. Fall back to the search/browse API
  3. For any entry without an embedded sequence, fetch from NCBI
  4. Write Data/raw/g4rna_positives.fasta

Note: G4-forming RNAs are LLPS-relevant because G-quadruplex structures
promote phase separation (e.g. TERRA, NEAT1, MALAT1). Use with caution —
G4-forming alone ≠ LLPS; filter by manually curated entries or combine with
experimental validation. Recommended: use only entries that also appear in
another LLPS database (e.g. RPS 2.0 or RNAPhaSep overlap).

Run:
    python -m Functions.data_collection.download_g4rna
"""

import io
import os
import re
import time
import argparse
import requests

BASE_URL    = "http://scottgroup.med.usherbrooke.ca/G4RNA"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TIMEOUT     = 30
MAX_RETRIES = 4


def _get(url: str, stream: bool = False, **kwargs) -> requests.Response | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=stream, **kwargs)
            if r.status_code == 200:
                return r
            print(f"  HTTP {r.status_code}  (attempt {attempt+1})", flush=True)
        except Exception as e:
            print(f"  {type(e).__name__}: {e}  (attempt {attempt+1}/{MAX_RETRIES})", flush=True)
        time.sleep(2 ** attempt)
    return None


# ── Attempt 1: direct XLS / CSV export ───────────────────────────────────────

def try_xls_export() -> list[dict] | None:
    candidates = [
        f"{BASE_URL}/download",
        f"{BASE_URL}/download.xls",
        f"{BASE_URL}/export",
        f"{BASE_URL}/G4RNA_screener_results.xls",
    ]
    for url in candidates:
        r = _get(url)
        if r is None:
            continue
        ct = r.headers.get("Content-Type", "")
        raw = r.content

        # XLS / XLSX
        if "excel" in ct or "spreadsheet" in ct or url.endswith(".xls"):
            return _parse_xls(raw)

        # CSV
        if "csv" in ct or "text/plain" in ct:
            return _parse_csv(r.text)

        # FASTA
        if r.text.strip().startswith(">"):
            entries = []
            for block in r.text.strip().split(">")[1:]:
                lines  = block.splitlines()
                header = lines[0].strip()
                seq    = "".join(lines[1:]).upper().replace("T", "U")
                entries.append({"name": header, "ncbi_id": "", "sequence": seq})
            return entries

    return None


def _parse_xls(content: bytes) -> list[dict]:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        rows  = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        header = [str(c or "").lower().strip() for c in rows[0]]
        return _rows_to_entries(rows[1:], header)
    except Exception:
        pass
    try:
        import xlrd
        wb   = xlrd.open_workbook(file_contents=content)
        ws   = wb.sheet_by_index(0)
        rows = [ws.row_values(i) for i in range(ws.nrows)]
        if not rows:
            return []
        header = [str(c).lower().strip() for c in rows[0]]
        return _rows_to_entries(rows[1:], header)
    except Exception:
        return []


def _parse_csv(text: str) -> list[dict]:
    import csv
    reader = csv.DictReader(io.StringIO(text))
    entries = []
    for row in reader:
        name    = row.get("rna", "") or row.get("name", "") or row.get("gene", "")
        ncbi_id = row.get("accession", "") or row.get("ncbi_id", "") or row.get("genbank", "")
        seq     = row.get("sequence", "") or row.get("seq", "")
        entries.append({"name": name.strip(), "ncbi_id": ncbi_id.strip(),
                        "sequence": seq.strip().upper().replace("T", "U")})
    return entries


def _rows_to_entries(rows, header: list[str]) -> list[dict]:
    # Try to locate key columns by header name patterns
    name_col = next((i for i, h in enumerate(header)
                     if any(k in h for k in ("rna", "name", "gene", "id"))), 0)
    acc_col  = next((i for i, h in enumerate(header)
                     if any(k in h for k in ("accession", "ncbi", "genbank", "refseq"))), -1)
    seq_col  = next((i for i, h in enumerate(header)
                     if any(k in h for k in ("sequence", "seq"))), -1)
    entries = []
    for row in rows:
        name    = str(row[name_col] or "") if name_col < len(row) else ""
        ncbi_id = str(row[acc_col]  or "") if acc_col >= 0 and acc_col < len(row) else ""
        seq     = str(row[seq_col]  or "") if seq_col >= 0 and seq_col < len(row) else ""
        if name or ncbi_id:
            entries.append({"name": name.strip(), "ncbi_id": ncbi_id.strip(),
                            "sequence": seq.strip().upper().replace("T", "U")})
    return entries


# ── Attempt 2: scrape browse table ───────────────────────────────────────────

def scrape_browse() -> list[dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")

    entries: list[dict] = []
    page = 1
    while True:
        url = f"{BASE_URL}/browse?page={page}"
        r = _get(url)
        if r is None:
            break
        soup  = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            break
        rows_html = table.find_all("tr")[1:]
        if not rows_html:
            break
        for tr in rows_html:
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) >= 2:
                ncbi_id = tds[1] if len(tds) > 1 else ""
                # Extract clean accession (NM_..., NR_..., etc.)
                acc_match = re.search(r"(NM_\d+|NR_\d+|XM_\d+|XR_\d+)", tds[1])
                if acc_match:
                    ncbi_id = acc_match.group(1)
                entries.append({"name": tds[0], "ncbi_id": ncbi_id,
                                "sequence": tds[-1] if len(tds) > 3 else ""})
        page += 1
        time.sleep(0.3)
        # Stop if page seems empty or unchanged
        if len(rows_html) < 5:
            break

    print(f"  Scraped {len(entries)} entries", flush=True)
    return entries


def fetch_ncbi_sequence(ncbi_id: str) -> str:
    if not ncbi_id:
        return ""
    params = {"db": "nuccore", "id": ncbi_id, "rettype": "fasta", "retmode": "text"}
    try:
        r = requests.get(NCBI_EFETCH, params=params, timeout=TIMEOUT)
        if r.status_code == 200 and r.text.strip().startswith(">"):
            lines = r.text.strip().splitlines()
            seq   = "".join(l for l in lines if not l.startswith(">"))
            return seq.upper().replace("T", "U")
    except Exception:
        pass
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main(out_dir: str = "Data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_fasta = os.path.join(out_dir, "g4rna_positives.fasta")

    print("=== G4RNA download ===", flush=True)
    print("Note: Use G4RNA entries cautiously — G4-forming ≠ LLPS.", flush=True)
    print("Recommend intersecting with RPS2/RNAPhaSep before adding to training set.", flush=True)

    entries = try_xls_export()
    if not entries:
        print("  Bulk export failed; trying browse scrape …", flush=True)
        entries = scrape_browse()

    if not entries:
        print("  ✗ G4RNA server is unavailable.")
        print("    Retry later, or email jean-michel.garant@usherbrooke.ca for a data dump.")
        return

    written, skipped = 0, 0
    with open(out_fasta, "w") as fh:
        for i, e in enumerate(entries, 1):
            seq = e.get("sequence", "")
            if not seq and e.get("ncbi_id"):
                seq = fetch_ncbi_sequence(e["ncbi_id"])
                time.sleep(0.35)

            if not seq:
                skipped += 1
                continue

            header = f"G4RNA|{e['name']}|{e.get('ncbi_id','')}".replace(" ", "_")
            fh.write(f">{header}\n{seq}\n")
            written += 1

    print(f"\nDone: {written} sequences written to {out_fasta}  ({skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw")
    args = parser.parse_args()
    main(args.out_dir)
