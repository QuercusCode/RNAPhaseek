"""
Download smOOPs mouse stress-granule RNA data and map to human sequences.

Source:
  Ivanov et al. 2025, Cell Genomics
  Supplementary Table S2 (mmc2.xlsx): 3,060 mouse genes with smOOPs flags
  URL: https://ars.els-cdn.com/content/image/1-s2.0-S2666979X25003210-mmc2.xlsx

Pipeline:
  1. Download the xlsx
  2. Filter rows where any smOOPs flag column == 1 (nPSCs / pPSCs / dPSCs)
  3. Map mouse Ensembl gene IDs (ENSMUSG) → human orthologs (ENSG) via
     Ensembl REST /homology endpoint
  4. Fetch longest mRNA for each human gene from Ensembl
  5. Write Data/raw/smoops_positives.fasta

Run:
    python -m Functions.data_collection.download_smoops
"""

import io
import os
import time
import argparse
import requests

XLSX_URL    = "https://ars.els-cdn.com/content/image/1-s2.0-S2666979X25003210-mmc2.xlsx"
ENSEMBL_URL = "https://rest.ensembl.org"
TIMEOUT     = 60


# ── Step 1: Parse the xlsx ────────────────────────────────────────────────────

def parse_smoops_xlsx(content: bytes) -> list[str]:
    """
    Return list of mouse Ensembl gene IDs that are smOOPs-positive in any
    condition (nPSCs, pPSCs, or dPSCs).
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required: pip install openpyxl")

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise ValueError("Empty xlsx")

    # Row 0 is a single-cell file-description paragraph; the real header is row 1.
    header_row_idx = 0
    for i, row in enumerate(rows[:5]):
        non_empty = [str(c).strip() for c in row if c is not None and str(c).strip()]
        if len(non_empty) >= 5 and any("gene_id" in c.lower() or "ensmusg" in c.lower() for c in non_empty):
            header_row_idx = i
            break

    header = [str(c).strip() if c else "" for c in rows[header_row_idx]]
    print(f"  Header row index: {header_row_idx}", flush=True)
    print(f"  Columns: {header[:10]} ...", flush=True)
    data_rows = rows[header_row_idx + 1 :]

    # Locate the Ensembl ID column and smOOPs flag columns
    ensmusg_col = next(
        (i for i, h in enumerate(header)
         if "ensmusg" in h.lower() or "ensembl" in h.lower() or "gene_id" in h.lower()),
        None,
    )
    smoops_cols = [
        i for i, h in enumerate(header)
        if "smoops" in h.lower() or "npsc" in h.lower() or "ppsc" in h.lower() or "dpsc" in h.lower()
    ]

    if ensmusg_col is None and data_rows:
        for i in range(len(header)):
            if i < len(data_rows[0]) and str(data_rows[0][i] or "").startswith("ENSMUSG"):
                ensmusg_col = i
                break

    if ensmusg_col is None:
        raise ValueError(f"Cannot find Ensembl gene ID column. Header: {header}")
    if not smoops_cols:
        raise ValueError(f"Cannot find smOOPs flag columns. Header: {header}")

    print(f"  Ensembl col={ensmusg_col}, smOOPs cols={smoops_cols}", flush=True)

    positives: list[str] = []
    for row in data_rows:
        gene_id = str(row[ensmusg_col] or "").strip().split(".")[0]
        if not gene_id.startswith("ENSMUSG"):
            continue
        flags = [row[c] for c in smoops_cols if c < len(row)]
        if any(str(f or "").strip() in ("1", "1.0", "True", "TRUE", "yes", "YES") for f in flags):
            positives.append(gene_id)

    unique = sorted(set(positives))
    print(f"  {len(unique)} smOOPs-positive mouse genes", flush=True)
    return unique


# ── Step 2: Mouse → Human ortholog mapping via Ensembl REST ──────────────────

def mouse_to_human_orthologs(mouse_ids: list[str], batch: int = 50) -> dict[str, str]:
    """
    Return {ENSMUSG_id: ENSG_human_id} using Ensembl /homology REST endpoint.
    """
    print(f"Mapping {len(mouse_ids)} mouse genes → human orthologs …", flush=True)
    result: dict[str, str] = {}

    for i, mid in enumerate(mouse_ids, 1):
        url    = f"{ENSEMBL_URL}/homology/id/mus_musculus/{mid}"
        params = {
            "type":         "orthologues",
            "target_species": "homo_sapiens",
        }
        try:
            r = requests.get(url, params=params,
                             headers={"Accept": "application/json"},
                             timeout=TIMEOUT)
            if r.status_code != 200:
                continue
            homologies = r.json().get("data", [{}])[0].get("homologies", [])
            # Prefer 1:1 ortholog
            for h in homologies:
                if h.get("type") == "ortholog_one2one":
                    result[mid] = h["target"]["id"]
                    break
            else:
                if homologies:
                    result[mid] = homologies[0]["target"]["id"]
        except Exception:
            pass

        if i % 100 == 0:
            print(f"  {i}/{len(mouse_ids)} mapped: {len(result)} found …", flush=True)
        time.sleep(0.08)

    print(f"  {len(result)}/{len(mouse_ids)} orthologs found", flush=True)
    return result


# ── Step 3: Fetch mRNA sequence from Ensembl ─────────────────────────────────

def fetch_longest_cdna(ensg_id: str) -> tuple[str, str]:
    """
    Return (cdna_sequence, transcript_id) for the longest transcript of a human gene.
    """
    # Get all transcripts for this gene
    url = f"{ENSEMBL_URL}/lookup/id/{ensg_id}"
    params = {"content-type": "application/json", "expand": 1}
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return "", ""
        gene_data    = r.json()
        transcripts  = [t["id"] for t in gene_data.get("Transcript", [])
                        if t.get("id", "").startswith("ENST")]
    except Exception:
        return "", ""

    best_seq, best_tx = "", ""
    for tx in transcripts[:5]:
        seq_url = f"{ENSEMBL_URL}/sequence/id/{tx}"
        try:
            sr = requests.get(seq_url,
                              params={"content-type": "text/plain", "type": "cdna"},
                              timeout=TIMEOUT)
            if sr.status_code == 200:
                seq = sr.text.strip().upper().replace("T", "U")
                if len(seq) > len(best_seq):
                    best_seq, best_tx = seq, tx
        except Exception:
            pass
        time.sleep(0.05)

    return best_seq, best_tx


# ── Main ──────────────────────────────────────────────────────────────────────

def main(out_dir: str = "Data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_fasta = os.path.join(out_dir, "smoops_positives.fasta")

    print("=== smOOPs download ===", flush=True)

    print(f"Downloading xlsx from {XLSX_URL} …", flush=True)
    r = requests.get(XLSX_URL, timeout=TIMEOUT)
    r.raise_for_status()

    mouse_ids      = parse_smoops_xlsx(r.content)
    mouse_to_human = mouse_to_human_orthologs(mouse_ids)

    written, no_ortholog, no_seq = 0, 0, 0
    with open(out_fasta, "w") as fh:
        total = len(mouse_ids)
        for i, mid in enumerate(mouse_ids, 1):
            ensg = mouse_to_human.get(mid, "")
            if not ensg:
                no_ortholog += 1
                continue

            seq, tx_id = fetch_longest_cdna(ensg)
            if not seq:
                no_seq += 1
                continue

            header = f"smOOPs|{mid}|{ensg}|{tx_id}"
            fh.write(f">{header}\n{seq}\n")
            written += 1

            if i % 100 == 0:
                print(f"  [{i}/{total}] written={written} no_ortholog={no_ortholog} "
                      f"no_seq={no_seq} …", flush=True)
            time.sleep(0.08)

    print(f"\nDone: {written} sequences written to {out_fasta}")
    print(f"  No human ortholog: {no_ortholog}")
    print(f"  No sequence found: {no_seq}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw")
    args = parser.parse_args()
    main(args.out_dir)
