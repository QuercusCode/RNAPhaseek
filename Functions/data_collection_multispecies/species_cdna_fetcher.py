"""
Generic species-aware symbol/ID -> cDNA fetcher.

For each canonical LLPS-positive gene list (from literature/supplementary tables),
this module converts gene symbols or Ensembl IDs into their RNA sequences via:
  - Ensembl REST     (for mouse, worm, fly, plant)
  - SGD direct files (for yeast)

All fetched sequences are returned as RNA (T -> U).
"""

import os
import re
import time
import gzip
import requests
from typing import Optional

ENSEMBL_URL = "https://rest.ensembl.org"
TIMEOUT     = 60


def _safe_get(url, params=None, headers=None, max_retries: int = 4):
    headers = headers or {"Accept": "application/json"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 30))
                continue
            return None
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def fetch_cdna_by_ensembl_gene(gene_id: str, species: str,
                                sleep_between: float = 0.33) -> tuple[str, str]:
    """
    Return (cdna_RNA, transcript_id) for the longest cDNA of an Ensembl gene.
    Works for any species whose IDs are in Ensembl (mouse ENSMUSG, worm WBGene,
    fly FBgn, plant AT...).
    """
    r = _safe_get(f"{ENSEMBL_URL}/lookup/id/{gene_id}", params={"expand": 1})
    if r is None:
        return "", ""
    transcripts = [t["id"] for t in r.json().get("Transcript", [])
                   if t.get("id", "")]
    best_seq, best_tx = "", ""
    for tx in transcripts[:5]:
        r = _safe_get(f"{ENSEMBL_URL}/sequence/id/{tx}",
                      params={"type": "cdna"},
                      headers={"Accept": "text/plain"})
        if r is not None:
            seq = r.text.strip().upper().replace("T", "U")
            if len(seq) > len(best_seq):
                best_seq, best_tx = seq, tx
        time.sleep(sleep_between)
    return best_seq, best_tx


def fetch_cdna_by_symbol(symbol: str, species: str,
                          sleep_between: float = 0.33) -> tuple[str, str]:
    """
    Return (cdna_RNA, transcript_id) for the longest cDNA of a gene symbol.
    species: 'mus_musculus' / 'caenorhabditis_elegans' / 'drosophila_melanogaster' etc.

    Uses /lookup/symbol/ (which returns the gene record incl. canonical_transcript)
    then fetches all the gene's transcripts via /lookup/id/?expand=1.
    """
    # Step 1: symbol -> gene record
    r = _safe_get(f"{ENSEMBL_URL}/lookup/symbol/{species}/{symbol}")
    if r is None:
        return "", ""
    gene = r.json()
    gene_id = gene.get("id", "")
    if not gene_id:
        return "", ""

    # Step 2: gene -> all transcripts
    r = _safe_get(f"{ENSEMBL_URL}/lookup/id/{gene_id}", params={"expand": 1})
    if r is None:
        return "", ""
    transcripts = [t["id"] for t in r.json().get("Transcript", [])
                   if t.get("id", "")]
    if not transcripts:
        # Fall back to canonical transcript if expand failed
        cano = gene.get("canonical_transcript", "").rstrip(".")
        if cano:
            transcripts = [cano]

    # Step 3: pick the longest cDNA among the first few transcripts
    best_seq, best_tx = "", ""
    for tx in transcripts[:5]:
        # Some species' canonical_transcript ends with a stable-id suffix (e.g. "F53G12.5b.1.")
        # The /sequence/id/ endpoint accepts the bare stable ID.
        tx_clean = tx.rstrip(".")
        r = _safe_get(f"{ENSEMBL_URL}/sequence/id/{tx_clean}",
                      params={"type": "cdna"},
                      headers={"Accept": "text/plain"})
        if r is not None:
            seq = r.text.strip().upper().replace("T", "U")
            if len(seq) > len(best_seq):
                best_seq, best_tx = seq, tx_clean
        time.sleep(sleep_between)
    return best_seq, best_tx


def fetch_yeast_orf_from_sgd(systematic_or_standard: str,
                              sgd_fasta_gz: str = "Data/raw/multispecies/refs/saccharomyces_cerevisiae_cdna.fa.gz",
                              ) -> tuple[str, str]:
    """
    Yeast lookup against SGD orf_coding_all FASTA. Indexes both systematic
    IDs (YAL001C) and standard names (PUB1) so either works.
    """
    if not hasattr(fetch_yeast_orf_from_sgd, "_index"):
        if not os.path.exists(sgd_fasta_gz):
            raise FileNotFoundError(
                f"{sgd_fasta_gz} not found. Run download_reference_transcriptomes first."
            )
        # Two-pass build:
        #   pass 1: walk fasta, accumulate (systematic, standard, sequence).
        #   pass 2: store sequence under BOTH keys (upper-case for case-insensitive lookup).
        idx_to_sys = {}   # any-key -> systematic ID
        seq_by_sys = {}   # systematic -> sequence
        with gzip.open(sgd_fasta_gz, "rt") as fh:
            sys_id, std_name, parts = None, None, []
            for line in fh:
                line = line.rstrip()
                if line.startswith(">"):
                    if sys_id is not None:
                        seq_by_sys[sys_id] = "".join(parts)
                    # SGD header: ">SYSTEMATIC [STANDARD] SGDID:... , Chr..."
                    head = line[1:].split(",", 1)[0]
                    toks = head.split()
                    sys_id  = toks[0]
                    std_name = None
                    if len(toks) >= 2 and not toks[1].startswith("SGDID:") and toks[1] != sys_id:
                        std_name = toks[1]
                    idx_to_sys[sys_id.upper()] = sys_id
                    if std_name:
                        idx_to_sys[std_name.upper()] = sys_id
                    parts = []
                else:
                    parts.append(line)
            if sys_id is not None:
                seq_by_sys[sys_id] = "".join(parts)
        fetch_yeast_orf_from_sgd._idx_to_sys = idx_to_sys
        fetch_yeast_orf_from_sgd._seq_by_sys = seq_by_sys
        print(f"  [SGD index] {len(seq_by_sys)} ORFs ({len(idx_to_sys)} aliases) loaded",
              flush=True)
    idx_to_sys = fetch_yeast_orf_from_sgd._idx_to_sys
    seq_by_sys = fetch_yeast_orf_from_sgd._seq_by_sys
    sys_id = idx_to_sys.get(systematic_or_standard.upper(), "")
    if not sys_id:
        return "", ""
    seq = seq_by_sys.get(sys_id, "")
    if not seq:
        return "", ""
    return seq.upper().replace("T", "U"), sys_id


def write_positives_fasta(records: list[tuple[str, str, str, str]],
                          out_path: str,
                          species_label: str) -> int:
    """
    records: list of (gene_id, transcript_id, sequence, source_tag).
    Header format: >llps_{species}|{gene_id}|{transcript_id}|{source_tag}
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w") as fh:
        for gid, tx, seq, src in records:
            if not seq or len(seq) < 50:
                continue
            fh.write(f">llps_{species_label}|{gid}|{tx}|{src}\n{seq}\n")
            written += 1
    return written
