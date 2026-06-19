"""
Download RNAPhaSep RNA phase-separation entries.

RNAPhaSep moved to a new domain. The original Functions/data_collection
script targeted http://www.rnaligands.top:8080/RNAphase/ which no longer
resolves. The active site is http://www.rnaphasep.cn/ and serves the
catalogue as a JSON API split by RNA type:

  /api/show_lncRNAs   /api/show_mRNAs   /api/show_mirna   /api/show_rRNAs
  /api/show_snorna    /api/show_snrna   /api/show_pirna   /api/show_siRNAs
  /api/show_virysrna  /api/show_totalrna

Each record has an `rna_sequence` field that may contain one or more
sequences separated by ";|" (corresponding to the same separator used
in the `rnas` field for multi-RNA complexes).

Run:
    python -m Functions.data_collection.download_rnaphase
"""

import argparse
import os
import re
import time
import requests

BASE_URL = "http://www.rnaphasep.cn"
TYPE_ENDPOINTS = [
    "show_lncRNAs", "show_mRNAs", "show_mirna", "show_rRNAs",
    "show_snorna", "show_snrna", "show_pirna", "show_siRNAs",
    "show_virysrna", "show_totalrna",
]
TIMEOUT     = 30
MAX_RETRIES = 4

SEQ_SPLIT = re.compile(r";\|")
VALID_NT  = set("ACGUNT")


def _get_json(url: str) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} from {url}  (attempt {attempt+1})", flush=True)
        except requests.RequestException as e:
            print(f"  {type(e).__name__}: {e}  (attempt {attempt+1}/{MAX_RETRIES})", flush=True)
        time.sleep(2 ** attempt)
    return None


def _clean_sequence(seq: str) -> str:
    s = re.sub(r"\s", "", (seq or "").upper())
    if len(s) < 20:
        return ""
    if not set(s) <= VALID_NT:
        return ""
    return s.replace("T", "U")


def main(out_dir: str = "Data/raw") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_fasta = os.path.join(out_dir, "rnaphase_positives.fasta")
    print("=== RNAPhaSep download ===", flush=True)

    written = skipped = 0
    seen_keys: set[tuple[str, str]] = set()
    with open(out_fasta, "w") as fh:
        for ep in TYPE_ENDPOINTS:
            data = _get_json(f"{BASE_URL}/api/{ep}")
            if not data:
                print(f"  {ep}: skipped (no response)", flush=True)
                continue
            items = data.get("list") or []
            ep_written = 0
            for it in items:
                f = it.get("fields", {})
                rpsid    = str(f.get("rpsid") or "").strip()
                rna_cls  = str(f.get("rna_classification") or "").strip()
                names    = SEQ_SPLIT.split(str(f.get("rnas") or ""))
                seqs     = SEQ_SPLIT.split(str(f.get("rna_sequence") or ""))
                classes  = SEQ_SPLIT.split(rna_cls)
                for i, raw in enumerate(seqs):
                    seq = _clean_sequence(raw)
                    if not seq:
                        skipped += 1
                        continue
                    name = (names[i] if i < len(names) else "").strip() or rpsid
                    cls  = (classes[i] if i < len(classes) else "").strip() or rna_cls
                    key  = (name, seq[:60])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    header = f"RNAPhaSep|{rpsid}|{name}|{cls}".replace(" ", "_")
                    fh.write(f">{header}\n{seq}\n")
                    written += 1
                    ep_written += 1
            print(f"  {ep}: {ep_written} written", flush=True)

    print(f"\nDone: {written} sequences written to {out_fasta}  ({skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw")
    args = parser.parse_args()
    main(args.out_dir)
