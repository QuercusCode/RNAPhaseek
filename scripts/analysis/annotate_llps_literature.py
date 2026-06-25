#!/usr/bin/env python
"""
Annotate the human transcriptome scoring results with PubMed LLPS literature.

For each scored transcript:
  - P(LLPS) > 0.9  : full search + abstract fetch + protein-free classification
  - P(LLPS) > 0.5  : search hit count only
  - P(LLPS) <= 0.5 : skipped (mark "not searched")

Outputs a TSV annotation file that can be merged into the Excel workbook.

Usage:
    python scripts/analysis/annotate_llps_literature.py \
        --db  outputs/human_transcriptome/rnaphaseek_human.db \
        --out outputs/human_transcriptome/llps_literature_annotations.tsv
"""

import argparse, sqlite3, time, json, re, sys
from pathlib import Path
from urllib import request, parse, error as urlerror

# ── NCBI E-utilities ───────────────────────────────────────────────────────
EMAIL = "amirmohammad.cheraghali@inserm.fr"
TOOL  = "RNAPhaseek"
BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# P-threshold: full abstract analysis above, count-only below, skip below 0.5
FULL_THRESHOLD  = 0.9
COUNT_THRESHOLD = 0.5

# ── LLPS search terms ──────────────────────────────────────────────────────
LLPS_QUERY = (
    '("{gene}"[Title/Abstract] OR "{gene}"[Gene Name]) AND '
    '("phase separation"[Title/Abstract] OR "LLPS"[Title/Abstract] OR '
    '"liquid-liquid phase"[Title/Abstract] OR "condensate"[Title/Abstract] OR '
    '"phase-separated"[Title/Abstract] OR "membraneless organelle"[Title/Abstract])'
)

# ── Protein-free positive indicators ──────────────────────────────────────
PROT_FREE_POS = [
    "protein-free", "protein free", "rna-only", "rna only",
    "without protein", "in the absence of protein", "pure rna",
    "naked rna", "rna alone undergoes", "rna-mediated condensation",
    "rna self-assembly", "rna self assembly", "intrinsic phase",
    "protein-independent", "protein independent phase",
]
# ── Protein-dependent indicators ──────────────────────────────────────────
PROT_DEP_IND = [
    "stress granule", "p-body", "p body", "processing body",
    "fus", "tdp-43", "g3bp", "hnrnp", "rbp", "rna-binding protein",
    "ewsr1", "taf15", "rbfox", "ybx1", "sfpq", "nono", "caprin1",
    "requires protein", "protein-driven", "protein-mediated phase",
    "with recombinant", "together with protein",
]


def _get(url, retries=3, delay=0.34):
    """GET with retry and rate-limit delay."""
    time.sleep(delay)
    for attempt in range(retries):
        try:
            with request.urlopen(url, timeout=15) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None


def search_count(gene):
    """Return (count, pmids_top5) from PubMed for a gene + LLPS query."""
    if not gene or gene.startswith("ENSG") and len(gene) > 15:
        # Raw Ensembl IDs without gene symbols rarely have lit — skip
        return 0, []
    q = LLPS_QUERY.format(gene=gene.replace('"', ''))
    url = (f"{BASE}/esearch.fcgi?db=pubmed&term={parse.quote(q)}"
           f"&retmax=10&usehistory=y&email={EMAIL}&tool={TOOL}&retmode=json")
    txt = _get(url)
    if not txt:
        return -1, []
    try:
        j = json.loads(txt)
        count = int(j["esearchresult"]["count"])
        pmids = j["esearchresult"]["idlist"][:5]
        return count, pmids
    except Exception:
        return -1, []


def fetch_abstracts(pmids):
    """Fetch PubMed abstracts for a list of PMIDs. Returns list of dicts."""
    if not pmids:
        return []
    ids = ",".join(pmids)
    url = (f"{BASE}/efetch.fcgi?db=pubmed&id={ids}&rettype=abstract"
           f"&retmode=xml&email={EMAIL}&tool={TOOL}")
    txt = _get(url, delay=0.5)
    if not txt:
        return []
    articles = []
    for m in re.finditer(
        r"<PubmedArticle>(.*?)</PubmedArticle>", txt, re.DOTALL
    ):
        art = m.group(1)
        pmid_m    = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
        title_m   = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.DOTALL)
        abstract_m = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.DOTALL)
        year_m    = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", art, re.DOTALL)
        articles.append({
            "pmid":     pmid_m.group(1) if pmid_m else "",
            "title":    re.sub(r"<[^>]+>", "", title_m.group(1)).strip() if title_m else "",
            "abstract": re.sub(r"<[^>]+>", "", abstract_m.group(1)).strip() if abstract_m else "",
            "year":     year_m.group(1) if year_m else "",
        })
    return articles


def classify_protein_free(articles):
    """
    Given fetched abstracts, classify protein-free evidence.
    Returns: "protein_free" | "protein_dependent" | "ambiguous" | "no_abstract"
    """
    if not articles:
        return "no_abstract"
    combined = " ".join(
        (a["title"] + " " + a["abstract"]).lower() for a in articles
    )
    pf_score = sum(1 for kw in PROT_FREE_POS if kw in combined)
    pd_score = sum(1 for kw in PROT_DEP_IND if kw in combined)
    if pf_score > 0 and pd_score == 0:
        return "protein_free"
    if pf_score > 0 and pd_score > 0:
        return "ambiguous_mixed"
    if pd_score > 0:
        return "protein_dependent"
    return "ambiguous"   # LLPS mentioned but no protein/RNA-free distinction


def classify_overall(count, pf_class, p_llps):
    """Assign a final LLPS classification label."""
    if count == 0:
        return "novel_no_literature"
    if count == -1:
        return "search_failed"
    if pf_class == "protein_free":
        return "known_protein_free"
    if pf_class == "protein_dependent":
        return "known_protein_dependent"
    if pf_class in ("ambiguous", "ambiguous_mixed", "no_abstract"):
        return "known_LLPS_classification_unclear"
    return "known_LLPS"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db",  default="outputs/human_transcriptome/rnaphaseek_human.db")
    ap.add_argument("--out", default="outputs/human_transcriptome/llps_literature_annotations.tsv")
    ap.add_argument("--resume", action="store_true",
                    help="Skip genes already in the output file")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT tx_id, gene_name, biotype, p_llps, length FROM transcripts ORDER BY p_llps DESC"
    ).fetchall()

    # Resume: load already-processed gene names
    done_genes = set()
    if args.resume and Path(args.out).exists():
        with open(args.out) as f:
            for line in f:
                if line.startswith("#") or line.startswith("tx_id"):
                    continue
                parts = line.split("\t")
                if parts:
                    done_genes.add(parts[0])
        print(f"[resume] {len(done_genes)} entries already done")

    out_mode = "a" if args.resume and Path(args.out).exists() else "w"
    fout = open(args.out, out_mode)

    if out_mode == "w":
        fout.write(
            "tx_id\tgene_name\tbiotype\tp_llps\tllps_literature\t"
            "n_papers\tprotein_free_class\tllps_classification\t"
            "top_pmids\ttop_title\n"
        )

    to_process_full  = [r for r in rows if r["p_llps"] > FULL_THRESHOLD  and r["tx_id"] not in done_genes]
    to_process_count = [r for r in rows if COUNT_THRESHOLD < r["p_llps"] <= FULL_THRESHOLD and r["tx_id"] not in done_genes]
    to_skip          = [r for r in rows if r["p_llps"] <= COUNT_THRESHOLD and r["tx_id"] not in done_genes]

    total_search = len(to_process_full) + len(to_process_count)
    print(f"[*] {len(to_process_full)} entries for full annotation (P>{FULL_THRESHOLD})")
    print(f"[*] {len(to_process_count)} entries for count-only (P>{COUNT_THRESHOLD})")
    print(f"[*] {len(to_skip)} entries skipped (P≤{COUNT_THRESHOLD})")
    print(f"[*] Total PubMed queries: ~{total_search + len(to_process_full)} (abstracts for P>0.9)")

    # ── Full annotation (P > 0.9) ──────────────────────────────────────────
    for i, r in enumerate(to_process_full):
        gene = r["gene_name"]
        print(f"  [{i+1}/{len(to_process_full)}] {gene} (P={r['p_llps']:.4f})", end=" ", flush=True)

        count, pmids = search_count(gene)
        if count > 0:
            articles = fetch_abstracts(pmids)
            pf_class = classify_protein_free(articles)
            top_title = articles[0]["title"][:100] if articles else ""
        else:
            articles, pf_class, top_title = [], "n/a", ""

        classification = classify_overall(count, pf_class, r["p_llps"])
        print(f"→ {count} papers, {pf_class}, {classification}")

        fout.write(
            f"{r['tx_id']}\t{gene}\t{r['biotype']}\t{r['p_llps']:.4f}\t"
            f"{'yes' if count>0 else 'no'}\t"
            f"{max(count,0)}\t{pf_class}\t{classification}\t"
            f"{','.join(pmids[:3])}\t{top_title}\n"
        )
        fout.flush()

    # ── Count-only (0.5 < P ≤ 0.9) ────────────────────────────────────────
    for i, r in enumerate(to_process_count):
        gene = r["gene_name"]
        print(f"  [count {i+1}/{len(to_process_count)}] {gene}", end=" ", flush=True)

        count, pmids = search_count(gene)
        pf_class = "not_fetched"
        classification = classify_overall(count, pf_class, r["p_llps"])
        print(f"→ {count} papers")

        fout.write(
            f"{r['tx_id']}\t{gene}\t{r['biotype']}\t{r['p_llps']:.4f}\t"
            f"{'yes' if count>0 else 'no'}\t"
            f"{max(count,0)}\t{pf_class}\t{classification}\t"
            f"{','.join(pmids[:3])}\t\n"
        )
        fout.flush()

    # ── Skipped ────────────────────────────────────────────────────────────
    print(f"[*] Writing {len(to_skip)} skipped entries ...")
    for r in to_skip:
        fout.write(
            f"{r['tx_id']}\t{r['gene_name']}\t{r['biotype']}\t{r['p_llps']:.4f}\t"
            f"not_searched\t0\tn/a\tnot_searched\t\t\n"
        )

    fout.close()
    print(f"\n[*] Done → {args.out}")


if __name__ == "__main__":
    main()
