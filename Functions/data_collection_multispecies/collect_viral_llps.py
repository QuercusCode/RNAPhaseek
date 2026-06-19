"""
Curated viral RNA LLPS-region subsequences.

PREVIOUS VERSION fetched full viral genomes (up to 30 kb) which the 1022-nt RNA-FM
window then truncated to the 5' end — meaning the model trained on 5'UTR/spike,
NOT the LLPS-relevant region.

THIS VERSION fetches specific, literature-supported subregions known to drive
viral RNA condensate / inclusion-body / viroplasm assembly. Each region is
≤ 1022 nt so it fits the RNA-FM context window without truncation. Each entry
carries an NCBI accession, start/stop coordinates, a human-readable region
label, and a PubMed ID for the supporting study.

Output:
  Data/raw/multispecies/viral_positives.fasta

Header convention:
  >llps_virus|<virus_short>|<acc>:<start>-<end>|<region_label>|PMID:<id>
"""

import argparse
import os
import time
import requests

NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TIMEOUT     = 30
RATE_SLEEP  = 0.34  # NCBI courtesy: ~3 req/sec


# ── Curated viral RNA LLPS-relevant subregions ────────────────────────────────
# Format:
#   (virus_short, accession, start, end, region_label, pubmed_id, biology_note)
# All coordinates are 1-based, inclusive, NCBI conventions.
# Each (end - start + 1) <= 1022 nt to fit RNA-FM context.

VIRAL_RNA_TARGETS = [
    # ── SARS-CoV-2 ────────────────────────────────────────────────────────────
    # N protein RNP condensates form on 5'UTR SL-loops, the frameshift element,
    # the 3'UTR s2m, and the N gene mRNA itself.
    ("SARS-CoV-2", "NC_045512.2",    1,   300,
     "5UTR_SL1-SL3",       "32976812",
     "5'UTR stem-loops 1-3 plus ORF1ab start; high-affinity N-protein RNA condensate driver (Iserman 2020 Mol Cell)"),

    ("SARS-CoV-2", "NC_045512.2", 13400, 13550,
     "Frameshift_element", "33564026",
     "Programmed -1 ribosomal frameshift element + slippery sequence; N-binding hotspot (Roman 2021 Science)"),

    ("SARS-CoV-2", "NC_045512.2", 29550, 29903,
     "3UTR_s2m_HVR",       "32896890",
     "3'UTR stem-loop II m and hypervariable region; condensate scaffold (Savastano 2020 Nat Comm)"),

    ("SARS-CoV-2", "NC_045512.2", 28274, 29240,
     "N_gene_LCD-binding", "32976812",
     "Nucleocapsid (N) gene mRNA; promotes condensate via LCD-RNA contacts (Iserman 2020 Mol Cell)"),

    # ── HCV (Hepatitis C virus) ──────────────────────────────────────────────
    # NS5A condensates form on 5'UTR IRES and 3'UTR stem-loops.
    ("HCV",        "NC_038882.1",    1,   340,
     "5UTR_IRES",          "23396535",
     "Internal ribosome entry site; NS5A condensate seed (Niepmann 2013 Biochim Biophys Acta)"),

    ("HCV",        "NC_038882.1", 9300,  9646,
     "3UTR_SL-X3",         "16332747",
     "3'UTR X-tail (X3 stem-loop), required for replication-condensate formation (Friebe 2005 EMBO J)"),

    # ── RSV (Respiratory Syncytial Virus) ────────────────────────────────────
    # Inclusion bodies form around N-protein/RNA condensates; N gene mRNA
    # is a strong scaffolding RNA.
    ("RSV",        "NC_038235.1",   99,  1120,
     "N_gene_mRNA",        "32404420",
     "Nucleocapsid N gene; scaffolds the cytoplasmic inclusion body (Galloux 2020 Cell Reports)"),

    # ── VSV (Vesicular Stomatitis Virus) ─────────────────────────────────────
    # Negri-body-like inclusion bodies around N gene mRNA.
    ("VSV",        "NC_001560.1",   51,  1072,
     "N_gene_mRNA",        "30013046",
     "Nucleocapsid N gene; cytoplasmic inclusion body component (Heinrich 2018 mBio)"),

    # ── HIV-1 ────────────────────────────────────────────────────────────────
    # 5'UTR + Gag start drives Gag/RNA biomolecular condensate formation; the
    # packaging signal Psi and dimer-initiation site are in this window.
    ("HIV-1",      "K03455.1",       1,  1100,
     "5UTR_Gag_start",     "32709822",
     "TAR + polyA + PBS + Psi packaging signal + Gag start; Gag/RNA condensate driver (Monette 2020 Cell Rep)"),

    # ── Rotavirus A ──────────────────────────────────────────────────────────
    # Viroplasm assembly is driven by NSP2/NSP5 binding viral (+) ssRNA;
    # segments 7 (NSP3) and 8 (NSP2) are well-studied templates.
    ("Rotavirus-A","DQ490542.1",     1,  1020,
     "Segment7_NSP3",      "33510141",
     "Segment 7 (NSP3); viroplasm scaffold component (Geiger 2021 PLoS Pathog)"),

    ("Rotavirus-A","DQ490543.1",     1,  1020,
     "Segment8_NSP2",      "33510141",
     "Segment 8 (NSP2); RNA-binding NTPase driving viroplasm condensate (Geiger 2021 PLoS Pathog)"),

    # ── Reovirus T3 ──────────────────────────────────────────────────────────
    # Viral factories ('viroplasms') assemble around the S2 segment.
    ("Reovirus",   "AF378003.1",     1,  1020,
     "Segment_S2",         "30787197",
     "Segment S2 (sigma-2 capsid protein mRNA); viral factory scaffold (Tenorio 2019 Front Microbiol)"),

    # ── Influenza A ──────────────────────────────────────────────────────────
    # Liquid viral inclusions form around viral RNPs in the cytoplasm.
    ("InfluenzaA", "NC_002020.1",    1,   890,
     "NS_segment_full",    "31092847",
     "Segment 8 NS (NS1+NEP); cytoplasmic viral inclusion body component (Alenquer 2019 Nat Comm)"),

    # ── Ebola virus (Zaire) ───────────────────────────────────────────────────
    # NP-driven inclusion bodies ("viral factories") form on N-protein mRNA.
    ("Ebola",      "NC_002549.1",   470,  1490,
     "NP_gene_mRNA",       "31061160",
     "Nucleoprotein NP gene; cytoplasmic viral inclusion body driver (Nikolic 2017 Nat Comm; Schudt 2015 PNAS)"),

    # ── Marburg virus ────────────────────────────────────────────────────────
    # NP-driven inclusions analogous to Ebola.
    ("Marburg",    "NC_001608.3",   100,  1120,
     "NP_gene_mRNA",       "27581155",
     "Nucleoprotein NP gene; filoviral inclusion body driver (Schudt 2013 J Virol)"),

    # ── Zika virus ───────────────────────────────────────────────────────────
    # 3'UTR forms condensate-relevant stem-loops; XRN1-resistant subgenomic
    # flaviviral RNA (sfRNA) accumulates in stress granules.
    ("Zika",       "NC_012532.1", 10380, 10794,
     "3UTR_sfRNA",         "26973610",
     "3'UTR + dumbbell elements; subgenomic flaviviral RNA scaffold (Bonenfant 2019; Schnettler 2012)"),

    # ── Dengue virus 2 ───────────────────────────────────────────────────────
    ("Dengue-2",   "NC_001474.2", 10270, 10723,
     "3UTR_sfRNA",         "21183745",
     "3'UTR sfRNA (XRN1-resistant); condensate-targeted (Pijlman 2008 Cell Host Microbe; Roth 2017)"),

    # ── West Nile virus ──────────────────────────────────────────────────────
    ("WNV",        "NC_009942.1", 10410, 11029,
     "3UTR_sfRNA",         "21183745",
     "3'UTR sfRNA; flavivirus stress-granule recruiter (Pijlman 2008)"),

    # ── Chikungunya virus ────────────────────────────────────────────────────
    # Stress-granule sequestration by alphavirus genome.
    ("Chikungunya","NC_004162.2",   77,  1090,
     "5UTR_nsP1_start",    "26468530",
     "5'UTR + nsP1 start; alphaviral stress-granule recruitment (Scholte 2015 J Virol)"),

    # ── Hantavirus (Andes) ───────────────────────────────────────────────────
    # P-body/granule sequestration of bunyaviral genome.
    ("Hantavirus", "NC_003466.1",   43,  1063,
     "N_gene_start",       "23720721",
     "Nucleocapsid N gene 5' region; P-body recruitment (Mir 2008)"),
]


def fetch_ncbi_subsequence(accession: str, start: int, end: int) -> str:
    """efetch a sequence subregion from NCBI Nucleotide.

    Uses seq_start/seq_stop parameters so the server returns only the requested
    coordinate window (avoids downloading 30 kb genomes just to clip).
    Returns the sequence as RNA (T->U replaced).
    """
    params = {
        "db":       "nuccore",
        "id":       accession,
        "rettype":  "fasta",
        "retmode":  "text",
        "seq_start": str(start),
        "seq_stop":  str(end),
    }
    try:
        r = requests.get(NCBI_EFETCH, params=params, timeout=TIMEOUT)
        if r.status_code == 200 and r.text.strip().startswith(">"):
            lines = r.text.strip().splitlines()
            seq   = "".join(l for l in lines if not l.startswith(">"))
            return seq.upper().replace("T", "U")
    except Exception as e:
        print(f"  [exception] {accession}:{start}-{end}: {e}")
    return ""


def main(out_dir: str = "Data/raw/multispecies") -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "viral_positives.fasta")

    # Back up the previous viral_positives.fasta if present (likely full genomes)
    if os.path.exists(out_path):
        backup = os.path.join(out_dir, "viral_positives_fullgenome_backup.fasta")
        if not os.path.exists(backup):
            import shutil
            shutil.copyfile(out_path, backup)
            print(f"  backed up existing viral_positives.fasta -> {backup}")

    print(f"Fetching {len(VIRAL_RNA_TARGETS)} curated viral LLPS-region subsequences ...",
          flush=True)
    written = 0
    with open(out_path, "w") as fh:
        for virus, acc, start, end, label, pmid, note in VIRAL_RNA_TARGETS:
            expected_len = end - start + 1
            seq = fetch_ncbi_subsequence(acc, start, end)
            if not seq:
                print(f"  [WARN] failed: {virus} {acc}:{start}-{end}")
                continue
            if len(seq) > 1022:
                # Defensive: clip if the server returned more than expected
                seq = seq[:1022]
            tag = f">llps_virus|{virus}|{acc}:{start}-{end}|{label}|PMID:{pmid}"
            fh.write(f"{tag}\n{seq}\n")
            written += 1
            print(f"  ok: {virus:<14} {acc}:{start}-{end} ({label})  {len(seq)} nt (expected {expected_len})",
                  flush=True)
            time.sleep(RATE_SLEEP)

    print(f"\nWrote {written}/{len(VIRAL_RNA_TARGETS)} viral LLPS-region sequences -> {out_path}")
    print("Each entry carries: virus_short, NCBI_accession:start-end, region_label, PubMed citation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Data/raw/multispecies")
    args = parser.parse_args()
    main(args.out_dir)
