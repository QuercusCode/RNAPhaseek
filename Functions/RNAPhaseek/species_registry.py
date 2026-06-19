"""
Species registry for the multi-species RNAPhaseek classifier.

Maps every supported species label to a fixed integer ID consumed by the
nn.Embedding in RNAFMHybridClassifier.species_emb.

The IDs are STABLE — never reuse a removed ID; always extend at the end.
This preserves checkpoint compatibility.
"""

import re as _re

# Stable species -> integer ID assignment. Extend at the END.
SPECIES_TO_ID = {
    "human":       0,
    "mouse":       1,
    "yeast":       2,
    "celegans":    3,
    "drosophila":  4,
    "arabidopsis": 5,
    "virus":       6,
    "unknown":     7,
    # Add new species below, never reuse a number:
    "zebrafish":   8,
    "xenopus":     9,
    "rat":        10,
    "rice":       11,
    "spombe":     12,    # Schizosaccharomyces pombe (fission yeast)
    "trypanosoma": 13,   # Trypanosoma brucei (and related kinetoplastids)
}
ID_TO_SPECIES = {v: k for k, v in SPECIES_TO_ID.items()}

N_SPECIES = max(SPECIES_TO_ID.values()) + 1

# Ensembl ID prefixes (and related model-organism database accessions)
# observed in raw cDNA FASTA headers. Mapped to canonical species labels.
# Matched via _ENSEMBL_RE below (prefix followed by ≥5 digits to disambiguate
# e.g. ENSMUST from ENST).
_ENSEMBL_ID_PREFIXES = {
    "ensmust":  "mouse",     "ensmusg":  "mouse",     "ensmusp":  "mouse",
    "enst":     "human",     "ensg":     "human",     "ensp":     "human",
    "ensdart":  "zebrafish", "ensdarg":  "zebrafish", "ensdarp":  "zebrafish",
    "ensrnot":  "rat",       "ensrnog":  "rat",       "ensrnop":  "rat",
    "ensxetg":  "xenopus",   "ensxett":  "xenopus",
}
# Match each prefix followed by at least one digit, with letter-only lookahead
# rejection. The longest matching prefix wins (so ENSMUST beats ENST).
_ENSEMBL_RE = _re.compile(
    r"\b(?P<prefix>"
    r"ensmust|ensmusg|ensmusp"
    r"|ensdart|ensdarg|ensdarp"
    r"|ensrnot|ensrnog|ensrnop"
    r"|ensxetg|ensxett"
    r"|enst|ensg|ensp"
    r")\d{5,}"
)
# Model-organism DB accessions (MGI:99956, RGD:1306081)
_MOD_DB_RE = _re.compile(r"\b(?P<db>mgi|rgd|wbgene|fbgn|tair):?\d+")
_MOD_DB_TO_SPECIES = {"mgi": "mouse", "rgd": "rat",
                       "wbgene": "celegans", "fbgn": "drosophila",
                       "tair": "arabidopsis"}

# Virus detection patterns. Treat underscore as a token boundary by using
# lookbehind/lookahead against [a-z] (so "Rotavirus_A" matches because '_'
# is not a letter). Short tokens (hiv, hcv, vsv, rsv, sars) reject substring
# false-positives from human gene names (HIVEP1, RSVP1, etc.).
_VIRUS_RE = _re.compile(
    r"(?<![a-z])sars(?![a-z])"          # SARS, SARS-CoV-2; reject 'sarsap...'
    r"|(?<![a-z])hcv(?![a-z])"
    r"|(?<![a-z])hiv(?![a-z])"          # HIV, HIV-1, HIV1; REJECT HIVEP1
    r"|(?<![a-z])vsv(?![a-z])"
    r"|(?<![a-z])rsv(?![a-z])"
    r"|(?<![a-z])(?:rotavirus|reovirus|orthoreovirus|coronavirus|papillomavirus)(?![a-z])"
    r"|(?<![a-z])(?:influenza|measles|rabies|nipah)(?![a-z])"
    r"|(?<![a-z])viral(?![a-z])"
    r"|(?<![a-z])virus(?![a-z])"        # standalone 'virus' inc. virus_RNA
)


def species_id_for(label_or_header: str) -> int:
    """
    Heuristic mapper: given any FASTA header or species label string, return
    the corresponding integer species ID. Falls back to 'unknown' (7).

    Accepts:
      - canonical labels:           "human", "mouse", "yeast", ...
      - LLPS pipe-headers:          "llps_yeast|...", "neg_mouse|..."
      - Latin names:                "Homo_sapiens", "Mus musculus"
      - Source tags (smOOPs, etc.)  best-effort
    """
    if not label_or_header:
        return SPECIES_TO_ID["unknown"]
    s = str(label_or_header).lower()

    # Exact matches
    if s in SPECIES_TO_ID:
        return SPECIES_TO_ID[s]

    # Pipe-prefixed header tags
    if s.startswith(("llps_", "neg_", "pos_")):
        tag = s.split("|", 1)[0].split("_", 1)[1]
        if tag in SPECIES_TO_ID:
            return SPECIES_TO_ID[tag]

    # Ensembl ID prefixes in raw cDNA headers (must run BEFORE the Latin-name
    # heuristics: an ENSMUST/ENSMUSG accession is the most reliable signal).
    # Uses regex with \d{5,} suffix so ENSMUST is not confused with ENST.
    m = _ENSEMBL_RE.search(s)
    if m:
        sp = _ENSEMBL_ID_PREFIXES.get(m.group("prefix"))
        if sp and sp in SPECIES_TO_ID:
            return SPECIES_TO_ID[sp]

    # Model-organism DB accessions (MGI:99956, RGD:1306081, WBGene00012345, ...)
    m = _MOD_DB_RE.search(s)
    if m:
        sp = _MOD_DB_TO_SPECIES.get(m.group("db"))
        if sp and sp in SPECIES_TO_ID:
            return SPECIES_TO_ID[sp]

    # Latin names and lab-source tags
    if "sapiens" in s or "hek293" in s or "u2os" in s or "_human" in s: return SPECIES_TO_ID["human"]
    if "musculus" in s or ("smoops" in s and "mouse" in s): return SPECIES_TO_ID["mouse"]
    if "cerevisiae" in s or "sgd" in s or "_yeast" in s: return SPECIES_TO_ID["yeast"]
    if "elegans" in s or "wbgene" in s or "_worm" in s: return SPECIES_TO_ID["celegans"]
    if "melanogaster" in s or "fbgn" in s or "_fly" in s: return SPECIES_TO_ID["drosophila"]
    if "arabidopsis" in s or "tair" in s or "_plant" in s: return SPECIES_TO_ID["arabidopsis"]
    if "rerio" in s or "zebrafish" in s: return SPECIES_TO_ID["zebrafish"]
    if "xenopus" in s or "laevis" in s: return SPECIES_TO_ID["xenopus"]
    if "norvegicus" in s or "rattus" in s: return SPECIES_TO_ID["rat"]
    if "oryza" in s or "_rice" in s: return SPECIES_TO_ID["rice"]
    if "pombe" in s or "spbc" in s or "spac" in s or "_spombe" in s: return SPECIES_TO_ID["spombe"]
    if "trypanosoma" in s or "tbrucei" in s or "tb927" in s or "_trypanosoma" in s: return SPECIES_TO_ID["trypanosoma"]
    if _VIRUS_RE.search(s):
        return SPECIES_TO_ID["virus"]

    # Source tag in the second pipe-delimited field (e.g. >RPS2|RPS_2|AHNAK|...|Homo_sapiens)
    if "|" in s:
        parts = s.split("|")
        for p in parts:
            p = p.strip()
            if p in SPECIES_TO_ID:
                return SPECIES_TO_ID[p]

    return SPECIES_TO_ID["unknown"]


def label_for(species_id: int) -> str:
    return ID_TO_SPECIES.get(int(species_id), "unknown")
