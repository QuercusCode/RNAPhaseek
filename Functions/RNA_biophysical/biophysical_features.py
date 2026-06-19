"""
RNA Biophysical Feature Extractor
===================================
Computes 26 sequence-level features grounded in two 2026 papers:

  RNA2PS (Tejedor et al., bioRxiv 2026)
  ──────────────────────────────────────
  Coarse-grained RNA condensation model with two-bead-per-nucleotide
  representation.  Key physical parameters used here:
    • Wang-Frenkel ε stacking energies (Table IV): A=6.430, U=5.958,
      C=6.531, G=6.033 kcal/mol
    • Base-pair k_bond weights (Table II): AU=1.90, CG=1.15, GU=1.80
    • T_c (critical condensation temperature) correlates with:
        – trinucleotide repeat length and composition (Fig. 4D, 5C)
        – net base-pairing propensity (inter- vs intra-molecular balance)
        – stacking energy density along the chain

  ENCORI (Zhou et al., Nature Methods 2026)
  ──────────────────────────────────────────
  Encyclopedia of RNA interactomes built from >2,675 CLIP-seq datasets.
  Key motifs used here:
    • RBFOX2: UGCAUG  (Fig. 2a, experimentally validated)
    • PUM2:   UGUAAAUA (Fig. 2a, experimentally validated)
    • FUS, TDP-43, hnRNPA1 motifs: known LLPS-driver RBP signatures
    • m⁶A DRACH motif: identified via rbsSeeker/YTHDF1/YTHDF2 analysis

Feature blocks
──────────────
  Block 1 – RNA2PS thermodynamics  (13 features)
  Block 2 – ENCORI RBP motifs      ( 9 features)
  Block 3 – Structural complexity  ( 4 features)
  ─────────────────────────────────────────────
  Total                             26 features

Usage
-----
    from Functions.RNA_biophysical import RNABiophysicalExtractor
    ext = RNABiophysicalExtractor()
    arr = ext.extract(sequences)   # np.ndarray (N, 26)
"""

import math
import re
import numpy as np
from typing import List, Union

# ── Alphabet ──────────────────────────────────────────────────────────────────
RNA_ALPHABET = "AUGC"

def _normalise(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    return re.sub(r"[^AUGC]", "N", seq)


# ══════════════════════════════════════════════════════════════════════════════
# Block 1 — RNA2PS thermodynamic features (13)
# ══════════════════════════════════════════════════════════════════════════════

# Wang-Frenkel ε stacking energy per nucleotide (kcal/mol), Table IV of RNA2PS.
# Higher ε → stronger stacking interaction → more condensation-prone.
RNA2PS_EPSILON = {'A': 6.430, 'U': 5.958, 'C': 6.531, 'G': 6.033, 'N': 6.0}

# Base-pair bond strength k_bp (kcal/mol/Å²), Table II of RNA2PS.
# Weights how strongly each pair type drives inter-strand association.
RNA2PS_KBP = {
    frozenset('AU'): 1.90,   # Watson-Crick A–U
    frozenset('CG'): 1.15,   # Watson-Crick C–G  (note: higher ε but lower k_bp)
    frozenset('GU'): 1.80,   # Wobble G–U
}

# Single-strand stacking ΔG (kcal/mol, 37 °C) — Freier et al. 1986 / Turner 2004.
# These are the thermodynamic values RNA2PS is calibrated to reproduce.
# Format: 5'-XY-3' → ΔG_stack
NN_SINGLE_STRAND: dict = {
    'AA': -0.38, 'AC': -1.69, 'AG': -1.73, 'AU': -0.82,
    'CA': -0.71, 'CC': -1.49, 'CG': -1.79, 'CU': -0.50,
    'GA': -1.36, 'GC': -2.34, 'GG': -2.10, 'GU': -1.39,
    'UA': -0.69, 'UC': -0.95, 'UG': -1.04, 'UU': -0.40,
}


def _rna2ps_features(seq: str) -> np.ndarray:
    """
    Compute 13 RNA2PS-inspired features for a single sequence.

    Features (indices 0-12):
      0  stacking_mean       – mean ε[i]·ε[i+1] across consecutive pairs
      1  stacking_std        – std thereof
      2  condensation_proxy  – mean ε[nt] (overall stacking density)
      3  au_pairing          – freq(AU + UA dinucs) × k_bp_AU
      4  cg_pairing          – freq(CG + GC dinucs) × k_bp_CG
      5  gu_wobble           – freq(GU + UG dinucs) × k_bp_GU
      6  total_pairing       – weighted sum of above three
      7  nn_energy_mean      – mean nearest-neighbour stacking ΔG
      8  nn_energy_std       – std thereof
      9  repeat_entropy_inv  – 1 – H(trinucleotides)/log2(64)  (0=random, 1=single repeat)
      10 longest_trinuc_rep  – longest trinucleotide repeat unit / len(seq)
      11 longest_mono_run    – longest mononucleotide run / len(seq)
      12 au_rich_score       – (A+U) / (A+U+G+C)  (high AU → flexible ssRNA scaffold)
    """
    n = len(seq)
    if n == 0:
        return np.zeros(13, dtype=np.float32)

    eps = np.array([RNA2PS_EPSILON.get(c, 6.0) for c in seq], dtype=np.float64)

    # 0-1  stacking energy profile
    if n > 1:
        stack_prod = eps[:-1] * eps[1:]            # ε_i × ε_{i+1}
        stack_mean = float(stack_prod.mean())
        stack_std  = float(stack_prod.std())
    else:
        stack_mean = stack_std = float(eps[0] ** 2)

    # 2  condensation proxy
    cond_proxy = float(eps.mean())

    # 3-6  dinucleotide pairing propensity
    au = cg = gu = total_di = 0
    nn_vals = []
    for i in range(n - 1):
        di = seq[i] + seq[i + 1]
        if di[0] in "AUGC" and di[1] in "AUGC":
            total_di += 1
            pair = frozenset(di)
            au += pair == frozenset('AU')
            cg += pair == frozenset('CG')
            gu += pair == frozenset('GU')
            if di in NN_SINGLE_STRAND:
                nn_vals.append(NN_SINGLE_STRAND[di])

    denom = max(total_di, 1)
    au_prop = (au / denom) * RNA2PS_KBP[frozenset('AU')]
    cg_prop = (cg / denom) * RNA2PS_KBP[frozenset('CG')]
    gu_prop = (gu / denom) * RNA2PS_KBP[frozenset('GU')]
    tot_prop = au_prop + cg_prop + gu_prop

    # 7-8  NN stacking ΔG statistics
    if nn_vals:
        nn_arr  = np.array(nn_vals)
        nn_mean = float(nn_arr.mean())
        nn_std  = float(nn_arr.std())
    else:
        nn_mean = nn_std = 0.0

    # 9  inverse trinucleotide Shannon entropy (repetitiveness)
    if n >= 3:
        counts = {}
        for i in range(n - 2):
            tri = seq[i:i+3]
            if all(c in "AUGCN" for c in tri):
                counts[tri] = counts.get(tri, 0) + 1
        total_tri = sum(counts.values())
        if total_tri > 0:
            probs = np.array(list(counts.values()), dtype=float) / total_tri
            H = -float(np.sum(probs * np.log2(probs + 1e-12)))
            H_max = math.log2(min(64, total_tri)) if total_tri > 1 else 1.0
            repeat_entropy_inv = 1.0 - H / max(H_max, 1.0)
        else:
            repeat_entropy_inv = 0.0
    else:
        repeat_entropy_inv = 0.0

    # 10  longest trinucleotide tandem repeat / len
    longest_trinuc = 0.0
    if n >= 6:
        # Scan all possible 3-mers as repeat units
        for unit_start in range(n - 2):
            unit = seq[unit_start:unit_start+3]
            if not all(c in "AUGC" for c in unit):
                continue
            count = 1
            pos = unit_start + 3
            while pos + 3 <= n and seq[pos:pos+3] == unit:
                count += 1
                pos += 3
            repeat_len = count * 3
            if repeat_len > longest_trinuc:
                longest_trinuc = repeat_len
    longest_trinuc_norm = longest_trinuc / n

    # 11  longest mononucleotide run / len
    longest_mono = 1
    current_run = 1
    for i in range(1, n):
        if seq[i] == seq[i-1] and seq[i] in "AUGC":
            current_run += 1
            if current_run > longest_mono:
                longest_mono = current_run
        else:
            current_run = 1
    longest_mono_norm = longest_mono / n

    # 12  AU-rich score
    na = seq.count('A')
    nu = seq.count('U')
    ng = seq.count('G')
    nc = seq.count('C')
    au_rich = (na + nu) / max(na + nu + ng + nc, 1)

    return np.array([
        stack_mean, stack_std, cond_proxy,
        au_prop, cg_prop, gu_prop, tot_prop,
        nn_mean, nn_std,
        repeat_entropy_inv, longest_trinuc_norm, longest_mono_norm, au_rich,
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Block 2 — ENCORI RBP motif features (9)
# ══════════════════════════════════════════════════════════════════════════════

# Motifs are grounded directly in ENCORI paper:
#   RBFOX2 = UGCAUG   (Fig. 2a — highest motif frequency, experimentally validated)
#   PUM2   = UGUAAAUA (Fig. 2a — experimentally validated)
# LLPS-driver RBP motifs (consensus from CLIP-seq literature):
#   FUS/TLS   = GGUG, GUGGU  (Rogelj et al., Nat Neurosci 2012)
#   TDP-43    = UGUG, GUGU   (Polymenidou et al., Nat Neurosci 2011)
#   hnRNPA1   = UAGG, UAGGG  (Liu & Dreyfuss 1995; Bekenstein 2005)
# m⁶A DRACH: detected by rbsSeeker on YTHDF1/YTHDF2 CLIP-seq (ENCORI Fig. 2c-d)
#   D=[AGU], R=[AG], A=A, C=C, H=[ACU]
_DRACH = re.compile(r'[AGU][AG]AC[ACU]')

_RBP_PATTERNS = {
    'fus':     [re.compile(p) for p in [r'GGUG', r'GUGGU']],
    'tdp43':   [re.compile(p) for p in [r'UGUG', r'GUGU']],
    'hnrnpa1': [re.compile(p) for p in [r'UAGG', r'UAGGG']],
    'rbfox2':  [re.compile(r'UGCAUG')],          # ENCORI Fig. 2a
    'pum2':    [re.compile(r'UGUAAAUA')],         # ENCORI Fig. 2a
    'are':     [re.compile(p) for p in [r'AUUUA', r'AUUUUA', r'UAUUU']],
    'm6a':     [_DRACH],
}


def _count_overlapping(pattern: re.Pattern, seq: str) -> int:
    """Count overlapping occurrences of a compiled regex pattern."""
    count, pos = 0, 0
    while True:
        m = pattern.search(seq, pos)
        if not m:
            break
        count += 1
        pos = m.start() + 1
    return count


def _encori_features(seq: str) -> np.ndarray:
    """
    Compute 9 ENCORI-inspired RBP motif features.

    Features (indices 13-21):
      13 fus_density       – FUS motif occurrences / len
      14 tdp43_density     – TDP-43 motif occurrences / len
      15 hnrnpa1_density   – hnRNPA1 motif occurrences / len
      16 rbfox2_density    – RBFOX2 UGCAUG density (ENCORI Fig. 2a)
      17 pum2_density      – PUM2 UGUAAAUA density (ENCORI Fig. 2a)
      18 are_density       – AU-rich element (AUUUA) density
      19 m6a_density       – DRACH m⁶A motif density
      20 llps_rbp_composite– FUS + TDP43 + hnRNPA1 combined density
      21 rbp_total         – sum of all individual motif densities
    """
    n = max(len(seq), 1)
    densities = {}
    for name, patterns in _RBP_PATTERNS.items():
        cnt = sum(_count_overlapping(p, seq) for p in patterns)
        densities[name] = cnt / n

    fus    = densities['fus']
    tdp43  = densities['tdp43']
    hnrnpa1= densities['hnrnpa1']
    rbfox2 = densities['rbfox2']
    pum2   = densities['pum2']
    are    = densities['are']
    m6a    = densities['m6a']
    llps_composite = fus + tdp43 + hnrnpa1
    rbp_total = fus + tdp43 + hnrnpa1 + rbfox2 + pum2 + are + m6a

    return np.array([
        fus, tdp43, hnrnpa1, rbfox2, pum2, are, m6a,
        llps_composite, rbp_total,
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Block 3 — Structural complexity features (4)
# ══════════════════════════════════════════════════════════════════════════════

def _complexity_features(seq: str) -> np.ndarray:
    """
    Compute 4 structural complexity features.

    Features (indices 22-25):
      22 lc_density      – low-complexity fraction (mono-nt runs ≥ 4)
      23 purine_fraction – (A+G) / len  (purines stack better)
      24 dinuc_entropy   – Shannon entropy of 16 dinucleotides (bits, 0-4)
      25 seq_complexity  – linguistic complexity: unique_kmers(k=3) / min(4³, len-2)
    """
    n = len(seq)
    if n == 0:
        return np.zeros(4, dtype=np.float32)

    # 22  low-complexity density
    lc_count = 0
    run_len, run_char = 1, seq[0]
    for c in seq[1:]:
        if c == run_char:
            run_len += 1
        else:
            if run_len >= 4:
                lc_count += run_len
            run_len, run_char = 1, c
    if run_len >= 4:
        lc_count += run_len
    lc_density = lc_count / n

    # 23  purine fraction
    pur = (seq.count('A') + seq.count('G')) / n

    # 24  dinucleotide Shannon entropy
    di_counts = {}
    for i in range(n - 1):
        di = seq[i:i+2]
        if all(c in "AUGC" for c in di):
            di_counts[di] = di_counts.get(di, 0) + 1
    total_di = sum(di_counts.values())
    if total_di > 0:
        probs = np.array(list(di_counts.values()), dtype=float) / total_di
        dinuc_H = -float(np.sum(probs * np.log2(probs + 1e-12)))
    else:
        dinuc_H = 0.0

    # 25  linguistic complexity (unique 3-mers / possible)
    if n >= 3:
        kmers = set(seq[i:i+3] for i in range(n - 2)
                    if all(c in "AUGC" for c in seq[i:i+3]))
        max_possible = min(64, n - 2)
        ling_complex = len(kmers) / max(max_possible, 1)
    else:
        ling_complex = 0.0

    return np.array([lc_density, pur, dinuc_H, ling_complex], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Block 4 — Absolute repeat & periodicity features (7)
# ══════════════════════════════════════════════════════════════════════════════

def _repeat_periodicity_features(seq: str) -> np.ndarray:
    """
    Compute 7 ABSOLUTE (NOT length-normalized) repeat & periodicity features.

    Motivation: the length-normalized repeat features in Block 1
    (longest_trinuc_repeat / len) give (CAG)20 and (CAG)31 the SAME value,
    so a classifier built on them cannot resolve the phase-separation repeat
    threshold. These absolute features encode the repeat COPY NUMBER and the
    PERIODICITY STRENGTH directly, which a diagnostic showed the model lacked
    (it could not separate sub-threshold repeats, nor ordered-vs-shuffled).

    Features (indices 26-32):
      26 abs_longest_mono_run   – longest single-base run (nt)
      27 abs_longest_tandem     – longest tandem repeat of period 1-6 (nt)
      28 max_tandem_copies      – max contiguous copies of any period-1-6 motif
      29 dominant_period        – period (1-6) of the strongest tandem (0 if none)
      30 periodicity_strength   – frac. of positions with seq[i]==seq[i+p] at the
                                  dominant period p (ordered repeats→1, shuffles→~1/k)
      31 tri_repeat_copies      – copies of the longest trinucleotide tandem (CAG/CUG)
      32 hexa_repeat_copies     – copies of the longest hexanucleotide tandem (G4C2)
    """
    n = len(seq)
    if n < 2:
        return np.zeros(7, dtype=np.float32)

    # Longest mononucleotide run
    longest_mono, run = 1, 1
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            run += 1
            if run > longest_mono:
                longest_mono = run
        else:
            run = 1

    # For each period p in 1..6: longest tandem repeat (consecutive identical
    # p-mers) and its copy count.
    def longest_tandem_for_period(p: int):
        if n < 2 * p:
            return 0, 1
        best_len, best_copies = 0, 1
        i = 0
        while i + p <= n:
            unit = seq[i:i + p]
            copies, j = 1, i + p
            while j + p <= n and seq[j:j + p] == unit:
                copies += 1
                j += p
            if copies >= 2 and copies * p > best_len:
                best_len, best_copies = copies * p, copies
            i = j if copies >= 2 else i + 1
        return best_len, best_copies

    per = {p: longest_tandem_for_period(p) for p in range(1, 7)}
    dom_p = max(range(1, 7), key=lambda p: per[p][0])
    dom_len, _ = per[dom_p]
    abs_longest_tandem = dom_len
    max_tandem_copies  = max(per[p][1] for p in range(1, 7))
    dominant_period    = dom_p if dom_len >= 2 * dom_p else 0
    tri_copies         = per[3][1] if per[3][0] >= 6 else 0
    hexa_copies        = per[6][1] if per[6][0] >= 12 else 0

    # Periodicity strength at the dominant period (composition-robust separation
    # of ordered repeats from shuffled controls of identical composition).
    if dominant_period > 0 and n > dominant_period:
        p = dominant_period
        matches = sum(1 for i in range(n - p) if seq[i] == seq[i + p])
        periodicity_strength = matches / (n - p)
    else:
        periodicity_strength = 0.0

    return np.array([
        float(longest_mono),
        float(abs_longest_tandem),
        float(max_tandem_copies),
        float(dominant_period),
        float(periodicity_strength),
        float(tri_copies),
        float(hexa_copies),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Block 5 — Self-complementarity / base-pairing (5)   indices 33-37
# ══════════════════════════════════════════════════════════════════════════════
# These measure REVERSE-COMPLEMENT structure (seq[i] pairs seq[j] across a loop),
# which Blocks 1-4 miss: au/cg/gu_pairing are adjacent-dinuc composition stats and
# periodicity_strength fires on DIRECT repeats (seq[i]==seq[i+p]), not on palindromes.
# A composition-matched scramble of a self-complementary RNA leaves Blocks 1-4 ~fixed
# but collapses these features — the discrimination v3 lacked (Fabrini A vs A-bar).
_COMP = {"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"}
_N_ORD = ord("N")


def _revcomp(seq: str) -> str:
    return "".join(_COMP.get(c, "N") for c in reversed(seq))


def _rc_selfalign_raw(seq: str, cap: int = 300) -> float:
    """Local (Smith-Waterman) alignment score of seq vs its reverse complement
    (WC match +1, mismatch -1, gap -2); RAW best score. Measures the best gapped
    self-complementary stem. O(n^2) — capped to a centered window of `cap` nt."""
    n = len(seq)
    if n == 0:
        return 0.0
    if n > cap:
        st = (n - cap) // 2
        seq = seq[st:st + cap]
    a = seq
    b = _revcomp(seq)
    m = len(a)
    bcodes = np.frombuffer(b.encode("ascii"), dtype=np.uint8)
    zeros = np.zeros(m, dtype=np.float64)
    prev = np.zeros(m + 1, dtype=np.float64)
    best = 0.0
    for i in range(1, m + 1):
        ai = ord(a[i - 1])
        if ai == _N_ORD:
            match = np.full(m, -1.0)
        else:
            match = np.where(bcodes == ai, 1.0, -1.0)
            match[bcodes == _N_ORD] = -1.0          # N never pairs
        diag = prev[:-1] + match                    # j = 1..m
        up = prev[1:] - 2.0
        tent = np.maximum(np.maximum(zeros, diag), up)
        row = np.empty(m + 1, dtype=np.float64)
        row[0] = 0.0
        c = 0.0
        for j in range(1, m + 1):                   # resolve left-gap dependency
            v = tent[j - 1]
            l = c - 2.0
            if l > v:
                v = l
            if v < 0.0:
                v = 0.0
            row[j] = v
            c = v
            if v > best:
                best = v
        prev = row
    return float(best)


def _rc_kmer_selfpair(seq: str, k: int = 6) -> float:
    """Fraction of distinct k-mers whose reverse complement also occurs in the
    sequence — a global intramolecular-pairing-potential measure. O(n)."""
    n = len(seq)
    if n < k:
        return 0.0
    kmers = {seq[i:i + k] for i in range(n - k + 1)}
    hits = sum(1 for km in kmers if "N" not in km and _revcomp(km) in kmers)
    return hits / max(len(kmers), 1)


def _longest_rc_stem(seq: str, min_loop: int = 3, max_stem: int = 30) -> int:
    """Longest perfect reverse-complement hairpin arm (bp) separated by a
    >= min_loop gap. Fires on RC symmetry, unlike Block-4 direct-repeat periodicity."""
    n = len(seq)
    best = 0
    for i in range(n):
        for j in range(i + min_loop + 1, min(n, i + 1 + min_loop + 2 * max_stem)):
            L = 0
            while (i - L >= 0 and j + L < n and L < max_stem
                   and seq[i - L] != "N" and _COMP.get(seq[i - L]) == seq[j + L]):
                L += 1
            if L > best:
                best = L
    return best


# Block-5 work window. RNA-FM itself only sees 1022-nt contexts and ViennaRNA's
# O(n^3) fold is intractable on multi-kb lncRNAs, so ALL self-complementarity
# features are computed over one centered window of <= STRUCT_WIN nt. Using a single
# shared window keeps the five sub-features mutually consistent in scale.
STRUCT_WIN = 800


def _structure_features(seq: str) -> np.ndarray:
    """5 self-complementarity / base-pairing features (indices 33-37), computed over a
    centered <= STRUCT_WIN-nt window:
      33 rc_selfalign_score          – local-align score vs reverse complement / win_len
      34 mfe_paired_fraction         – ViennaRNA MFE paired fraction (fallback: rc_kmer)
      35 mfe_energy_per_nt           – ViennaRNA MFE kcal/mol per nt (fallback: -align/win_len)
      36 longest_rc_palindrome_stem  – longest RC hairpin arm (bp) / win_len
      37 rc_kmer_selfpair_frac       – frac of 6-mers whose RC also occurs
    ViennaRNA is used when importable; otherwise indices 34/35 degrade to sequence-only
    surrogates (same scale, sign and monotonicity) so the block never raises or returns NaN."""
    n = len(seq)
    if n < 4:
        return np.zeros(5, dtype=np.float32)

    if n > STRUCT_WIN:
        st = (n - STRUCT_WIN) // 2
        seq = seq[st:st + STRUCT_WIN]
    w = len(seq)

    align_raw = _rc_selfalign_raw(seq)
    rc_align = align_raw / w
    rc_kmer = _rc_kmer_selfpair(seq, 6)
    rc_stem = _longest_rc_stem(seq) / w

    paired_frac = rc_kmer            # sequence-only fallback default
    energy_pn = -align_raw / w       # sequence-only fallback default (neg, scales w/ stem)
    try:
        import RNA
        fc = RNA.fold_compound(seq)
        struct, mfe = fc.mfe()
        paired_frac = (struct.count("(") + struct.count(")")) / w
        energy_pn = mfe / w
    except Exception:
        pass                         # keep sequence-only fallbacks; never raises

    return np.array([
        float(rc_align),
        float(paired_frac),
        float(energy_pn),
        float(rc_stem),
        float(rc_kmer),
    ], dtype=np.float32)


def _pf_features(seq: str) -> np.ndarray:
    """Block 6 — partition-function (ensemble) structure (indices 38-41), over the centered
    <= STRUCT_WIN window. Unlike the single MFE fold (block 5), these reflect the whole
    Boltzmann ensemble:
      38 pf_energy_per_nt        – ensemble free energy / nt (more neg = more structured)
      39 mean_bpp                – mean expected base pairs / nt (ensemble pairing)
      40 ensemble_diversity      – mean base-pair distance / nt (competing structures)
      41 positional_pair_entropy – mean binary entropy of per-base pairing prob (flexibility)"""
    n = len(seq)
    if n < 4:
        return np.zeros(4, dtype=np.float32)
    if n > STRUCT_WIN:
        st = (n - STRUCT_WIN) // 2
        seq = seq[st:st + STRUCT_WIN]
    w = len(seq)
    out = np.zeros(4, dtype=np.float32)
    try:
        import RNA
        fc = RNA.fold_compound(seq)
        _, pf_energy = fc.pf()
        B = np.array(fc.bpp(), dtype=np.float64)          # (w+1, w+1) upper-tri, 1-indexed
        pp = (B.sum(axis=0) + B.sum(axis=1))[1:w + 1]      # per-base total pairing prob
        pp = np.clip(pp, 0.0, 1.0)
        m = (pp > 1e-9) & (pp < 1 - 1e-9)
        ent = np.zeros_like(pp)
        ent[m] = -(pp[m] * np.log(pp[m]) + (1 - pp[m]) * np.log(1 - pp[m]))
        out = np.array([pf_energy / w, pp.sum() / (2 * w),
                        fc.mean_bp_distance() / w, float(ent.mean())], dtype=np.float32)
    except Exception:
        pass
    # sanitize ViennaRNA edge cases (rare short/degenerate seqs give non-physical |E/nt|): -> neutral 0
    return np.where(np.isfinite(out) & (np.abs(out) < 50), out, 0.0).astype(np.float32)


def _intermolecular_features(seq: str) -> np.ndarray:
    """Block 7 — intermolecular multivalency (indices 42-44): the RNA-RNA pairing that drives
    self-LLPS but that intramolecular folding (blocks 5-6) misses — the kissing-loop blind spot.
    Over the centered <= STRUCT_WIN window:
      42 selfduplex_energy_per_nt       – best self-self intermolecular duplex MFE / nt
      43 selfduplex_bp_frac             – intermolecular base pairs in that duplex / nt
      44 interhalf_duplex_energy_per_nt – best duplex between the two halves / nt (distinct-region pairing)"""
    n = len(seq)
    if n < 8:
        return np.zeros(3, dtype=np.float32)
    if n > STRUCT_WIN:
        st = (n - STRUCT_WIN) // 2
        seq = seq[st:st + STRUCT_WIN]
    w = len(seq)
    out = np.zeros(3, dtype=np.float32)
    try:
        import RNA
        d = RNA.duplexfold(seq, seq)
        h = w // 2
        d2 = RNA.duplexfold(seq[:h], seq[h:])
        out = np.array([d.energy / w, d.structure.count("(") / w, d2.energy / max(h, 1)],
                       dtype=np.float32)
    except Exception:
        pass
    # sanitize ViennaRNA edge cases (rare short/degenerate seqs give non-physical |E/nt|): -> neutral 0
    return np.where(np.isfinite(out) & (np.abs(out) < 50), out, 0.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Feature names and dimensions
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES: List[str] = [
    # Block 1 — RNA2PS (13)
    "rna2ps_stacking_mean",
    "rna2ps_stacking_std",
    "rna2ps_condensation_proxy",
    "rna2ps_au_pairing",
    "rna2ps_cg_pairing",
    "rna2ps_gu_wobble",
    "rna2ps_total_pairing",
    "nn_energy_mean",
    "nn_energy_std",
    "repeat_entropy_inv",
    "longest_trinuc_repeat",
    "longest_mono_run",
    "au_rich_score",
    # Block 2 — ENCORI (9)
    "fus_motif_density",
    "tdp43_motif_density",
    "hnrnpa1_motif_density",
    "rbfox2_motif_density",       # ENCORI Fig. 2a
    "pum2_motif_density",         # ENCORI Fig. 2a
    "are_density",
    "m6a_drach_density",
    "llps_rbp_composite",
    "rbp_total_density",
    # Block 3 — Complexity (4)
    "lc_density",
    "purine_fraction",
    "dinuc_entropy",
    "seq_linguistic_complexity",
    # Block 4 — Absolute repeat & periodicity (7)
    "abs_longest_mono_run",
    "abs_longest_tandem",
    "max_tandem_copies",
    "dominant_period",
    "periodicity_strength",
    "tri_repeat_copies",
    "hexa_repeat_copies",
    # Block 5 — Self-complementarity / base-pairing (5)
    "rc_selfalign_score",
    "mfe_paired_fraction",
    "mfe_energy_per_nt",
    "longest_rc_palindrome_stem",
    "rc_kmer_selfpair_frac",
]

N_FEATURES: int = len(FEATURE_NAMES)  # 38

# Extended set (opt-in via RNABiophysicalExtractor(extended=True)) — adds Block 6 (partition-
# function ensemble) + Block 7 (intermolecular multivalency). Keeps the 38-dim default intact
# so existing v6/v7 models are unaffected.
EXTENDED_FEATURE_NAMES: List[str] = FEATURE_NAMES + [
    "pf_energy_per_nt", "mean_bpp", "ensemble_diversity", "positional_pair_entropy",       # Block 6 (4)
    "selfduplex_energy_per_nt", "selfduplex_bp_frac", "interhalf_duplex_energy_per_nt",     # Block 7 (3)
]
N_FEATURES_EXT: int = len(EXTENDED_FEATURE_NAMES)  # 45


# ══════════════════════════════════════════════════════════════════════════════
# Main extractor class
# ══════════════════════════════════════════════════════════════════════════════

class RNABiophysicalExtractor:
    """
    Extract 26 biophysical features per RNA sequence.

    Parameters
    ----------
    normalize : bool
        If True, apply per-feature z-score normalization using statistics
        fit on the provided sequences (fit_transform mode) or pre-fitted
        mean/std arrays.
    """

    def __init__(self, normalize: bool = False, extended: bool = False):
        self.normalize = normalize
        self.extended = extended       # True -> 45-dim (Block 6+7); False -> 38-dim (v6-compatible)
        self._mean: np.ndarray = None
        self._std:  np.ndarray = None

    def _compute_one(self, seq: str) -> np.ndarray:
        seq = _normalise(seq)
        blocks = [
            _rna2ps_features(seq),
            _encori_features(seq),
            _complexity_features(seq),
            _repeat_periodicity_features(seq),
            _structure_features(seq),   # indices 33-37
        ]
        if self.extended:
            blocks += [_pf_features(seq),             # indices 38-41
                       _intermolecular_features(seq)]  # indices 42-44
        return np.concatenate(blocks)  # (38,) default, (45,) extended

    def extract(
        self,
        sequences: Union[List[str], np.ndarray],
    ) -> np.ndarray:
        """
        Extract features for a list of RNA sequences.

        Returns
        -------
        np.ndarray of shape (N, 26), dtype float32
        """
        feats = np.stack([self._compute_one(str(s)) for s in sequences],
                         axis=0).astype(np.float32)

        if self.normalize:
            if self._mean is None:
                self._mean = feats.mean(axis=0)
                self._std  = feats.std(axis=0).clip(min=1e-8)
            feats = (feats - self._mean) / self._std

        return feats

    def fit_transform(self, sequences: Union[List[str], np.ndarray]) -> np.ndarray:
        """Fit normalization stats and return normalized features."""
        self.normalize = True
        self._mean = None   # reset so extract() refits
        return self.extract(sequences)

    def transform(self, sequences: Union[List[str], np.ndarray]) -> np.ndarray:
        """Apply pre-fitted normalization (call fit_transform first)."""
        if self._mean is None:
            raise RuntimeError("Call fit_transform() before transform().")
        self.normalize = True
        return self.extract(sequences)

    def save_stats(self, path: str) -> None:
        """Save normalization mean/std to a .npz file."""
        if self._mean is None:
            raise RuntimeError("No stats to save — call fit_transform() first.")
        np.savez(path, mean=self._mean, std=self._std)

    def load_stats(self, path: str) -> None:
        """Load normalization stats from a previously saved .npz file."""
        z = np.load(path)
        self._mean = z['mean']
        self._std  = z['std']
        self.normalize = True
