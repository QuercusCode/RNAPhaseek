"""
RNA-FEGS Feature Extraction
============================
Adapted from Phaseek's FEGS (Feature Extraction from Graph Statistics)
for protein sequences. All amino-acid-specific logic is replaced with
RNA nucleotide logic.

Three feature blocks are concatenated:
  EL  – Maximal Eigenvalue per RNA motif group  (graph-theoretic walk feature)
  FC  – Nucleotide Composition                  (4 values: A, U, G, C)
  FD  – Dinucleotide Composition                (16 values, flattened)
  FT  – Trinucleotide Composition               (64 values, flattened)  [optional]
  FG4 – G-Quadruplex density features           (window-based G4 motif count)

Usage
-----
from Functions.RNA_FEGS.RNA_FEGS_feature_extraction import RNAFEGSFeatureExtractor
extractor = RNAFEGSFeatureExtractor()
features  = extractor.extract(sequences)   # list of RNA strings or FASTA path
"""

import os
import numpy as np
import re
from multiprocessing import Pool
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs
from Bio import SeqIO


# ── Alphabet ─────────────────────────────────────────────────────────────────
RNA_ALPHABET = "AUGC"           # canonical RNA bases
RNA_N        = len(RNA_ALPHABET)  # 4

# Degenerate nucleotide expansion (for sequences that still contain DNA T)
_DEGENERATE = {"T": "U", "t": "u"}

def _normalise_rna(seq: str) -> str:
    """Convert DNA to RNA and uppercase; keep only AUGC, replace anything else with N."""
    seq = seq.upper()
    seq = seq.replace("T", "U")
    return re.sub(r"[^AUGC]", "N", seq)


# ── G-Quadruplex motif detection ─────────────────────────────────────────────
# Canonical G4 pattern: G{x>=2} [loop 1-7] G{x>=2} [loop 1-7] G{x>=2} [loop 1-7] G{x>=2}
_G4_PATTERN = re.compile(
    r"(G{2,}[AUGCN]{1,7}){3}G{2,}", re.IGNORECASE
)

def count_g4_motifs(seq: str) -> int:
    """Count overlapping G4 motif occurrences in a sequence."""
    count = 0
    start = 0
    while True:
        m = _G4_PATTERN.search(seq, start)
        if not m:
            break
        count += 1
        start = m.start() + 1
    return count


def g4_density_windows(seq: str, window: int = 50, step: int = 25) -> float:
    """Average G4 motif count across sliding windows (normalised by window size)."""
    if len(seq) < window:
        return count_g4_motifs(seq) / max(len(seq), 1)
    counts = []
    for i in range(0, len(seq) - window + 1, step):
        counts.append(count_g4_motifs(seq[i : i + window]))
    return float(np.mean(counts)) / window if counts else 0.0


# ── GRS walk (Graph Representation of Sequence) ──────────────────────────────
# Place 4 nucleotides at equal angles on the unit circle in 3D (z=1 for scale).
def _build_nt_coordinates():
    """
    Returns P (4×3): each row is the 3D coordinate of one nucleotide.
    Transition vectors V[i][j] are the one-step displacement from nt i to nt j.
    """
    pt = [
        np.array([np.cos(i * 2 * np.pi / RNA_N),
                  np.sin(i * 2 * np.pi / RNA_N),
                  1.0])
        for i in range(RNA_N)
    ]
    P = np.vstack(pt)                                          # (4, 3)
    V = [[pt[i] + (1.0 / RNA_N) * (pt[j] - pt[i])            # (4, 4, 3)
          for j in range(RNA_N)] for i in range(RNA_N)]
    return P, V


_P_GLOBAL, _V_GLOBAL = _build_nt_coordinates()
_NT2IDX = {nt: i for i, nt in enumerate(RNA_ALPHABET)}


def _GRS_static(seq: str, P: np.ndarray, V, motif_groups: list) -> list:
    """
    Trace one GRS walk per motif group.
    motif_groups: list of strings — each string is a subset of 'AUGC' that
                  defines which nucleotides belong to that motif.
    Returns list of (L+1, 3) walk arrays.

    x is always shape (4,) — one entry per nucleotide in RNA_ALPHABET.
    x[j] = 1  iff  current nt == RNA_ALPHABET[j]  AND  RNA_ALPHABET[j] in motif.
    This ensures x @ P has shape (3,) regardless of motif length.
    """
    g = []
    for motif in motif_groups:
        motif_set = set(motif)    # O(1) membership test
        c = [np.zeros(3)]
        d = np.zeros(3, dtype=float)
        y = None
        for i, nt in enumerate(seq):
            # x[j] = 1 iff current nucleotide is RNA_ALPHABET[j] and in motif
            x = np.array(
                [(nt == base) and (base in motif_set) for base in RNA_ALPHABET],
                dtype=float,
            )                          # shape (4,) — always matches P (4, 3)
            if i == 0:
                c.append(c[0] + x @ P)
            else:
                if not np.any(x):
                    d = d * (i - 1) / i
                    c.append(c[i] + np.array([0.0, 0.0, 1.0]) + d)
                elif y is None or not np.any(y):
                    d = d * (i - 1) / i
                    c.append(c[i] + x @ P + d)
                else:
                    prev_idx = int(np.where(y)[0][0])
                    curr_idx = int(np.where(x)[0][0])
                    d = d * (i - 1) / i + V[prev_idx][curr_idx] / i
                    c.append(c[i] + x @ P + d)
            y = x
        g.append(np.vstack(c))
    return g


def _ME_static(W: np.ndarray) -> float:
    """
    Maximal Eigenvalue of the normalised graph Laplacian built from
    the GRS trajectory matrix W.
    """
    W = W[1:]                           # drop origin
    x = W.shape[0]
    if x < 2:
        return 0.0
    D   = pdist(W)
    E   = squareform(D)
    sdist = np.zeros((x, x), dtype=float)
    for i in range(x):
        for j in range(i, x):
            if j - i == 1:
                sdist[i, j] = E[i, j]
            elif j - i > 1:
                sdist[i, j] = sdist[i, j - 1] + E[j - 1, j]
    sdist += sdist.T
    sdd = sdist + np.diag(np.ones(x))
    L   = E / sdd
    try:
        val = eigs(L, k=1, which="LM", return_eigenvectors=False)[0]
        return float(np.real(val) / x)
    except Exception:
        return float(np.linalg.eigvals(L).real.max() / x)


# ── Composition features ──────────────────────────────────────────────────────
def _composition_features(seq: str) -> tuple:
    """
    Returns:
      FC  (4,)    nucleotide composition
      FD  (16,)   dinucleotide composition (row-major: AA, AU, AG, AC, UA, …)
      FT  (64,)   trinucleotide composition
    """
    nt    = RNA_ALPHABET
    n     = len(seq)
    nc    = {b: 0 for b in nt}
    dc    = {a + b: 0 for a in nt for b in nt}
    tc    = {a + b + c: 0 for a in nt for b in nt for c in nt}

    for ch in seq:
        if ch in nc:
            nc[ch] += 1
    for i in range(len(seq) - 1):
        k = seq[i] + seq[i + 1]
        if k in dc:
            dc[k] += 1
    for i in range(len(seq) - 2):
        k = seq[i] + seq[i + 1] + seq[i + 2]
        if k in tc:
            tc[k] += 1

    FC = np.array([nc[b] / max(n, 1)         for b in nt],                  dtype=float)
    FD = np.array([dc[a + b] / max(n - 1, 1) for a in nt for b in nt],      dtype=float)
    FT = np.array([tc[a+b+c] / max(n - 2, 1) for a in nt for b in nt for c in nt], dtype=float)
    return FC, FD, FT


# ── RNA motif groups (analog of M.mat protein motifs) ────────────────────────
# Default: 10 biologically motivated nucleotide groupings
DEFAULT_MOTIF_GROUPS = [
    "AU",     # Watson-Crick pair partners
    "GC",     # GC pair partners
    "AG",     # purines
    "UC",     # pyrimidines
    "G",      # G alone  (G4 driver)
    "C",      # C alone
    "A",      # A alone
    "U",      # U alone
    "GU",     # wobble pair
    "AUG",    # start-codon-like, AUG-rich
]


# ── Main Extractor Class ──────────────────────────────────────────────────────
class RNAFEGSFeatureExtractor:
    """
    Compute RNA-FEGS feature vectors for a list of RNA sequences.

    Parameters
    ----------
    motif_groups : list of str, optional
        Each string defines the nucleotides in one motif group.
        Defaults to DEFAULT_MOTIF_GROUPS (10 groups).
    include_trinuc : bool
        Whether to append trinucleotide composition (64 features). Default True.
    include_g4 : bool
        Whether to append G4-quadruplex density features. Default True.
    processes : int or None
        Number of parallel worker processes. None = use all CPUs.
    """

    def __init__(
        self,
        motif_groups: list = None,
        include_trinuc: bool = True,
        include_g4: bool = True,
        processes: int = None,
    ):
        self.motif_groups  = motif_groups or DEFAULT_MOTIF_GROUPS
        self.include_trinuc = include_trinuc
        self.include_g4    = include_g4
        self.processes     = processes
        self.num_M         = len(self.motif_groups)

    @staticmethod
    def _load_sequences(sequences, start: int, end):
        if isinstance(sequences, str) and sequences.lower().endswith((".fa", ".fasta", ".fna")):
            seqs = [_normalise_rna(str(r.seq)) for r in SeqIO.parse(sequences, "fasta")]
        elif isinstance(sequences, (list, tuple, np.ndarray)):
            seqs = [_normalise_rna(str(s)) for s in sequences]
        else:
            raise ValueError("`sequences` must be a FASTA path or a list/array of RNA strings.")
        return seqs[start:end]

    @staticmethod
    def _worker_grs(args):
        seq, motif_groups = args
        P, V = _build_nt_coordinates()
        return _GRS_static(seq, P, V, motif_groups)

    @staticmethod
    def _worker_comp(args):
        seq, include_trinuc, include_g4 = args
        FC, FD, FT = _composition_features(seq)
        parts = [FC, FD]
        if include_trinuc:
            parts.append(FT)
        if include_g4:
            g4 = np.array([
                count_g4_motifs(seq) / max(len(seq), 1),
                g4_density_windows(seq),
                seq.count("G") / max(len(seq), 1),  # G fraction (G4 proxy)
            ], dtype=float)
            parts.append(g4)
        return np.concatenate(parts)

    def extract(
        self,
        sequences,
        start_seq: int = 0,
        end_seq: int = None,
    ) -> np.ndarray:
        """
        Extract RNA-FEGS features.

        Parameters
        ----------
        sequences : str (FASTA path) or list of RNA strings
        start_seq, end_seq : slice bounds (optional)

        Returns
        -------
        np.ndarray of shape (N, num_features)
            num_features = num_M  [EL]
                         + 4      [FC]
                         + 16     [FD]
                         + 64     [FT, if include_trinuc]
                         + 3      [G4, if include_g4]
        """
        seqs = self._load_sequences(sequences, start_seq, end_seq)
        N    = len(seqs)

        # ── EL (graph eigenvalue) features ───────────────────────────────────
        with Pool(processes=self.processes) as pool:
            grs_results = pool.map(
                self._worker_grs,
                [(seq, self.motif_groups) for seq in seqs]
            )

        EL_vals = [
            self._ME_static(W)
            for g_list in tqdm(grs_results, desc="ME eigenvalue", leave=False)
            for W in g_list
        ]
        EL = np.array(EL_vals, dtype=float).reshape(N, self.num_M)

        # ── Composition + G4 features ─────────────────────────────────────────
        with Pool(processes=self.processes) as pool:
            comp_results = pool.map(
                self._worker_comp,
                [(seq, self.include_trinuc, self.include_g4) for seq in seqs]
            )
        COMP = np.vstack(comp_results)

        return np.hstack([EL, COMP]).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        base = self.num_M + 4 + 16
        if self.include_trinuc:
            base += 64
        if self.include_g4:
            base += 3
        return base
