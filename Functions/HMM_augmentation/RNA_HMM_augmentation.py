"""
RNA HMM-Guided Data Augmentation
==================================
Adapted from Phaseek's CategoricalHMM_augmentation for protein sequences.

Strategy
--------
For protein sequences, Phaseek used IUPred disorder scores to identify
"folded" segments, then trained an HMM on those segments and augmented
the positive dataset by mutating only folded regions (low mutation rate,
BLOSUM-conservative prior).

For RNA, we replace IUPred with RNAfold base-pair probabilities (bpp):
  - Paired positions   (bpp > threshold) → "structured" → preserve
  - Unpaired positions (bpp ≤ threshold) → "loopy"       → mutate

The HMM is trained on unpaired (loop) segments of LLPS+ RNAs, then used
to guide mutations at loop positions only, keeping structural stems intact.

Requirements
------------
  pip install hmmlearn biopython ViennaRNA

ViennaRNA Python bindings (RNA module) must be installed. If unavailable,
a fallback GC-content-based secondary-structure proxy is used.
"""

import numpy as np
import random
import math
import os
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

try:
    from hmmlearn.hmm import MultinomialHMM
    _HMMLEARN_OK = True
except ImportError:
    _HMMLEARN_OK = False
    print("[WARNING] hmmlearn not found. Install with: pip install hmmlearn")

try:
    import RNA as _RNA          # ViennaRNA Python bindings
    _VIENNA_OK = True
except ImportError:
    _VIENNA_OK = False
    print("[WARNING] ViennaRNA Python bindings not found. Using GC-proxy fallback.")


# ── Alphabet ─────────────────────────────────────────────────────────────────
RNA_BASES  = "AUGC"
BASE2I     = {b: i for i, b in enumerate(RNA_BASES)}
I2BASE     = {i: b for b, i in BASE2I.items()}

# Nearest-neighbour substitution table (conservative for RNA)
# For each base, list biologically tolerated substitutions.
RNA_NEIGHBOURS = {
    "A": "AG",   # purine ↔ purine
    "U": "UC",   # pyrimidine ↔ pyrimidine
    "G": "GA",   # purine ↔ purine
    "C": "CU",   # pyrimidine ↔ pyrimidine
}


# ── Base-pair probability via ViennaRNA ──────────────────────────────────────
def get_bpp_vector(seq: str) -> np.ndarray:
    """
    Compute per-position base-pair probability (bpp) using RNAfold.
    Returns array of shape (len(seq),) with values in [0, 1].
    Higher bpp = position is more likely paired = more structured.

    Falls back to a simple GC-content sliding window if ViennaRNA is unavailable.
    """
    if _VIENNA_OK:
        md   = _RNA.md()
        fc   = _RNA.fold_compound(seq, md)
        _mfe_struct, _mfe = fc.mfe()
        fc.exp_params_rescale(_mfe)
        fc.pf()
        bp_probs = fc.bpp()   # (L+1, L+1) 1-indexed
        L   = len(seq)
        bpp = np.zeros(L, dtype=float)
        for i in range(1, L + 1):
            for j in range(i + 1, L + 1):
                p = bp_probs[i][j]
                if p > 0:
                    bpp[i - 1] += p
                    bpp[j - 1] += p
        return np.clip(bpp, 0.0, 1.0)
    else:
        # Fallback: GC content in a 7-nt window as a proxy for structure
        L   = len(seq)
        bpp = np.zeros(L, dtype=float)
        win = 7
        for i in range(L):
            s   = seq[max(0, i - win // 2): min(L, i + win // 2 + 1)]
            gc  = (s.count("G") + s.count("C")) / max(len(s), 1)
            bpp[i] = gc
        return bpp


# ── Segment identification ────────────────────────────────────────────────────
def loop_mask(bpp: np.ndarray, thr: float = 0.3) -> list:
    """Return list of booleans: True = unpaired/loop, False = structured/stem."""
    return [b < thr for b in bpp]


def loop_segments(seq: str, mask: list, min_len: int = 4) -> list:
    """Extract (start, end) index pairs of loop segments of at least min_len."""
    segs, start = [], None
    for i, m in enumerate(mask + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    return segs


# ── HMM training ─────────────────────────────────────────────────────────────
def train_hmm_on_loops(
    loop_strings: list,
    n_states_grid: range = range(3, 8),
    random_state: int = 0,
) -> tuple:
    """
    Train a Multinomial HMM on concatenated loop sequences.
    Model selection by BIC over n_states_grid.

    Returns (best_model, model_info_dict).
    """
    if not _HMMLEARN_OK:
        raise RuntimeError("hmmlearn is required. Install with: pip install hmmlearn")

    obs, lengths = [], []
    for seg in loop_strings:
        enc = [BASE2I[b] for b in seg if b in BASE2I]
        if enc:
            obs.extend(enc)
            lengths.append(len(enc))

    if not obs:
        raise ValueError("No valid loop segments found for HMM training.")

    X = np.array(obs, dtype=int).reshape(-1, 1)
    best = None
    N    = X.shape[0]

    for n in n_states_grid:
        model = MultinomialHMM(
            n_components=n, n_iter=200, tol=1e-3,
            random_state=random_state, init_params="ste"
        )
        model.fit(X, lengths)
        k   = n * (n - 1) + (n - 1) + n * (len(RNA_BASES) - 1)
        ll  = model.score(X, lengths)
        bic = -2 * ll + k * math.log(max(N, 1))
        if best is None or bic < best["bic"]:
            best = {"n": n, "model": model, "bic": bic, "ll": ll}

    return best["model"], best


# ── HMM-guided mutation ───────────────────────────────────────────────────────
def _posterior_for_segment(model, seg: str) -> np.ndarray:
    enc = np.array([BASE2I[b] for b in seg if b in BASE2I], dtype=int).reshape(-1, 1)
    L   = len(enc)
    if L == 0:
        return np.zeros((0, model.n_components))
    try:
        return model.predict_proba(enc, [L])
    except Exception:
        states = model.predict(enc, [L])
        post   = np.zeros((L, model.n_components))
        post[np.arange(L), states] = 1.0
        return post


def _mixture_emission(model, post: np.ndarray) -> np.ndarray:
    return post @ model.emissionprob_


def _temperature(p: np.ndarray, tau: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    if tau == 1.0:
        return p / (p.sum() + eps)
    q = np.power(p + eps, 1.0 / tau)
    return q / (q.sum() + eps)


def _conservative_prior(
    p: np.ndarray,
    orig_base: str,
    strength: float = 1.0,
    eps: float = 1e-12
) -> np.ndarray:
    allowed = set(RNA_NEIGHBOURS.get(orig_base, orig_base))
    mask    = np.array(
        [1.0 if (b in allowed or b == orig_base) else 1e-6
         for b in RNA_BASES],
        dtype=float,
    )
    q = p * (mask ** strength)
    return q / (q.sum() + eps)


def hmm_guided_mutate_rna(
    seq: str,
    bpp: np.ndarray,
    model,
    thr: float      = 0.3,
    rate: float     = 0.05,
    temperature: float = 0.9,
    conservative_strength: float = 0.5,
    avoid_identity: bool = True,
) -> str:
    """
    Mutate only loop (unpaired) positions of seq guided by the HMM.

    Parameters
    ----------
    seq    : RNA sequence (AUGC)
    bpp    : per-position base-pair probability (from get_bpp_vector)
    model  : trained MultinomialHMM
    thr    : bpp threshold below which a position is considered a loop
    rate   : per-position mutation probability
    temperature : sampling temperature
    conservative_strength : how strongly to enforce RNA_NEIGHBOURS prior
    avoid_identity : never sample the same nucleotide
    """
    out   = list(seq)
    mask  = loop_mask(bpp, thr)
    runs  = loop_segments(seq, mask, min_len=1)

    for s, e in runs:
        seg  = "".join(out[s:e])
        post = _posterior_for_segment(model, seg)
        if post.shape[0] == 0:
            continue
        mixed = _mixture_emission(model, post)

        for t in range(e - s):
            idx = s + t
            if out[idx] not in BASE2I:
                continue
            if random.random() >= rate:
                continue

            p = mixed[t].copy()
            if conservative_strength > 0:
                p = _conservative_prior(p, out[idx], strength=conservative_strength)
            p = _temperature(p, tau=temperature)

            if avoid_identity:
                p[BASE2I[out[idx]]] = 0.0
                s_sum = p.sum()
                p = (p / s_sum) if s_sum > 0 else mixed[t].copy()

            new_base = np.random.choice(list(RNA_BASES), p=p)
            out[idx] = new_base

    return "".join(out)


# ── Dataset augmentation pipeline ────────────────────────────────────────────
def compute_bpp_map(seqs: list) -> dict:
    """Compute per-position bpp for every sequence. Returns {idx: (seq, bpp)}."""
    return {i: (seq, get_bpp_vector(seq)) for i, seq in enumerate(seqs)}


def augment_dataset(
    pos_map: dict,
    model,
    n_copies: int    = 10,
    thr: float       = 0.3,
    rate: float      = 0.05,
    temperature: float = 0.9,
    conservative_strength: float = 0.5,
    avoid_identity: bool = True,
    seed: int        = 48,
) -> list:
    """
    Generate augmented positive RNA sequences.

    Parameters
    ----------
    pos_map : dict {idx: (seq, bpp)}  (from compute_bpp_map)
    model   : trained MultinomialHMM
    n_copies: number of augmented sequences per original

    Returns list of augmented RNA sequence strings.
    """
    np.random.seed(seed)
    random.seed(seed)
    augmented = []
    for idx, (seq, bpp) in pos_map.items():
        for _ in range(n_copies):
            aug = hmm_guided_mutate_rna(
                seq, bpp, model,
                thr=thr, rate=rate, temperature=temperature,
                conservative_strength=conservative_strength,
                avoid_identity=avoid_identity,
            )
            augmented.append(aug)
    return augmented


# ── Convenience: full augmentation pipeline from FASTA ───────────────────────
def augment_from_fasta(
    pos_fasta: str,
    out_fasta: str,
    n_copies: int = 10,
    n_states_grid: range = range(3, 8),
    thr: float = 0.3,
    rate: float = 0.05,
    temperature: float = 0.9,
    conservative_strength: float = 0.5,
    seed: int = 48,
) -> str:
    """
    Full pipeline: read positive FASTA → compute bpp → train HMM →
    augment → write augmented FASTA.

    Returns path to written FASTA file.
    """
    pos_seqs = [str(r.seq).upper().replace("T", "U")
                for r in SeqIO.parse(pos_fasta, "fasta")]

    print(f"Computing base-pair probabilities for {len(pos_seqs)} sequences...")
    pos_map = compute_bpp_map(pos_seqs)

    # Collect loop segments for HMM training
    loop_strs = []
    for seq, bpp in pos_map.values():
        mask = loop_mask(bpp, thr)
        for s, e in loop_segments(seq, mask, min_len=4):
            seg = "".join(b for b in seq[s:e] if b in BASE2I)
            if seg:
                loop_strs.append(seg)

    print(f"Training HMM on {len(loop_strs)} loop segments...")
    hmm, info = train_hmm_on_loops(loop_strs, n_states_grid=n_states_grid)
    print(f"Best HMM: n_states={info['n']} | BIC={info['bic']:.2f}")

    print(f"Augmenting (×{n_copies}) ...")
    aug_seqs = augment_dataset(
        pos_map, hmm, n_copies=n_copies, thr=thr,
        rate=rate, temperature=temperature,
        conservative_strength=conservative_strength, seed=seed,
    )

    records = [
        SeqRecord(Seq(seq), id=f"aug_{i+1}", description="rna_llps_aug")
        for i, seq in enumerate(aug_seqs)
    ]
    SeqIO.write(records, out_fasta, "fasta")
    print(f"Wrote {len(records)} augmented sequences to {out_fasta}")
    return out_fasta
