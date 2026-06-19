"""
RNAPhaseek — De Novo LLPS RNA Sequence Generator
==================================================
Four generation methods, ordered by increasing biological sophistication:

  1. AutoRegressive   – greedy left-to-right discrete design          (fast)
  2. SeqProp          – gradient-based continuous relaxation           (optimal)
  3. StructSeqProp    – SeqProp + differentiable structural constraints (RNA-specific)
  4. CondSeqProp      – SeqProp conditioned on a target condensate type (RNA-specific)

Methods 1–2 are direct ports of Phaseek's generator.py adapted for the
4-nucleotide RNA alphabet.
Methods 3–4 are new contributions unique to RNA: protein generators cannot
exploit secondary structure cheaply, and Phaseek's training data lacks the
per-condensate labelling that RPS 2.0 provides.

Technical note — why SeqProp works here
-----------------------------------------
The RNAPhaseekClassifier uses a learned embedding table (transformer.wte,
vocab_size × d_model). SeqProp bypasses the discrete tokeniser entirely and
instead passes a *soft embedding*:

    E = P_soft @ W_NT       P_soft : (L, 4)   W_NT : (4, d_model)

Because matrix multiplication is differentiable, ∂loss/∂theta flows back
through E into theta — the positional-probability logits we optimise.
Graph-bias (Lhat_stack) is dropped during generation because FEGS matrices
cannot be pre-computed for hypothetical sequences; the sequence attention
alone provides sufficient signal.

Usage — CLI
-----------
    # Autoregressive
    python Functions/generator.py --method ar --length 200 --num_seqs 5

    # SeqProp
    python Functions/generator.py --method seqprop --length 150 --steps 400

    # Structure-constrained SeqProp
    python Functions/generator.py --method struct --length 150 --g4_weight 0.3

    # Condensate-conditioned SeqProp
    python Functions/generator.py --method cond --condensate stress_granule --length 150

    # List valid condensate names
    python Functions/generator.py --list_condensates
"""

import sys
import argparse
import math
import re
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Functions.RNAPhaseek.RNAPhaseek        import RNAPhaseekClassifier, Config
from Functions.RNAPhaseek.RNAPhaseek_utils  import setup_device

# ── ViennaRNA (optional) ──────────────────────────────────────────────────────
try:
    import RNA as _ViennaRNA
    _VIENNA_OK = True
except ImportError:
    _VIENNA_OK = False

# =============================================================================
# Constants
# =============================================================================

RNA_BASES = "AUGC"          # fixed order — index 0=A, 1=U, 2=G, 3=C
N_BASES   = 4
A_IDX, U_IDX, G_IDX, C_IDX = 0, 1, 2, 3

# Condensate vocabulary from RPS 2.0 (24 types)
# Used by CondSeqProp; requires model trained with condensate conditioning.
CONDENSATE_TYPES = {
    "stress_granule":        0,
    "p_body":                1,
    "nuclear_speckle":       2,
    "cajal_body":            3,
    "nucleolus":             4,
    "paraspeckle":           5,
    "germ_granule":          6,
    "processing_body":       7,
    "neuronal_rna_granule":  8,
    "pml_body":              9,
    "heterochromatin":      10,
    "chromatin":            11,
    "nuclear_pore":         12,
    "centrosome":           13,
    "mitochondria":         14,
    "chloroplast":          15,
    "cytoplasm":            16,
    "synapse":              17,
    "p_granule":            18,
    "z_granule":            19,
    "stress_body":          20,
    "rna_transport_granule":21,
    "amyloid":              22,
    "other":                23,
}

# G4 motif: G{2+}[loop 1-7]G{2+}[loop 1-7]G{2+}[loop 1-7]G{2+}
_G4_RE = re.compile(r"(G{2,}[AUGCN]{1,7}){3}G{2,}", re.IGNORECASE)

# =============================================================================
# Model loader
# =============================================================================

def load_model(ckpt_path: str, device: str) -> RNAPhaseekClassifier:
    """Load a trained RNAPhaseekClassifier from a checkpoint."""
    state      = torch.load(ckpt_path, map_location=device, weights_only=True)
    vocab_size = state["transformer.wte.weight"].shape[0]
    n_embd     = state["transformer.wte.weight"].shape[1]
    block_size = state["transformer.wpe.weight"].shape[0]
    n_layers   = sum(1 for k in state
                     if k.startswith("transformer.h.") and k.endswith(".ln_1.weight"))
    n_heads    = state.get("_n_heads", 8)          # saved at training time if possible
    topk_m     = state["mixer.alpha"].shape[0]

    cfg = Config(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layers, n_head=n_heads, n_embd=n_embd,
        embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
        causal=False, use_graph_bias=True,
    )
    model = RNAPhaseekClassifier(cfg, topk_m=topk_m).to(device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_nt_embedding_matrix(model: RNAPhaseekClassifier,
                             nt_token_ids: list,
                             device: str) -> torch.Tensor:
    """
    Extract the (4, d_model) sub-matrix of wte for the four nucleotide tokens.
    nt_token_ids[i] = vocabulary index for RNA_BASES[i] (A, U, G, C).
    """
    W_full = model.transformer.wte.weight          # (vocab_size, d_model)
    ids    = torch.tensor(nt_token_ids, dtype=torch.long, device=device)
    return W_full[ids, :].detach()                 # (4, d_model)  — no grad needed


# =============================================================================
# Shared forward pass (soft embedding → classifier score)
# =============================================================================

def _soft_forward(
    model:   RNAPhaseekClassifier,
    P_soft:  torch.Tensor,       # (L, 4)  soft nucleotide distribution
    W_NT:    torch.Tensor,       # (4, d_model)  nucleotide embedding matrix
    device:  str,
) -> torch.Tensor:
    """
    Differentiable forward pass through the classifier using a soft embedding.
    Returns a scalar LLPS probability in [0, 1].

    Skips graph-bias (Lhat_stack=None) because FEGS matrices are not available
    for hypothetical sequences.
    """
    E   = P_soft @ W_NT                                    # (L, d_model)
    x   = E.unsqueeze(0)                                   # (1, L, d_model)
    B, T, C = x.size()

    pos     = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)
    pos_emb = model.transformer.wpe(pos)
    x       = model.transformer.drop(x + pos_emb)

    for block in model.transformer.h:
        # No graph bias, no causal mask, no padding mask — full bidirectional attn
        x = block(x, bias_per_head=None, key_padding_mask=None)

    x      = model.transformer.ln_f(x)
    pooled = x.mean(dim=1)                                 # global average pool
    logits = model.head(pooled)                            # (1, 2)
    score  = torch.softmax(logits, dim=-1)[0, 1]           # scalar LLPS prob
    return score


# =============================================================================
# Scoring utilities (discrete sequences)
# =============================================================================

def _encode_sequence(seq: str, vocab: dict, seq_len: int) -> torch.Tensor:
    """Tokenise a single RNA string into a (1, seq_len) long tensor (zero-padded)."""
    ids = [vocab.get(c, vocab.get("<unk>", 1)) for c in seq[:seq_len]]
    ids += [0] * (seq_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def score_sequence(seq: str, model: RNAPhaseekClassifier,
                   vocab: dict, seq_len: int, device: str) -> float:
    """Score a discrete RNA sequence string → LLPS probability."""
    ids    = _encode_sequence(seq, vocab, seq_len).to(device)
    with torch.no_grad():
        logits, _ = model(ids, targets=None, Lhat_stack=None)
        prob = float(torch.softmax(logits, dim=-1)[0, 1].item())
    return prob


# =============================================================================
# Greedy single-position refinement (post-decode, discrete)
# =============================================================================

def greedy_refine(
    seq:        str,
    model:      RNAPhaseekClassifier,
    vocab:      dict,
    seq_len:    int,
    device:     str,
    max_iters:  int = 200,
) -> tuple:
    """
    Iteratively try every single-nucleotide substitution and accept any that
    improves the LLPS score. Stops when no improvement is found.
    Returns (refined_seq, final_score).
    """
    best = list(seq)
    best_score = score_sequence("".join(best), model, vocab, seq_len, device)

    for _ in range(max_iters):
        improved = False
        for pos in range(len(best)):
            orig = best[pos]
            for nt in RNA_BASES:
                if nt == orig:
                    continue
                best[pos] = nt
                s = score_sequence("".join(best), model, vocab, seq_len, device)
                if s > best_score + 1e-5:
                    best_score = s
                    improved   = True
                    break           # restart outer loop from new best
                best[pos] = orig    # revert
            if improved:
                break
        if not improved:
            break

    return "".join(best), best_score


# =============================================================================
# Differentiable soft structural penalties  (RNA-specific, used by Methods 3 & 4)
# =============================================================================

def _soft_g4_reward(P_soft: torch.Tensor, window: int = 4) -> torch.Tensor:
    """
    Differentiable approximation of G4 content.

    A G4 quadruplex requires at least 4 consecutive G-rich windows.
    We approximate this as the mean probability of encountering
    windows of ≥ `window` consecutive high-P(G) positions.

    Returns a scalar in [0, 1]: higher = more G4-like.
    """
    P_G   = P_soft[:, G_IDX]                        # (L,)  prob of G at each pos
    L     = P_G.shape[0]
    if L < window * 4:
        return P_G.mean()                            # sequence too short for real G4

    # Convolve: compute mean P(G) in every window of size `window`
    kernel = torch.ones(1, 1, window, device=P_G.device, dtype=P_G.dtype) / window
    pg_win = F.conv1d(P_G.view(1, 1, -1), kernel, padding=0).squeeze()  # (L-window+1,)

    # "G-run score" = mean of soft-max(pg_win - 0.5, 0)
    g_run  = F.relu(pg_win - 0.5)
    return g_run.mean()


def _soft_gc_balance_penalty(P_soft: torch.Tensor,
                              gc_min: float = 0.35,
                              gc_max: float = 0.65) -> torch.Tensor:
    """
    Penalises sequences whose expected GC content falls outside [gc_min, gc_max].
    Returns a non-negative scalar; 0 = within bounds.
    LLPS-associated RNAs tend to avoid extreme GC content.
    """
    P_GC  = P_soft[:, G_IDX] + P_soft[:, C_IDX]    # expected GC fraction per pos
    gc    = P_GC.mean()
    low   = F.relu(gc_min - gc)
    high  = F.relu(gc - gc_max)
    return low + high


def _soft_repeat_reward(P_soft: torch.Tensor,
                         min_period: int = 3,
                         max_period: int = 6) -> torch.Tensor:
    """
    Reward low-complexity / repetitive regions.
    LLPS is strongly associated with repeat-containing RNAs (CAG, CUG, etc.).
    Approximates periodicity by measuring self-correlation of P_soft at
    lags [min_period, max_period].
    Returns a scalar in [0, 1].
    """
    L = P_soft.shape[0]
    scores = []
    for lag in range(min_period, min(max_period + 1, L // 2)):
        corr = (P_soft[:L - lag] * P_soft[lag:]).sum(dim=-1).mean()
        scores.append(corr)
    if not scores:
        return torch.tensor(0.0, device=P_soft.device)
    return torch.stack(scores).mean()


def _soft_au_rich_reward(P_soft: torch.Tensor) -> torch.Tensor:
    """
    Reward AU-rich elements (AREs), common in stress-granule-associated mRNAs.
    Returns expected fraction of A+U per position.
    """
    return (P_soft[:, A_IDX] + P_soft[:, U_IDX]).mean()


def _vienna_fold_score(seq: str) -> float:
    """
    Call ViennaRNA to get the MFE structure energy.
    Returns -mfe/L (normalised, higher = more structured).
    Falls back to 0.0 if ViennaRNA unavailable.
    """
    if not _VIENNA_OK or not seq:
        return 0.0
    seq_dna = seq.replace("U", "T")   # ViennaRNA accepts both
    try:
        _, mfe = _ViennaRNA.fold(seq_dna)
        return float(-mfe / max(len(seq), 1))
    except Exception:
        return 0.0


# =============================================================================
# Method 1 — AutoRegressive Design
# =============================================================================

def run_autoregressive(
    model:      RNAPhaseekClassifier,
    vocab:      dict,
    length:     int   = 150,
    k_choices:  int   = 3,
    start_seq:  str   = "AUG",       # mimic a start-codon-like seed
    seq_len:    int   = 1024,
    device:     str   = "cpu",
    seed:       int   = 0,
    refine:     bool  = True,
) -> tuple:
    """
    Greedy autoregressive sequence design.

    At each position, all 4 nucleotide extensions are scored; the next
    nucleotide is sampled uniformly from the top-k scoring choices.

    Parameters
    ----------
    k_choices : int
        Diversity knob — k=1 is fully greedy, k=4 is random.
    start_seq : str
        Seed (typically 1–5 nucleotides). Positions not in RNA_BASES are dropped.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    built = [b for b in start_seq.upper().replace("T", "U") if b in RNA_BASES]
    if not built:
        built = [np.random.choice(list(RNA_BASES))]

    pbar = trange(len(built), length, desc="AutoRegressive", ncols=90, leave=False)
    for _ in pbar:
        candidates   = ["".join(built) + nt for nt in RNA_BASES]
        scores       = [score_sequence(c, model, vocab, seq_len, device) for c in candidates]
        top_k_nts    = [RNA_BASES[i] for i in np.argsort(scores)[::-1][:k_choices]]
        chosen       = np.random.choice(top_k_nts)
        built.append(chosen)
        pbar.set_postfix(score=f"{max(scores):.4f}", nt=chosen)

    seq   = "".join(built[:length])
    score = score_sequence(seq, model, vocab, seq_len, device)
    print(f"\nAutoRegressive — raw score: {score:.4f}")

    if refine:
        seq, score = greedy_refine(seq, model, vocab, seq_len, device)
        print(f"After greedy refinement:    {score:.4f}")

    return seq, score


# =============================================================================
# Method 2 — SeqProp  (gradient-based, continuous relaxation)
# =============================================================================

def run_seqprop(
    model:          RNAPhaseekClassifier,
    W_NT:           torch.Tensor,
    length:         int   = 150,
    steps:          int   = 400,
    lr:             float = 0.08,
    entropy_weight: float = 0.08,
    temp_start:     float = 2.0,
    temp_end:       float = 0.1,
    seed:           int   = 0,
    log_every:      int   = 50,
    device:         str   = "cpu",
    refine:         bool  = True,
    vocab:          dict  = None,
    seq_len:        int   = 1024,
) -> tuple:
    """
    Gradient-based continuous-relaxation design (SeqProp).

    Optimises theta ∈ R^{L×4} to maximise the LLPS classifier score while
    maintaining sequence diversity (entropy penalty).

    Temperature annealing: high temperature early (exploration) →
    low temperature late (exploitation / sharpening).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    L     = length
    theta = torch.randn(L, N_BASES, device=device, requires_grad=True)
    opt   = torch.optim.Adam([theta], lr=lr)

    best_theta  = None
    best_score  = -1.0

    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"SeqProp | L={L}  steps={steps}  lr={lr}  ew={entropy_weight}  "
          f"T {temp_start}→{temp_end}")

    for step in range(steps):
        temp    = temp_start + (temp_end - temp_start) * (step / steps)
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            P_soft  = F.softmax(theta / temp, dim=-1)      # (L, 4)
            score   = _soft_forward(model, P_soft, W_NT, device)

            # Entropy penalty — prevent collapse to a single nucleotide per pos
            P_real  = F.softmax(theta, dim=-1)
            entropy = -(P_real * P_real.clamp(min=1e-9).log()).sum(-1).mean()
            loss    = -score + entropy_weight * entropy

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        sv = float(score.item())
        if sv > best_score and step > steps // 3:
            best_score = sv
            best_theta = theta.detach().clone()

        if (step + 1) % log_every == 0 or step == 0:
            print(f"  step {step+1:>4}/{steps}  T={temp:.2f}  "
                  f"score={sv:.4f}  entropy={float(entropy.item()):.3f}")

    if best_theta is None:
        best_theta = theta.detach()

    # Argmax decode
    with torch.no_grad():
        idxs = best_theta.argmax(dim=-1).cpu().numpy()
    seq   = "".join(RNA_BASES[i] for i in idxs)
    score_raw = score_sequence(seq, model, vocab, seq_len, device) if vocab else best_score
    print(f"\nSeqProp argmax decode — score: {score_raw:.4f}")

    if refine and vocab:
        seq, score_raw = greedy_refine(seq, model, vocab, seq_len, device)
        print(f"After greedy refinement  — score: {score_raw:.4f}")

    return seq, score_raw


# =============================================================================
# Method 3 — StructSeqProp  (SeqProp + differentiable structural constraints)
# =============================================================================

def run_struct_seqprop(
    model:              RNAPhaseekClassifier,
    W_NT:               torch.Tensor,
    length:             int   = 150,
    steps:              int   = 400,
    lr:                 float = 0.08,
    entropy_weight:     float = 0.08,
    g4_weight:          float = 0.25,      # reward G4 motif probability
    repeat_weight:      float = 0.10,      # reward repeat / low-complexity content
    au_weight:          float = 0.00,      # reward AU-rich elements (for SG RNAs)
    gc_penalty_weight:  float = 0.20,      # penalise extreme GC content
    gc_min:             float = 0.35,
    gc_max:             float = 0.65,
    temp_start:         float = 2.0,
    temp_end:           float = 0.1,
    seed:               int   = 0,
    log_every:          int   = 50,
    device:             str   = "cpu",
    refine:             bool  = True,
    vocab:              dict  = None,
    seq_len:            int   = 1024,
    use_vienna_reward:  bool  = False,    # call ViennaRNA every `vienna_every` steps
    vienna_every:       int   = 50,
    vienna_weight:      float = 0.10,
) -> tuple:
    """
    Structure-constrained SeqProp — unique to RNA.

    Extends SeqProp with four differentiable soft penalties derived from
    known RNA sequence features associated with LLPS:

      g4_weight        →  reward G-quadruplex-forming sequence patterns
      repeat_weight    →  reward low-complexity / repeat regions
      au_weight        →  reward AU-rich elements (stress granule association)
      gc_penalty_weight→  penalise extreme GC content

    Optionally calls ViennaRNA at intervals to add a non-differentiable
    structure reward (averaged over the last few decoded steps).

    Recommended weight presets:
      G4-rich / P-body RNAs      : g4=0.3, repeat=0.05, au=0.0,  gc_pen=0.2
      Stress-granule mRNA-like   : g4=0.1, repeat=0.10, au=0.15, gc_pen=0.2
      General LLPS-prone ncRNA   : g4=0.2, repeat=0.10, au=0.05, gc_pen=0.2  (default)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    L      = length
    theta  = torch.randn(L, N_BASES, device=device, requires_grad=True)
    opt    = torch.optim.Adam([theta], lr=lr)

    best_theta = None
    best_score = -1.0
    vienna_bonus = 0.0          # updated asynchronously

    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"StructSeqProp | L={L}  g4={g4_weight}  repeat={repeat_weight}  "
          f"au={au_weight}  gc_pen={gc_penalty_weight}")

    for step in range(steps):
        temp   = temp_start + (temp_end - temp_start) * (step / steps)
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            P_soft = F.softmax(theta / temp, dim=-1)            # (L, 4)

            # ── Primary LLPS score ────────────────────────────────────────────
            llps_score = _soft_forward(model, P_soft, W_NT, device)

            # ── Diversity (entropy) penalty ───────────────────────────────────
            P_real  = F.softmax(theta, dim=-1)
            entropy = -(P_real * P_real.clamp(min=1e-9).log()).sum(-1).mean()

            # ── Structural rewards / penalties ────────────────────────────────
            g4_rew    = _soft_g4_reward(P_soft)          if g4_weight     > 0 else 0.0
            rep_rew   = _soft_repeat_reward(P_soft)      if repeat_weight > 0 else 0.0
            au_rew    = _soft_au_rich_reward(P_soft)     if au_weight     > 0 else 0.0
            gc_pen    = _soft_gc_balance_penalty(
                            P_soft, gc_min, gc_max)      if gc_penalty_weight > 0 else 0.0

            loss = (
                - llps_score
                + entropy_weight      * entropy
                - g4_weight           * g4_rew
                - repeat_weight       * rep_rew
                - au_weight           * au_rew
                + gc_penalty_weight   * gc_pen
                - vienna_weight       * vienna_bonus     # non-differentiable, scalar
            )

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # ── Optional ViennaRNA structure reward (non-differentiable) ─────────
        if use_vienna_reward and _VIENNA_OK and (step + 1) % vienna_every == 0:
            with torch.no_grad():
                decoded = "".join(RNA_BASES[i] for i in
                                  P_soft.detach().argmax(dim=-1).cpu().numpy())
            vienna_bonus = _vienna_fold_score(decoded)

        sv = float(llps_score.item())
        if sv > best_score and step > steps // 3:
            best_score = sv
            best_theta = theta.detach().clone()

        if (step + 1) % log_every == 0 or step == 0:
            print(f"  step {step+1:>4}/{steps}  T={temp:.2f}  llps={sv:.4f}  "
                  f"g4={float(g4_rew.item() if hasattr(g4_rew,'item') else g4_rew):.3f}  "
                  f"gc_pen={float(gc_pen.item() if hasattr(gc_pen,'item') else gc_pen):.3f}")

    if best_theta is None:
        best_theta = theta.detach()

    with torch.no_grad():
        idxs = best_theta.argmax(dim=-1).cpu().numpy()
    seq        = "".join(RNA_BASES[i] for i in idxs)
    score_raw  = score_sequence(seq, model, vocab, seq_len, device) if vocab else best_score
    print(f"\nStructSeqProp argmax decode — score: {score_raw:.4f}")

    # ViennaRNA structure summary
    if _VIENNA_OK:
        struct_score = _vienna_fold_score(seq)
        print(f"ViennaRNA structure score (−MFE/L): {struct_score:.4f}")

    if refine and vocab:
        seq, score_raw = greedy_refine(seq, model, vocab, seq_len, device)
        print(f"After greedy refinement  — score: {score_raw:.4f}")

    return seq, score_raw


# =============================================================================
# Method 4 — CondSeqProp  (condensate-conditioned generation)
# =============================================================================
# Requires the classifier to have been trained with a condensate prefix token.
# If the loaded model does NOT have a condensate embedding, this falls back
# to StructSeqProp with AU-rich weighting (stress-granule default).

def run_cond_seqprop(
    model:          RNAPhaseekClassifier,
    W_NT:           torch.Tensor,
    condensate:     str   = "stress_granule",
    length:         int   = 150,
    steps:          int   = 400,
    lr:             float = 0.08,
    entropy_weight: float = 0.08,
    temp_start:     float = 2.0,
    temp_end:       float = 0.1,
    seed:           int   = 0,
    log_every:      int   = 50,
    device:         str   = "cpu",
    refine:         bool  = True,
    vocab:          dict  = None,
    seq_len:        int   = 1024,
) -> tuple:
    """
    Condensate-conditioned SeqProp.

    Designs an RNA whose LLPS score is maximised *specifically for* the
    requested condensate type. Requires the model to have a condensate embedding
    table (``model.condensate_emb``), trained with condensate-label prefixing.

    If the model was trained without condensate conditioning (no
    ``condensate_emb`` attribute), the function gracefully falls back to
    StructSeqProp with preset weights tuned for the requested condensate.

    Condensate-specific weight presets (fallback mode)
    ---------------------------------------------------
    stress_granule   →  high AU-rich reward, moderate repeat
    p_body           →  high G4 reward, low AU
    nucleolus        →  high GC balance, high repeat
    other / default  →  balanced weights
    """
    if condensate not in CONDENSATE_TYPES:
        raise ValueError(
            f"Unknown condensate '{condensate}'. "
            f"Valid names: {sorted(CONDENSATE_TYPES.keys())}"
        )

    cond_id = CONDENSATE_TYPES[condensate]

    # ── Check whether the model has condensate conditioning ──────────────────
    has_cond_emb = hasattr(model, "condensate_emb")

    if has_cond_emb:
        return _cond_seqprop_native(
            model, W_NT, cond_id, condensate,
            length, steps, lr, entropy_weight,
            temp_start, temp_end, seed, log_every,
            device, refine, vocab, seq_len,
        )
    else:
        print(f"[CondSeqProp] Model has no condensate embedding — using "
              f"StructSeqProp with '{condensate}'-tuned weights.")
        g4_w, rep_w, au_w = _condensate_weights(condensate)
        return run_struct_seqprop(
            model=model, W_NT=W_NT, length=length, steps=steps,
            lr=lr, entropy_weight=entropy_weight,
            g4_weight=g4_w, repeat_weight=rep_w, au_weight=au_w,
            gc_penalty_weight=0.20,
            temp_start=temp_start, temp_end=temp_end,
            seed=seed, log_every=log_every, device=device,
            refine=refine, vocab=vocab, seq_len=seq_len,
        )


def _condensate_weights(condensate: str) -> tuple:
    """
    Return (g4_weight, repeat_weight, au_weight) preset for each condensate type.
    Based on known biology of each condensate's RNA composition.
    """
    presets = {
        # condensate        g4     repeat  AU-rich
        "stress_granule":  (0.05,  0.15,   0.20),   # AU-rich mRNAs, low complexity
        "p_body":          (0.30,  0.05,   0.00),   # G4-forming, structured
        "nucleolus":       (0.10,  0.20,   0.00),   # repeat-rich rRNA regions
        "nuclear_speckle": (0.10,  0.10,   0.05),   # pre-mRNA splicing context
        "paraspeckle":     (0.15,  0.10,   0.10),   # lncRNA-rich
        "germ_granule":    (0.10,  0.20,   0.05),   # piRNA/repeat-associated
        "cajal_body":      (0.05,  0.10,   0.00),
        "neuronal_rna_granule": (0.10, 0.15, 0.10),
    }
    return presets.get(condensate, (0.20, 0.10, 0.05))   # balanced default


def _cond_seqprop_native(
    model, W_NT, cond_id, condensate_name,
    length, steps, lr, entropy_weight,
    temp_start, temp_end, seed, log_every,
    device, refine, vocab, seq_len,
) -> tuple:
    """
    Full condensate-conditioned SeqProp for models with a condensate_emb table.

    The condensate embedding is prepended to the soft sequence embedding so
    the transformer sees: [COND_TOKEN] + [soft_nt_0] + ... + [soft_nt_{L-1}].
    The classifier's mean-pool ignores the condensate token (it is masked).
    """
    torch.manual_seed(seed)

    L     = length
    theta = torch.randn(L, N_BASES, device=device, requires_grad=True)
    opt   = torch.optim.Adam([theta], lr=lr)

    cond_tok = torch.tensor([cond_id], dtype=torch.long, device=device)
    cond_emb = model.condensate_emb(cond_tok)               # (1, d_model)

    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_theta = None
    best_score = -1.0

    print(f"CondSeqProp (native) | condensate='{condensate_name}'  L={L}")

    for step in range(steps):
        temp   = temp_start + (temp_end - temp_start) * (step / steps)
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            P_soft    = F.softmax(theta / temp, dim=-1)     # (L, 4)
            seq_emb   = P_soft @ W_NT                       # (L, d_model)

            # Prepend condensate token embedding
            x = torch.cat([cond_emb, seq_emb], dim=0).unsqueeze(0)  # (1, L+1, d)
            B, T, C = x.size()
            pos     = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)
            x       = model.transformer.drop(x + model.transformer.wpe(pos))

            for block in model.transformer.h:
                x = block(x, bias_per_head=None, key_padding_mask=None)
            x = model.transformer.ln_f(x)

            # Pool over sequence positions only (skip condensate token at pos 0)
            pooled = x[:, 1:, :].mean(dim=1)
            logits = model.head(pooled)
            score  = torch.softmax(logits, dim=-1)[0, 1]

            P_real  = F.softmax(theta, dim=-1)
            entropy = -(P_real * P_real.clamp(min=1e-9).log()).sum(-1).mean()
            loss    = -score + entropy_weight * entropy

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        sv = float(score.item())
        if sv > best_score and step > steps // 3:
            best_score = sv
            best_theta = theta.detach().clone()

        if (step + 1) % log_every == 0 or step == 0:
            print(f"  step {step+1:>4}/{steps}  T={temp:.2f}  score={sv:.4f}")

    if best_theta is None:
        best_theta = theta.detach()

    with torch.no_grad():
        idxs = best_theta.argmax(dim=-1).cpu().numpy()
    seq       = "".join(RNA_BASES[i] for i in idxs)
    score_raw = score_sequence(seq, model, vocab, seq_len, device) if vocab else best_score
    print(f"\nCondSeqProp decode — score: {score_raw:.4f}")

    if refine and vocab:
        seq, score_raw = greedy_refine(seq, model, vocab, seq_len, device)
        print(f"After greedy refinement — score: {score_raw:.4f}")

    return seq, score_raw


# =============================================================================
# Batch generation helper
# =============================================================================

def batch_generate(
    method:     str,
    num_seqs:   int,
    model:      RNAPhaseekClassifier,
    W_NT:       torch.Tensor,
    vocab:      dict,
    device:     str,
    out_fasta:  Optional[str] = None,
    **method_kwargs,
) -> list:
    """
    Run any of the four generation methods `num_seqs` times.
    Each run uses a different random seed for diversity.

    Returns list of (id, sequence, score) tuples.
    If out_fasta is given, writes results to a FASTA file.
    """
    results = []

    for i in range(num_seqs):
        print(f"\n{'='*60}")
        print(f"  Generating sequence {i+1}/{num_seqs}  (method={method})")
        print(f"{'='*60}")

        kwargs  = {**method_kwargs, "seed": method_kwargs.get("seed", 0) + i}
        kwargs.update({"model": model, "W_NT": W_NT, "device": device,
                       "vocab": vocab})

        if method == "ar":
            kwargs.pop("W_NT", None)   # AR doesn't use W_NT
            seq, score = run_autoregressive(**{k: v for k, v in kwargs.items()
                                               if k in run_autoregressive.__code__.co_varnames})
        elif method == "seqprop":
            seq, score = run_seqprop(**{k: v for k, v in kwargs.items()
                                        if k in run_seqprop.__code__.co_varnames})
        elif method == "struct":
            seq, score = run_struct_seqprop(**{k: v for k, v in kwargs.items()
                                               if k in run_struct_seqprop.__code__.co_varnames})
        elif method == "cond":
            seq, score = run_cond_seqprop(**{k: v for k, v in kwargs.items()
                                             if k in run_cond_seqprop.__code__.co_varnames})
        else:
            raise ValueError(f"Unknown method '{method}'. Choose: ar, seqprop, struct, cond")

        seq_id = f"rna_llps_design_{i+1:04d}_{method}_score{score:.3f}"
        results.append((seq_id, seq, score))
        print(f"\nFinal: {seq_id}")
        print(f"  Sequence : {seq[:60]}{'...' if len(seq) > 60 else ''}")
        print(f"  Length   : {len(seq)} nt")
        print(f"  Score    : {score:.4f}")

        if _VIENNA_OK:
            struct, mfe = _ViennaRNA.fold(seq.replace("U", "T"))
            g4_count = len(_G4_RE.findall(seq))
            gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
            print(f"  MFE      : {mfe:.2f} kcal/mol")
            print(f"  GC       : {gc:.2%}")
            print(f"  G4 motifs: {g4_count}")

    if out_fasta:
        records = [SeqRecord(Seq(s), id=sid, description=f"LLPS_score={sc:.4f}")
                   for sid, s, sc in results]
        SeqIO.write(records, out_fasta, "fasta")
        print(f"\nWrote {len(records)} sequences → {out_fasta}")

    return results


# =============================================================================
# Quick sequence analytics
# =============================================================================

def analyse_sequence(seq: str) -> dict:
    """Return a dict of key RNA LLPS-relevant sequence properties."""
    L  = len(seq)
    gc = (seq.count("G") + seq.count("C")) / max(L, 1)
    g4 = len(_G4_RE.findall(seq))
    au = (seq.count("A") + seq.count("U")) / max(L, 1)

    # Low-complexity: fraction of most common trinucleotide
    tri_counts = {}
    for i in range(L - 2):
        t = seq[i:i+3]
        tri_counts[t] = tri_counts.get(t, 0) + 1
    lc = max(tri_counts.values()) / max(L - 2, 1) if tri_counts else 0.0

    result = {"length": L, "GC": round(gc, 3), "AU": round(au, 3),
              "G4_motifs": g4, "low_complexity": round(lc, 3)}

    if _VIENNA_OK:
        _, mfe = _ViennaRNA.fold(seq.replace("U", "T"))
        result["MFE_kcal_mol"] = round(mfe, 2)
        result["MFE_per_nt"]   = round(mfe / max(L, 1), 4)

    return result


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RNAPhaseek de novo LLPS RNA sequence designer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", choices=["ar", "seqprop", "struct", "cond"],
                   default="seqprop",
                   help="Generation method: ar=AutoRegressive, seqprop=SeqProp, "
                        "struct=Structure-constrained, cond=Condensate-conditioned")
    p.add_argument("--model_ckpt", type=str, default="model/rna_phaseek_best.pt")
    p.add_argument("--vocab_path", type=str, default="model/rna_bpe_vocab.json",
                   help="Path to BPE vocabulary JSON")
    p.add_argument("--length",    type=int,   default=150)
    p.add_argument("--num_seqs",  type=int,   default=1)
    p.add_argument("--steps",     type=int,   default=400,    help="SeqProp steps")
    p.add_argument("--lr",        type=float, default=0.08)
    p.add_argument("--entropy_weight", type=float, default=0.08)
    p.add_argument("--temp_start",     type=float, default=2.0)
    p.add_argument("--temp_end",       type=float, default=0.1)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--log_every", type=int,   default=50)
    p.add_argument("--no_refine", action="store_true")
    p.add_argument("--out_fasta", type=str,   default=None,
                   help="Output FASTA path for generated sequences")
    # AutoRegressive specific
    p.add_argument("--k_choices", type=int,   default=3,
                   help="AR: top-k nucleotides to sample from at each step")
    p.add_argument("--start_seq", type=str,   default="AUG",
                   help="AR: seed sequence")
    # StructSeqProp specific
    p.add_argument("--g4_weight",        type=float, default=0.25)
    p.add_argument("--repeat_weight",    type=float, default=0.10)
    p.add_argument("--au_weight",        type=float, default=0.05)
    p.add_argument("--gc_penalty_weight",type=float, default=0.20)
    p.add_argument("--gc_min",           type=float, default=0.35)
    p.add_argument("--gc_max",           type=float, default=0.65)
    p.add_argument("--use_vienna_reward",action="store_true")
    p.add_argument("--vienna_every",     type=int,   default=50)
    p.add_argument("--vienna_weight",    type=float, default=0.10)
    # CondSeqProp specific
    p.add_argument("--condensate", type=str, default="stress_granule",
                   help="Target condensate type for --method cond")
    p.add_argument("--list_condensates", action="store_true",
                   help="Print all valid condensate names and exit")
    return p


def main():
    args = _build_parser().parse_args()

    if args.list_condensates:
        print("\nValid condensate names (RPS 2.0):")
        for name, idx in sorted(CONDENSATE_TYPES.items(), key=lambda x: x[1]):
            print(f"  {idx:>2}  {name}")
        return

    device = setup_device()

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = args.model_ckpt
    if not Path(ckpt).exists():
        print(f"[!] Checkpoint not found: {ckpt}")
        print("    Train the model first:  python Functions/RNAPhaseek/RNAPhaseek_train.py")
        sys.exit(1)

    print(f"Loading model from {ckpt} ...")
    model = load_model(ckpt, device)

    # ── Load vocabulary ───────────────────────────────────────────────────────
    vocab = {}
    if Path(args.vocab_path).exists():
        import json
        with open(args.vocab_path) as f:
            vocab = json.load(f)
    else:
        print(f"[!] Vocabulary file not found: {args.vocab_path}")
        print("    Scoring and refinement will be disabled.")

    # ── Build nucleotide embedding matrix ─────────────────────────────────────
    # Map single-nucleotide characters A, U, G, C to their vocab IDs.
    nt_ids = []
    for nt in RNA_BASES:
        if nt in vocab:
            nt_ids.append(vocab[nt])
        else:
            # Fallback: use position 1-4 (avoid PAD=0)
            nt_ids.append(RNA_BASES.index(nt) + 1)
            print(f"[!] '{nt}' not in vocab — using fallback id {nt_ids[-1]}")

    W_NT = get_nt_embedding_matrix(model, nt_ids, device)

    # ── Shared kwargs ─────────────────────────────────────────────────────────
    common = dict(
        length=args.length, steps=args.steps, lr=args.lr,
        entropy_weight=args.entropy_weight, temp_start=args.temp_start,
        temp_end=args.temp_end, seed=args.seed, log_every=args.log_every,
        device=device, refine=not args.no_refine, vocab=vocab if vocab else None,
        seq_len=1024,
    )
    method_extra = {
        "ar":     dict(k_choices=args.k_choices, start_seq=args.start_seq),
        "seqprop":dict(),
        "struct": dict(g4_weight=args.g4_weight, repeat_weight=args.repeat_weight,
                       au_weight=args.au_weight, gc_penalty_weight=args.gc_penalty_weight,
                       gc_min=args.gc_min, gc_max=args.gc_max,
                       use_vienna_reward=args.use_vienna_reward,
                       vienna_every=args.vienna_every, vienna_weight=args.vienna_weight),
        "cond":   dict(condensate=args.condensate),
    }

    # ── Run ───────────────────────────────────────────────────────────────────
    results = batch_generate(
        method=args.method,
        num_seqs=args.num_seqs,
        model=model,
        W_NT=W_NT,
        vocab=vocab,
        device=device,
        out_fasta=args.out_fasta,
        **{**common, **method_extra[args.method]},
    )

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'ID':<45} {'Score':>7}  {'Len':>5}  {'GC':>6}  {'G4':>4}")
    print(f"{'─'*70}")
    for sid, seq, score in results:
        stats = analyse_sequence(seq)
        print(f"{sid:<45} {score:>7.4f}  {stats['length']:>5}  "
              f"{stats['GC']:>5.1%}  {stats['G4_motifs']:>4}")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
