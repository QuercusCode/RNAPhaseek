"""
Hybrid-model de novo LLPS RNA sequence generator
==================================================
SeqProp-style gradient design through the RNAPhaseek Hybrid classifier
(frozen RNA-FM + FEGSTrans adapter + biophysical fusion + head).

Why a separate file from Functions/generator.py?
  The base generator targets RNAPhaseekClassifier (small char-level transformer
  with a simple (vocab × d_model) embedding table). RNA-FM has a 28-token
  vocabulary, tokenizer-mediated input, and does not expose a clean 4-row
  embedding for the four nucleotides. This module extracts those four rows
  manually and bypasses the tokenizer entirely so gradients flow through a
  soft nucleotide distribution into RNA-FM.

How SeqProp works here
-----------------------
  1. Optimize  theta ∈ R^(L × 4)  (one logit per position × nucleotide)
  2. P_soft = softmax(theta / temperature)              -- soft distribution
  3. E_soft = P_soft @ W_NT                             -- (L, 640) soft embed
  4. Build a full RNA-FM input by prepending [CLS] and appending [EOS]:
        inputs_embeds = [CLS_emb, E_soft, EOS_emb]      -- (1, L+2, 640)
  5. Forward through model.backbone(inputs_embeds=...)  -- frozen RNA-FM
  6. Forward through adapter blocks                      -- with Lhat=None
  7. Mean-pool, fuse with zero biophys, classify        -- LLPS prob
  8. loss = -score + entropy_weight·entropy + optional biological terms
  9. Anneal temperature, gradient-descend theta over many steps
 10. argmax(theta) -> discrete sequence + greedy refinement (optional)

Methods supported
-----------------
  seqprop  : pure classifier-guided SeqProp
  struct   : SeqProp + differentiable biological rewards (G4/GC/repeat/AU)
  cond     : SeqProp + condensate-specific reward preset

Run
---
    # Pure classifier-guided design
    python -m Functions.generator_hybrid --method seqprop \
        --ckpt model/phase1/hybrid_best.pt \
        --length 300 --steps 400 --num_seqs 10 \
        --out_fasta designed.fasta

    # Structure-aware: AU-rich + G4 + GC-balanced
    python -m Functions.generator_hybrid --method struct \
        --ckpt model/phase1/hybrid_best.pt \
        --length 300 --steps 400 \
        --g4_weight 0.2 --au_weight 0.3 --gc_min 0.45 --gc_max 0.65 \
        --out_fasta designed_struct.fasta

    # Condensate-specific (stress granules)
    python -m Functions.generator_hybrid --method cond \
        --ckpt model/phase1/hybrid_best.pt \
        --condensate stress_granule --length 300 --num_seqs 50 \
        --out_fasta designed_stress_granules.fasta
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import multimolecule  # noqa: F401 -- registers RnaFmModel with transformers
from transformers import AutoTokenizer

from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs, RNA_FM_DIM
from Functions.RNAPhaseek.RNAPhaseek_utils         import setup_device, set_seed

# Reuse the biology-informed rewards from the base generator -- they operate on
# P_soft and are model-agnostic.
from Functions.generator import (
    _soft_g4_reward,
    _soft_gc_balance_penalty,
    _soft_repeat_reward,
    _soft_au_rich_reward,
    _vienna_fold_score,
    _condensate_weights,
)


# ── RNA-FM nucleotide / special-token IDs (verified at multimolecule v0.0.9) ──
RNA_BASES   = "AUGC"           # P_soft column order: index 0=A, 1=U, 2=G, 3=C
N_BASES     = 4
A_IDX, U_IDX, G_IDX, C_IDX = 0, 1, 2, 3
NT_TOKEN_ID = {"A": 6, "U": 9, "G": 8, "C": 7}   # rows we extract from word_embeddings
CLS_ID      = 1
EOS_ID      = 2


# =============================================================================
# Model loading + W_NT extraction
# =============================================================================

def load_hybrid_for_generation(ckpt_path: str, device: str) -> RNAFMHybridClassifier:
    """Build the hybrid model and load weights; biophys branch is kept.
    Auto-detects bio_dim from the checkpoint so v1 (26-dim) and v2/v3 (33-dim)
    models both load correctly."""
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    bio_dim = int(sd["bio_proj.0.weight"].shape[0]) if "bio_proj.0.weight" in sd else 26
    args = HybridTrainArgs(
        backbone           = "multimolecule/rnafm",
        freeze_backbone    = True,
        unfreeze_last_n    = 0,
        n_adapter_layers   = 2,
        n_heads            = 8,
        topk_m             = 10,
        bio_dim            = bio_dim,
        batch_size         = 1,
        num_workers        = 0,
        fp16_bias          = False,
    )
    model = RNAFMHybridClassifier(args).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  ckpt loaded: bio_dim={bio_dim}  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model


def get_nt_embeddings(model: RNAFMHybridClassifier, device: str) -> tuple:
    """
    Return (W_NT (4, 640), cls_emb (1, 1, 640), eos_emb (1, 1, 640)) from
    RNA-FM's word embedding table. Frozen — no grad needed.
    """
    W_full = model.backbone.embeddings.word_embeddings.weight   # (28, 640)
    nt_rows = torch.tensor([NT_TOKEN_ID[b] for b in RNA_BASES], device=device)
    W_NT    = W_full[nt_rows, :].detach()                       # (4, 640)
    cls_emb = W_full[CLS_ID].detach().view(1, 1, RNA_FM_DIM)
    eos_emb = W_full[EOS_ID].detach().view(1, 1, RNA_FM_DIM)
    return W_NT, cls_emb, eos_emb


# =============================================================================
# Differentiable forward through the hybrid model with a soft sequence
# =============================================================================

def _soft_forward_hybrid(
    model:    RNAFMHybridClassifier,
    P_soft:   torch.Tensor,        # (L, 4)
    W_NT:     torch.Tensor,        # (4, 640)
    cls_emb:  torch.Tensor,        # (1, 1, 640)
    eos_emb:  torch.Tensor,        # (1, 1, 640)
    bio_zero: torch.Tensor,        # (1, 26)  -- placeholder biophys
    device:   str,
) -> torch.Tensor:
    """
    Differentiable LLPS-probability for a soft sequence (L, 4) -> scalar in [0, 1].

    Mirrors RNAFMHybridClassifier.forward() but feeds inputs_embeds instead of
    token_ids, skips FEGS bias (Lhat_stack=None), and uses an all-ones attention
    mask (the soft sequence has no padding).
    """
    L = P_soft.shape[0]

    # Soft nucleotide embedding -> [CLS] + soft + [EOS]
    E_soft = (P_soft @ W_NT).unsqueeze(0)                  # (1, L, 640)
    inputs_embeds = torch.cat([cls_emb, E_soft, eos_emb], dim=1)  # (1, L+2, 640)
    T = L + 2
    attention_mask = torch.ones(1, T, dtype=torch.long, device=device)

    # ── RNA-FM (frozen) ──────────────────────────────────────────────────────
    out = model.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    x = out.last_hidden_state                              # (1, T, 640)

    # ── FEGSTrans adapter, no graph bias for hypothetical sequences ──────────
    key_padding_mask = attention_mask.bool()
    for block in model.adapter:
        x = block(x, bias_per_head=None, key_padding_mask=key_padding_mask)
    x = model.adapter_ln(x)                                # (1, T, 640)

    # ── Mean-pool over real tokens (here, all tokens are real) ───────────────
    valid  = attention_mask.float().unsqueeze(-1)
    pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)  # (1, 640)

    # ── Biophysical fusion (placeholder zeros -- generator has no real seq yet)
    if model.bio_proj is not None:
        bio_norm = bio_zero.to(device=pooled.device, dtype=pooled.dtype)
        pooled   = torch.cat([pooled, model.bio_proj(bio_norm)], dim=-1)

    logits = model.head(pooled)                            # (1, 2)
    score  = torch.softmax(logits, dim=-1)[0, 1]
    return score


# =============================================================================
# Scoring discrete sequences (for the refinement step + post-decode check)
# =============================================================================

def score_discrete_sequence(
    seq: str,
    model: RNAFMHybridClassifier,
    tokenizer,
    bio_zero: torch.Tensor,
    device: str,
) -> float:
    """Score a discrete RNA via the normal tokenized forward path."""
    enc = tokenizer(seq, return_tensors="pt", padding=False, truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(device)
    att_mask  = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits, _ = model(input_ids, att_mask, labels=None,
                          Lhat_stack=None, bio_features=bio_zero.to(device))
        return float(torch.softmax(logits, dim=-1)[0, 1].item())


def greedy_refine(
    seq:       str,
    model:     RNAFMHybridClassifier,
    tokenizer,
    bio_zero:  torch.Tensor,
    device:    str,
    max_iters: int = 200,
) -> tuple:
    """
    Discrete local search: at each iteration, try mutating every position to
    each other nucleotide; keep the best single-position swap that improves.
    """
    current_seq   = seq
    current_score = score_discrete_sequence(current_seq, model, tokenizer, bio_zero, device)

    for it in range(max_iters):
        best_swap_seq, best_swap_score = current_seq, current_score
        for i in range(len(current_seq)):
            for nt in RNA_BASES:
                if nt == current_seq[i]:
                    continue
                trial = current_seq[:i] + nt + current_seq[i + 1:]
                s = score_discrete_sequence(trial, model, tokenizer, bio_zero, device)
                if s > best_swap_score:
                    best_swap_seq, best_swap_score = trial, s

        if best_swap_score <= current_score + 1e-6:
            break
        current_seq, current_score = best_swap_seq, best_swap_score

    return current_seq, current_score


# =============================================================================
# SeqProp optimization loop (the heart of the generator)
# =============================================================================

def run_seqprop_hybrid(
    model,
    tokenizer,
    W_NT,
    cls_emb,
    eos_emb,
    bio_zero,
    length:         int   = 300,
    steps:          int   = 400,
    lr:             float = 0.08,
    entropy_weight: float = 0.08,
    temp_start:     float = 2.0,
    temp_end:       float = 0.1,
    seed:           int   = 0,
    log_every:      int   = 50,
    device:         str   = "cpu",
    refine:         bool  = True,
    # Optional biological reward weights (struct/cond modes set these to nonzero)
    g4_weight:        float = 0.0,
    repeat_weight:    float = 0.0,
    au_weight:        float = 0.0,
    gc_penalty_weight: float = 0.0,
    gc_min:           float = 0.35,
    gc_max:           float = 0.65,
) -> tuple:
    """
    Single SeqProp run -- returns (sequence, final_score).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    theta = torch.randn(length, N_BASES, device=device, requires_grad=True)
    opt   = torch.optim.Adam([theta], lr=lr)

    best_theta, best_score = None, -1.0

    has_bio_rewards = any(w > 0 for w in (g4_weight, repeat_weight, au_weight, gc_penalty_weight))
    print(f"  SeqProp | L={length}  steps={steps}  lr={lr}  ew={entropy_weight}  "
          f"T {temp_start}->{temp_end}  bio_rewards={has_bio_rewards}", flush=True)

    for step in range(steps):
        temp = temp_start + (temp_end - temp_start) * (step / steps)
        opt.zero_grad(set_to_none=True)

        P_soft = F.softmax(theta / temp, dim=-1)                       # (L, 4)
        score  = _soft_forward_hybrid(
            model, P_soft, W_NT, cls_emb, eos_emb, bio_zero, device,
        )

        # Entropy penalty — prevents collapse to a single nucleotide per position
        P_real  = F.softmax(theta, dim=-1)
        entropy = -(P_real * P_real.clamp(min=1e-9).log()).sum(-1).mean()

        loss = -score + entropy_weight * entropy

        # Optional biological rewards (struct/cond modes)
        if g4_weight > 0:
            loss = loss - g4_weight * _soft_g4_reward(P_soft)
        if repeat_weight > 0:
            loss = loss - repeat_weight * _soft_repeat_reward(P_soft)
        if au_weight > 0:
            loss = loss - au_weight * _soft_au_rich_reward(P_soft)
        if gc_penalty_weight > 0:
            loss = loss + gc_penalty_weight * _soft_gc_balance_penalty(P_soft, gc_min, gc_max)

        loss.backward()
        opt.step()

        sv = float(score.item())
        if sv > best_score and step > steps // 3:
            best_score, best_theta = sv, theta.detach().clone()

        if (step + 1) % log_every == 0 or step == 0:
            print(f"    step {step+1:>4}/{steps}  T={temp:.2f}  "
                  f"score={sv:.4f}  entropy={float(entropy.item()):.3f}",
                  flush=True)

    if best_theta is None:
        best_theta = theta.detach()

    # Argmax decode -> discrete sequence
    idxs = best_theta.argmax(dim=-1).cpu().numpy()
    seq  = "".join(RNA_BASES[i] for i in idxs)
    raw  = score_discrete_sequence(seq, model, tokenizer, bio_zero, device)
    print(f"\n  argmax decode -> score={raw:.4f}", flush=True)

    if refine:
        seq, raw = greedy_refine(seq, model, tokenizer, bio_zero, device)
        print(f"  greedy refine -> score={raw:.4f}", flush=True)

    return seq, raw


# =============================================================================
# Sequence analysis (post-generation quality checks)
# =============================================================================

def analyse_sequence(seq: str) -> dict:
    L = len(seq)
    return {
        "length": L,
        "gc":     (seq.count("G") + seq.count("C")) / max(L, 1),
        "au":     (seq.count("A") + seq.count("U")) / max(L, 1),
        "vienna_fold_score": _vienna_fold_score(seq),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="De novo LLPS RNA generator (hybrid-model SeqProp)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", choices=["seqprop", "struct", "cond"], default="seqprop",
                   help="seqprop = pure classifier; struct = + biological rewards; "
                        "cond = + condensate-specific preset weights")
    p.add_argument("--ckpt",  required=True, help="Hybrid model checkpoint .pt")
    p.add_argument("--length", type=int, default=300, help="Target sequence length (nt)")
    p.add_argument("--num_seqs", type=int, default=5, help="Number of sequences to generate")
    p.add_argument("--steps", type=int, default=400, help="SeqProp optimization steps")
    p.add_argument("--lr",    type=float, default=0.08)
    p.add_argument("--entropy_weight", type=float, default=0.08)
    p.add_argument("--temp_start", type=float, default=2.0)
    p.add_argument("--temp_end",   type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; sequences i use seed + i to encourage diversity")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--no_refine", action="store_true",
                   help="Skip greedy post-decode refinement (~2x faster)")
    p.add_argument("--out_fasta", default="designed_llps_rnas.fasta")

    # Biological rewards (used by struct + cond)
    p.add_argument("--g4_weight",        type=float, default=0.0)
    p.add_argument("--repeat_weight",    type=float, default=0.0)
    p.add_argument("--au_weight",        type=float, default=0.0)
    p.add_argument("--gc_penalty_weight", type=float, default=0.10)
    p.add_argument("--gc_min", type=float, default=0.40)
    p.add_argument("--gc_max", type=float, default=0.65)

    # Condensate preset
    p.add_argument("--condensate", default="stress_granule",
                   help="Condensate-specific reward preset (only for --method cond)")
    p.add_argument("--list_condensates", action="store_true")

    args = p.parse_args()

    if args.list_condensates:
        print("Available condensate presets (--condensate <name>):")
        for n in ["stress_granule", "p_body", "nucleolus", "nuclear_speckle",
                  "paraspeckle", "germ_granule", "cajal_body", "neuronal_rna_granule"]:
            print(f"  {n}")
        return

    # Resolve reward weights based on method
    g4_w, rep_w, au_w = args.g4_weight, args.repeat_weight, args.au_weight
    if args.method == "cond":
        g4_w, rep_w, au_w = _condensate_weights(args.condensate)
        print(f"Condensate preset '{args.condensate}': "
              f"g4={g4_w:.2f}  repeat={rep_w:.2f}  au={au_w:.2f}")

    # GC penalty active for struct + cond, off for pure seqprop
    gc_pw = args.gc_penalty_weight if args.method in ("struct", "cond") else 0.0

    # Setup
    set_seed(args.seed)
    device = setup_device()
    print(f"Loading hybrid model on {device}: {args.ckpt}")
    model     = load_hybrid_for_generation(args.ckpt, device)
    tokenizer = AutoTokenizer.from_pretrained("multimolecule/rnafm", trust_remote_code=True)
    W_NT, cls_emb, eos_emb = get_nt_embeddings(model, device)
    bio_zero = torch.zeros(1, model.args.bio_dim, device=device)

    # Generate
    print(f"\nGenerating {args.num_seqs} sequences of length {args.length} "
          f"with method='{args.method}'\n" + "=" * 60)
    generated = []
    for i in range(args.num_seqs):
        print(f"\n[ Sequence {i+1}/{args.num_seqs} ]  seed={args.seed + i}")
        seq, score = run_seqprop_hybrid(
            model, tokenizer, W_NT, cls_emb, eos_emb, bio_zero,
            length         = args.length,
            steps          = args.steps,
            lr             = args.lr,
            entropy_weight = args.entropy_weight,
            temp_start     = args.temp_start,
            temp_end       = args.temp_end,
            seed           = args.seed + i,
            log_every      = args.log_every,
            device         = device,
            refine         = not args.no_refine,
            g4_weight        = g4_w,
            repeat_weight    = rep_w,
            au_weight        = au_w,
            gc_penalty_weight= gc_pw,
            gc_min           = args.gc_min,
            gc_max           = args.gc_max,
        )
        stats = analyse_sequence(seq)
        generated.append((seq, score, stats))
        print(f"  -> P(LLPS)={score:.4f}  GC={stats['gc']:.3f}  "
              f"AU={stats['au']:.3f}  fold={stats['vienna_fold_score']:.2f}")

    # Write FASTA, sorted by score (highest first)
    generated.sort(key=lambda t: -t[1])
    with open(args.out_fasta, "w") as f:
        for i, (seq, score, stats) in enumerate(generated, 1):
            f.write(f">designed_{i:03d}|method={args.method}|score={score:.4f}|"
                    f"len={stats['length']}|gc={stats['gc']:.3f}|au={stats['au']:.3f}\n")
            f.write(f"{seq}\n")

    print(f"\n{'=' * 60}")
    print(f"Wrote {len(generated)} designed sequences -> {args.out_fasta}")
    print(f"Score range: {generated[-1][1]:.4f} - {generated[0][1]:.4f}")
    print(f"Top-3 scores: {[round(g[1], 4) for g in generated[:3]]}")


if __name__ == "__main__":
    main()
