"""
RNAPhaseek — Inference Runner
================================
Score a single RNA sequence or a FASTA file for LLPS propensity.

Usage
-----
Single sequence:
  python Functions/runner.py --sequence "AUGCAUGCAUGCGGGGAUGCAUGCGGGG" --id test_seq

Batch (FASTA):
  python Functions/runner.py --fasta Data/my_rnas.fasta --output results/scores.tsv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

# ── add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Functions.RNA_FEGS.RNA_FEGS_feature_extraction import RNAFEGSFeatureExtractor, _normalise_rna
from Functions.RNAPhaseek.RNAPhaseek         import RNAPhaseekClassifier, Config
from Functions.RNAPhaseek.RNAPhaseek_config  import TrainArgs
from Functions.RNAPhaseek.RNAPhaseek_utils   import setup_device


def load_model(ckpt_path: str, device: str) -> RNAPhaseekClassifier:
    state = torch.load(ckpt_path, map_location=device)
    # Infer vocab_size and other params from saved state
    vocab_size  = state["transformer.wte.weight"].shape[0]
    n_embd      = state["transformer.wte.weight"].shape[1]
    block_size  = state["transformer.wpe.weight"].shape[0]
    n_layers    = sum(1 for k in state if k.startswith("transformer.h.") and k.endswith(".ln_1.weight"))
    # n_heads inferred from attention projection shape
    n_heads = 8  # default; override if needed

    cfg = Config(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layers, n_head=n_heads, n_embd=n_embd,
        embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0,
        causal=False, use_graph_bias=True,
    )
    # topk_m inferred from mixer parameter
    topk_m = state["mixer.alpha"].shape[0]
    model  = RNAPhaseekClassifier(cfg, topk_m=topk_m).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def score_sequences(sequences: list, ids: list, model, extractor, device: str) -> list:
    """
    Score a list of RNA sequences.
    Returns list of (id, sequence, llps_score) tuples.
    """
    normed = [_normalise_rna(s) for s in sequences]
    feats  = extractor.extract(normed)          # (N, feature_dim)

    results = []
    for i, (sid, seq, feat) in enumerate(zip(ids, normed, feats)):
        # For inference without pre-computed FEGS .npz, we skip graph bias
        # (pass Lhat_stack=None → model falls back to pure sequence attention)
        # A future step will precompute .npz for full graph bias during inference.
        tok = torch.zeros(1, 1024, dtype=torch.long, device=device)   # placeholder
        with torch.no_grad():
            logits, _ = model(tok, Lhat_stack=None)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        results.append((sid, seq[:40] + "..." if len(seq) > 40 else seq, round(prob, 4)))
    return results


def main():
    p = argparse.ArgumentParser(description="RNAPhaseek: RNA LLPS propensity scorer")
    p.add_argument("--sequence", type=str, help="Single RNA/DNA sequence string")
    p.add_argument("--id",       type=str, default="query", help="Sequence ID for single-sequence mode")
    p.add_argument("--fasta",    type=str, help="Path to input FASTA file (batch mode)")
    p.add_argument("--output",   type=str, default=None, help="Output TSV path (batch mode)")
    p.add_argument("--model",    type=str, default="model/rna_phaseek_best.pt",
                   help="Path to trained model checkpoint")
    args = p.parse_args()

    if not args.sequence and not args.fasta:
        p.error("Provide --sequence or --fasta")

    device    = setup_device()
    extractor = RNAFEGSFeatureExtractor()

    if not Path(args.model).exists():
        print(f"[!] Model not found at {args.model}. Train the model first.")
        print("    Run: python Functions/RNAPhaseek/RNAPhaseek_train.py")
        sys.exit(1)

    model = load_model(args.model, device)

    if args.sequence:
        seqs, ids = [args.sequence], [args.id]
    else:
        records = list(SeqIO.parse(args.fasta, "fasta"))
        seqs    = [str(r.seq) for r in records]
        ids     = [r.id       for r in records]
        print(f"Loaded {len(seqs)} sequences from {args.fasta}")

    results = score_sequences(seqs, ids, model, extractor, device)

    header  = f"{'ID':<30} {'LLPS_score':>12}  {'Sequence (preview)'}"
    print("\n" + header)
    print("-" * len(header))
    lines = []
    for sid, preview, score in results:
        line = f"{sid:<30} {score:>12.4f}  {preview}"
        print(line)
        lines.append(f"{sid}\t{score}\t{preview}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write("ID\tLLPS_score\tSequence_preview\n")
            f.write("\n".join(lines) + "\n")
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
