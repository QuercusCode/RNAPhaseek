"""
RNAPhaseek Hybrid Full-Sequence Classifier
============================================
Same RNA-FM backbone + FEGSTrans adapter + biophysical fusion as the original
hybrid, but trained AND evaluated on the entire RNA sequence instead of just
the first 1022 nucleotides.

Strategy: sliding-window encoding with max-pooling over window embeddings.

  RNA (length L)
      |
      |  slice into N_w windows of `window` nucleotides each,
      |  stride `stride` (default 512 = 50% overlap)
      v
  N_w windows
      |
      |  tokenize each window with RNA-FM tokenizer
      |  flatten (B, N_w, T_tok) -> (B*N_w, T_tok) for one efficient RNA-FM pass
      v
  RNA-FM backbone (FROZEN, 100M, 640-dim)
      |
      v
  FEGSTrans adapter (TRAINABLE, no graph bias)
      |
      v
  Mean-pool over real tokens per window  ->  (B*N_w, 640)
      |
      |  reshape back to (B, N_w, 640)
      |  ATTENTION-POOL over the window axis: small Linear(640->1) head
      |  produces a per-window score; softmax over valid windows; weighted sum.
      |  Model learns which windows carry LLPS signal; gradient flows to all.
      v
  (B, 640) per-RNA embedding
      |
      |  fuse with biophysical features (computed on FULL RNA, not per-window)
      v
  (B, 800)
      |
      v
  Classifier head: Linear(800,256) -> GELU -> Dropout -> Linear(256,2)

Pooling history (what was tried and why):
  1. MAX-pool over windows: failed (AUROC ~0.5 across 16 epochs). Each backward
     pass only flows gradient through the argmax window per channel, leaving
     ~30 of 32 windows untrained.
  2. MEAN-pool over windows: also failed (AUROC stuck at 0.52 across 4 epochs).
     Equal weighting dilutes LLPS-driving local motifs by ~32x, drowning the
     discriminative signal.
  3. ATTENTION-pool over windows (current): a small Linear(640->1) head
     produces a per-window score. Softmax over valid windows yields weights.
     Weighted sum gives the RNA embedding. Every window receives gradient
     (through the attention softmax), but the model can learn to emphasize
     informative windows. Combines mean-pool's stable gradients with max-pool's
     focus on relevant regions.

Recommended init:
  Load Phase 1's frozen-RNA-FM checkpoint via --init_from. Phase 1 already
  knows how to score a single 1022-nt window; attention-pooling just lets it
  see and combine multiple windows. Should converge in far fewer epochs than
  training from scratch.

FEGS structure bias is DISABLED in this model. The learned beta values were
near zero in the previous Phase 1 and Phase 2 runs ("If beta stays near zero,
the FEGS structure signal is not helping" -- per the project spec), so we drop
the bias and the associated O(L^2) FEGS .npz storage cost.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .RNAPhaseek         import Block, Config
from .RNAPhaseek_hybrid_fullseq_config import HybridFullSeqArgs, RNA_FM_DIM


class RNAFMHybridFullSeq(nn.Module):
    """RNA-FM + FEGSTrans adapter, with window-level max-pool for long RNAs."""

    def __init__(self, args: HybridFullSeqArgs):
        super().__init__()
        self.args         = args
        self.label_smooth = args.label_smooth

        # ── RNA-FM backbone ───────────────────────────────────────────────────
        import multimolecule  # noqa: F401 -- registers RnaFmModel with transformers
        from transformers import AutoModel
        print(f"Loading backbone: {args.backbone} ...", flush=True)
        self.backbone = AutoModel.from_pretrained(args.backbone, trust_remote_code=True)

        # Freeze backbone (unfreezing not exposed in this model -- keep it simple
        # for the first full-seq training run).
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ── FEGSTrans adapter stack (no graph bias, no FEGS) ─────────────────
        adapter_cfg = Config(
            n_embd        = RNA_FM_DIM,
            n_head        = args.n_heads,
            n_layer       = args.n_adapter_layers,
            block_size    = 1024,
            attn_pdrop    = args.attn_pdrop,
            resid_pdrop   = args.resid_pdrop,
            use_graph_bias= False,     # KEY: no FEGS bias in full-seq model
            causal        = False,
        )
        self.adapter    = nn.ModuleList([Block(adapter_cfg) for _ in range(args.n_adapter_layers)])
        self.adapter_ln = nn.LayerNorm(RNA_FM_DIM)

        # ── Window-level attention pool ──────────────────────────────────────
        # Tiny linear head produces a per-window scalar; softmax over valid
        # windows (masked by window_mask) gives weights. Model learns which
        # windows carry LLPS signal. ~640 params.
        self.window_attn = nn.Linear(RNA_FM_DIM, 1, bias=False)

        # ── Biophysical branch ────────────────────────────────────────────────
        head_in = RNA_FM_DIM
        self.bio_proj = None
        if args.bio_dim > 0:
            self.bio_proj = nn.Sequential(
                nn.LayerNorm(args.bio_dim),
                nn.Linear(args.bio_dim, 160),
                nn.GELU(),
            )
            head_in += 160

        # ── Classifier head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

        self._init_new_weights()

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_train:,}", flush=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_new_weights(self):
        modules_to_init = [*self.adapter, self.adapter_ln, self.window_attn, self.head]
        if self.bio_proj is not None:
            modules_to_init.append(self.bio_proj)
        for m in modules_to_init:
            for name, p in m.named_parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                elif "weight" in name and p.dim() == 1:
                    nn.init.ones_(p)
                else:
                    nn.init.zeros_(p)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        token_ids:      torch.Tensor,    # (B, N_w, T_tok)
        attention_mask: torch.Tensor,    # (B, N_w, T_tok)
        window_mask:    torch.Tensor,    # (B, N_w)  True=real window
        labels:         Optional[torch.Tensor] = None,
        bio_features:   Optional[torch.Tensor] = None,
    ):
        B, N_w, T = token_ids.shape

        # Flatten windows into a single batch dimension so RNA-FM processes
        # everything in one efficient call: (B*N_w, T_tok).
        ids_flat  = token_ids.reshape(B * N_w, T)
        mask_flat = attention_mask.reshape(B * N_w, T)

        # ── RNA-FM encoding ───────────────────────────────────────────────────
        out = self.backbone(input_ids=ids_flat, attention_mask=mask_flat)
        x   = out.last_hidden_state                                    # (B*N_w, T, 640)

        # ── FEGSTrans adapter (no graph bias) ─────────────────────────────────
        key_padding_mask = mask_flat.bool()
        for block in self.adapter:
            x = block(x, bias_per_head=None, key_padding_mask=key_padding_mask)
        x = self.adapter_ln(x)                                         # (B*N_w, T, 640)

        # ── Mean-pool over real tokens per window ────────────────────────────
        valid  = mask_flat.float().unsqueeze(-1)                       # (B*N_w, T, 1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)  # (B*N_w, 640)

        # ── Unflatten and attention-pool over the window axis ────────────────
        # Mean-pool failed (signal diluted), max-pool failed (sparse gradient).
        # Attention-pool: model learns per-window importance; all windows get
        # gradient via the softmax-weighted sum.
        pooled = pooled.reshape(B, N_w, RNA_FM_DIM)                    # (B, N_w, 640)
        attn_scores = self.window_attn(pooled).squeeze(-1)             # (B, N_w)
        # Mask padding windows to -inf so they get zero weight after softmax.
        neg_inf = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~window_mask, neg_inf)
        attn_weights = F.softmax(attn_scores, dim=-1)                  # (B, N_w)
        rna_emb = (pooled * attn_weights.unsqueeze(-1)).sum(dim=1)     # (B, 640)

        # ── Biophysical fusion ────────────────────────────────────────────────
        if self.bio_proj is not None and bio_features is not None:
            bio = bio_features.to(device=rna_emb.device, dtype=rna_emb.dtype)
            rna_emb = torch.cat([rna_emb, self.bio_proj(bio)], dim=-1)

        # ── Classify ─────────────────────────────────────────────────────────
        logits = self.head(rna_emb)                                    # (B, 2)
        loss   = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smooth)
        return logits, loss

    # ── Optimizer (single param group, no backbone fine-tuning here) ──────────

    def configure_optimizers(self, args: HybridFullSeqArgs):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay,    "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, betas=(0.9, 0.95))
