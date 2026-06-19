"""
RNAPhaseek Hybrid Classifier
==============================
RNA-FM backbone  +  FEGSTrans adapter stack  +  biophysical feature fusion

Architecture
------------
1. RNA-FM encoder (frozen by default)
      Nucleotide-level tokens → 640-dim contextual representations [B, L+2, 640]
2. FEGSTrans adapter layers (2 × trainable transformer blocks at 640-dim)
      Same HeadMixture + learnable β graph-bias as in RNAPhaseek.
      Injects RNA secondary-structure information from FEGS eigenvalue matrices
      directly into the attention scores of the adapter layers.
3. Mean pooling (over real tokens, excluding padding)
4. Biophysical branch (26 features: RNA2PS + ENCORI)
      LayerNorm → Linear → GELU, fused via concatenation
5. Classification head
      Linear(768, 256) → GELU → Dropout → Linear(256, 2)

Input
-----
  token_ids       : (B, T)          long — RNA-FM token IDs, padded
  attention_mask  : (B, T)          long — 1=real token, 0=padding
  Lhat_stack      : (B, m, T, T)    float — FEGS bias (padded for CLS/EOS)
  bio_features    : (B, 26)         float — optional biophysical features
  labels          : (B,)            long  — 0=non-LLPS, 1=LLPS  (optional)

Output
------
  logits : (B, 2)
  loss   : scalar or None
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .RNAPhaseek         import HeadMixture, FEGSTrans, Block, Config
from .RNAPhaseek_hybrid_config import HybridTrainArgs, RNA_FM_DIM


class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (DANN): identity on the forward pass, gradient * -lambda on
    the backward pass. Placed before the organism head so minimizing organism loss pushes the
    shared representation to become organism-INVARIANT."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class RNAFMHybridClassifier(nn.Module):
    """RNA-FM + FEGSTrans adapter hybrid for RNA LLPS prediction."""

    def __init__(self, args: HybridTrainArgs):
        super().__init__()
        self.args          = args
        self.label_smooth  = args.label_smooth
        self.topk_m        = args.topk_m

        # ── Backbone (RNA-FM by default; ERNIE-RNA = base-pairing-aware alternative) ──
        import multimolecule  # noqa: F401 — registers multimolecule/* models with transformers
        from transformers import AutoModel
        print(f"Loading backbone: {args.backbone} ...", flush=True)
        if "ernie" in args.backbone.lower():
            # ERNIE-RNA needs a compat shim (input_embeds typo) and its own 28-token tokenizer
            # (AutoModel's default tokenizer trips a vocab-size check). See ernierna_compat.
            from .ernierna_compat import load_ernierna
            self.backbone = load_ernierna(args.backbone)
        else:
            self.backbone = AutoModel.from_pretrained(args.backbone, trust_remote_code=True)
        # Backbone hidden size drives the adapter / pooling / head widths (RNA-FM 640, ERNIE-RNA 768).
        self.bb_dim = int(self.backbone.config.hidden_size)
        assert self.bb_dim % args.n_heads == 0, f"hidden_size {self.bb_dim} not divisible by n_heads {args.n_heads}"
        self._nan_guard = "ernie" in args.backbone.lower()   # clamp ERNIE's intermittent long-seq overflow

        # Freeze entire backbone first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optionally unfreeze the last N encoder layers (fine-tuning mode)
        if not args.freeze_backbone or args.unfreeze_last_n > 0:
            n = args.unfreeze_last_n
            if n == 0:
                # unfreeze everything
                for p in self.backbone.parameters():
                    p.requires_grad = True
                print(f"  Backbone fully unfrozen ({sum(p.numel() for p in self.backbone.parameters()):,} params)")
            else:
                layers = self._get_encoder_layers()
                for layer in layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                unfrozen = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
                print(f"  Unfrozen last {n} backbone layers ({unfrozen:,} params)")

        # ── FEGSTrans adapter ─────────────────────────────────────────────────
        # Reuses the same Block and Config from RNAPhaseek.py,
        # but operating at RNA-FM's 640-dim instead of 256-dim.
        adapter_cfg = Config(
            n_embd      = self.bb_dim,
            n_head      = args.n_heads,
            block_size  = args.max_nucleotides + 2,   # +2 for CLS + EOS
            attn_pdrop  = args.attn_pdrop,
            resid_pdrop = args.resid_pdrop,
            embd_pdrop  = 0.0,
            causal      = False,
            use_graph_bias = True,
        )
        self.adapter    = nn.ModuleList([Block(adapter_cfg) for _ in range(args.n_adapter_layers)])
        self.adapter_ln = nn.LayerNorm(self.bb_dim)

        # HeadMixture: learns weighted mix of top-k FEGS matrices per attention head
        self.mixer = HeadMixture(m=args.topk_m, n_heads=args.n_heads, tau=1.5, l2_delta=1e-4)

        # ── Biophysical branch ────────────────────────────────────────────────
        if args.bio_dim > 0:
            bio_hidden    = max(self.bb_dim // 4, args.bio_dim)
            self.bio_proj = nn.Sequential(
                nn.LayerNorm(args.bio_dim),
                nn.Linear(args.bio_dim, bio_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            head_in = self.bb_dim + bio_hidden
        else:
            self.bio_proj = None
            head_in = self.bb_dim

        # ── Species embedding (optional; for multi-species training) ──────────
        # Off by default so existing single-species checkpoints still load with
        # strict=False. When enabled, contributes args.species_dim to head input.
        self.use_species_embed = bool(getattr(args, "use_species_embed", False))
        if self.use_species_embed:
            self.species_emb = nn.Embedding(args.n_species, args.species_dim)
            head_in         += args.species_dim
        else:
            self.species_emb = None

        # Sanity: embedding size must match the registry so checkpoints transfer cleanly.
        if self.use_species_embed:
            from .species_registry import N_SPECIES as _REG_N
            assert args.n_species >= _REG_N, (
                f"n_species ({args.n_species}) < registry N_SPECIES ({_REG_N}); "
                "species IDs from species_id_for() may exceed the embedding table."
            )

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

        # ── Domain-adversarial organism head (optional; off by default = v6-safe) ──
        # Gradient-reversal -> the backbone+bio representation becomes organism-invariant,
        # forcing reliance on the LLPS mechanism rather than yeast-specific cues.
        self.adv_organism = bool(getattr(args, "adv_organism", False))
        if self.adv_organism:
            self.adv_lambda = float(getattr(args, "adv_lambda", 1.0))
            n_org = int(getattr(args, "n_organisms", 2))
            self.org_head = nn.Sequential(
                nn.Linear(head_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, n_org))
        else:
            self.org_head = None

        # Initialise only the newly added parameters (not the backbone)
        self._init_new_weights()

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_train:,}", flush=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_encoder_layers(self) -> nn.ModuleList:
        """Return the list of encoder transformer layers from the backbone."""
        # ESM-style (RNA-FM, RNA-BERT)
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        # BertModel-style
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
            return self.backbone.encoder.layers
        # GPT-style
        if hasattr(self.backbone, "transformer") and hasattr(self.backbone.transformer, "h"):
            return self.backbone.transformer.h
        raise AttributeError(
            f"Cannot locate encoder layers in {type(self.backbone).__name__}. "
            "Check `unfreeze_last_n` logic and update `_get_encoder_layers()`."
        )

    def _init_new_weights(self):
        """Xavier/normal init for all non-backbone parameters."""
        modules_to_init = [
            *self.adapter,
            self.adapter_ln,
            self.mixer,
            self.head,
        ]
        if self.bio_proj is not None:
            modules_to_init.append(self.bio_proj)
        if self.species_emb is not None:
            modules_to_init.append(self.species_emb)
        if self.org_head is not None:
            modules_to_init.append(self.org_head)

        for m in modules_to_init:
            for name, p in m.named_parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                elif "weight" in name and p.dim() == 1:
                    nn.init.ones_(p)    # LayerNorm weights
                else:
                    nn.init.zeros_(p)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        token_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
        Lhat_stack:     Optional[torch.Tensor] = None,
        bio_features:   Optional[torch.Tensor] = None,
        species_id:     Optional[torch.Tensor] = None,
        organism_labels: Optional[torch.Tensor] = None,
    ):
        # 1. Backbone encoding → [B, L_tok, bb_dim]
        out = self.backbone(input_ids=token_ids, attention_mask=attention_mask)
        x   = out.last_hidden_state                          # [B, T, bb_dim]
        # ERNIE-RNA's recursive cross-layer pairing bias can overflow to NaN/inf on certain long
        # sequences (intermittent, both CPU/MPS). Sequences are independent in a batch, so clamp
        # per-position non-finite values to 0 — keeps the forward finite (offending seq -> ~zero rep)
        # instead of poisoning the whole batch. Identity (no-op) for RNA-FM, which never overflows.
        if self._nan_guard:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. FEGSTrans adapter
        key_padding_mask = attention_mask.bool()             # True = real token
        reg_mixture = x.new_zeros(1).squeeze()
        Bh = None

        if Lhat_stack is not None:
            Lhat_stack = Lhat_stack.to(device=x.device, dtype=x.dtype)
            Bh, _pi, reg = self.mixer(Lhat_stack)           # [B, nH, T, T]
            reg_mixture  = reg

        for block in self.adapter:
            x = block(x, bias_per_head=Bh, key_padding_mask=key_padding_mask)
        x = self.adapter_ln(x)                              # [B, T, 640]

        # 3. Mean pool — exclude padding; CLS and EOS are included (minimal effect)
        valid  = attention_mask.float().unsqueeze(-1)        # [B, T, 1]
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)  # [B, 640]

        # 4. Biophysical fusion
        if self.bio_proj is not None and bio_features is not None:
            bio    = bio_features.to(device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, self.bio_proj(bio)], dim=-1)

        # 4b. Species embedding fusion (multi-species training only)
        # If use_species_embed=True but species_id wasn't provided (e.g., legacy
        # inference path), default to 'unknown' (id=7) so the head's input shape
        # stays consistent. The model will still produce a valid prediction; the
        # caller should provide species_id for best accuracy.
        if self.species_emb is not None:
            if species_id is None:
                from .species_registry import SPECIES_TO_ID as _S2I
                default_id = _S2I.get("unknown", 0)
                species_id = pooled.new_full((pooled.size(0),), default_id, dtype=torch.long)
            sp_vec = self.species_emb(species_id.to(device=pooled.device, dtype=torch.long))
            pooled = torch.cat([pooled, sp_vec.to(dtype=pooled.dtype)], dim=-1)

        # 5. Classify
        logits = self.head(pooled)                           # [B, 2]
        loss   = None
        if labels is not None:
            loss = (
                F.cross_entropy(logits, labels, label_smoothing=self.label_smooth)
                + reg_mixture
            )
            # Domain-adversarial organism term: gradient reversal makes `pooled` organism-invariant.
            if self.org_head is not None and organism_labels is not None:
                org_logits = self.org_head(grad_reverse(pooled, self.adv_lambda))
                loss = loss + F.cross_entropy(org_logits, organism_labels.to(org_logits.device))
        return logits, loss

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self, args: HybridTrainArgs):
        """
        Two param groups:
          - backbone (unfrozen layers): very low lr, moderate weight decay
          - adapter + head + bio_proj + mixer: higher lr, moderate weight decay
        Biases and LayerNorm parameters are always excluded from weight decay.
        """
        backbone_params, adapter_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append((name, p))
            else:
                adapter_params.append((name, p))

        def split_decay(param_list):
            decay, no_decay = [], []
            for name, p in param_list:
                # Exclude from weight decay: biases, LayerNorm gains/biases,
                # and embedding tables (species_emb.weight) — convention is
                # to never apply weight decay to embedding lookup tables.
                is_embedding = name.endswith(".weight") and ("species_emb" in name or "embeddings" in name)
                if p.dim() <= 1 or name.endswith(".bias") or is_embedding:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay   = split_decay(backbone_params)
        ad_decay, ad_no_decay   = split_decay(adapter_params)

        optim_groups = [
            {"params": ad_decay,   "lr": args.lr,         "weight_decay": args.weight_decay},
            {"params": ad_no_decay,"lr": args.lr,         "weight_decay": 0.0},
        ]
        if bb_decay or bb_no_decay:
            optim_groups += [
                {"params": bb_decay,   "lr": args.backbone_lr, "weight_decay": args.weight_decay},
                {"params": bb_no_decay,"lr": args.backbone_lr, "weight_decay": 0.0},
            ]

        return torch.optim.AdamW(optim_groups, betas=(0.9, 0.95))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
