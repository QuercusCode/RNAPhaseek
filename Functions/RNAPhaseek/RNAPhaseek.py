"""
RNAPhaseek Transformer Model
==============================
Ported directly from Phaseek_v3.py (TransformerClassifier + HeadMixture + FEGSTrans).
All protein-specific logic is removed; only hyper-parameter defaults are changed.

The architecture is identical to Phaseek v3:
  - Token + positional embedding
  - N Transformer blocks each with:
      FEGSTrans: multi-head self-attention + learnable graph-bias term (β × Lhat_per_head)
      MLP (GELU, 4× expansion)
  - HeadMixture: learns a weighted mixture of the top-k RNA-FEGS graph matrices
    per attention head
  - Global mean-pooling (non-causal) → linear classification head

Input
-----
  idx         : (B, T) long tensor of BPE-tokenised RNA k-mers
  targets     : (B,)   long tensor of labels (1=LLPS, 0=non-LLPS)
  Lhat_stack  : (B, m, T, T) float tensor of top-k RNA-FEGS eigenvalue matrices

Output
------
  logits : (B, 2)
  loss   : scalar (cross-entropy + label smoothing + HeadMixture L2 reg)
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── HeadMixture ───────────────────────────────────────────────────────────────
class HeadMixture(nn.Module):
    """
    Learns a per-head weighted mixture over m RNA-FEGS graph matrices.
    Produces a bias tensor of shape (B, n_heads, T, T) for FEGSTrans.
    """
    def __init__(self, m: int, n_heads: int, tau: float = 1.5, l2_delta: float = 1e-4):
        super().__init__()
        self.m, self.nh, self.tau = m, n_heads, tau
        self.alpha = nn.Parameter(torch.zeros(m))           # shared logits
        self.delta = nn.Parameter(torch.zeros(n_heads, m))  # head-specific offsets
        self.l2_delta = l2_delta

    def forward(self, Lhat_stack):
        """
        Lhat_stack : (B, m, T, T)
        Returns    : Bh (B, n_heads, T, T), pi (n_heads, m), reg scalar
        """
        B, m, T, _ = Lhat_stack.shape
        logits = self.alpha.unsqueeze(0) + self.delta          # (n_heads, m)
        pi     = torch.softmax(logits / self.tau, dim=-1)      # (n_heads, m)
        pi     = pi.to(Lhat_stack.dtype).to(Lhat_stack.device)
        Bh     = torch.einsum("hm,bmij->hbij", pi, Lhat_stack) # (n_heads, B, T, T)
        Bh     = Bh.permute(1, 0, 2, 3).contiguous()           # (B, n_heads, T, T)
        reg    = self.l2_delta * (self.delta ** 2).sum()
        return Bh, pi, reg


# ── FEGSTrans (attention with learnable graph bias) ───────────────────────────
class FEGSTrans(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj  = nn.Linear(config.n_embd,     config.n_embd)
        self.attn_dropout  = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.causal         = getattr(config, "causal",         False)
        self.use_graph_bias = getattr(config, "use_graph_bias", True)
        self.beta           = nn.Parameter(torch.tensor(0.1))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
              .view(1, 1, config.block_size, config.block_size)
        )

    @torch.no_grad()
    def _masked_zscore(self, bm, key_mask):
        dtype_orig = bm.dtype
        bm  = bm.float()
        B, H, T, _ = bm.shape
        row_mask = key_mask[:, None, :, None]
        col_mask = key_mask[:, None, None, :]
        valid    = row_mask & col_mask
        if H > 1:
            valid = valid.expand(-1, H, -1, -1)
        eps       = 1e-6
        bm_valid  = torch.where(valid, bm, torch.zeros_like(bm))
        count     = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        mean      = bm_valid.sum(dim=(-2, -1), keepdim=True) / count
        var       = torch.where(valid, (bm - mean) ** 2,
                                torch.zeros_like(bm)).sum(dim=(-2, -1), keepdim=True) / count
        std       = var.sqrt().clamp_min(eps)
        return ((bm - mean) / std).to(dtype_orig)

    def forward(self, x, bias_per_head=None, key_padding_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head

        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

        if self.use_graph_bias and bias_per_head is not None:
            bm = bias_per_head.to(att.dtype).to(att.device)
            if key_padding_mask is not None:
                bm = self._masked_zscore(bm, key_padding_mask)
            att = att + self.beta * bm

        if key_padding_mask is not None:
            att = att.masked_fill(~key_padding_mask[:, None, None, :T], -1e4)
        if self.causal:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, -1e4)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ── Transformer Block ─────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = FEGSTrans(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, bias_per_head=None, key_padding_mask=None):
        x = x + self.attn(self.ln_1(x),
                          bias_per_head=bias_per_head,
                          key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# ── Config helper ─────────────────────────────────────────────────────────────
class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# ── RNAPhaseek Transformer Classifier ────────────────────────────────────────
class RNAPhaseekClassifier(nn.Module):
    """
    Drop-in replacement of TransformerClassifier from Phaseek_v3.py,
    renamed and documented for RNA LLPS prediction.
    """

    def __init__(
        self,
        config: Config,
        topk_m: int,
        label_smooth: float = 0.05,
        weight_decay: float = 0.1,
        bio_dim: int = 0,
    ):
        super().__init__()
        self.config        = config
        self.label_smooth  = label_smooth
        self.weight_decay  = weight_decay
        self.bio_dim       = bio_dim

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.mixer = HeadMixture(m=topk_m, n_heads=config.n_head, tau=1.5, l2_delta=1e-4)

        # Biophysical fusion: project bio features → d_model/4, then concatenate.
        # A BN layer normalises the raw (z-scored) bio features before projection,
        # ensuring stable gradient flow regardless of feature scale.
        if bio_dim > 0:
            bio_hidden = max(config.n_embd // 4, bio_dim)
            self.bio_proj = nn.Sequential(
                nn.LayerNorm(bio_dim),
                nn.Linear(bio_dim, bio_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            head_in = config.n_embd + bio_hidden
        else:
            self.bio_proj = None
            head_in = config.n_embd

        self.head = nn.Linear(head_in, 2)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("attn.c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, Lhat_stack=None, bio_features=None):
        B, T = idx.size()
        pos  = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x    = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        key_padding_mask = (idx != 0)   # True = real token

        reg_mixture = torch.tensor(0.0, device=idx.device)
        Bh = None
        if Lhat_stack is not None:
            Lhat_stack = Lhat_stack.to(x.device, non_blocking=True)
            Bh, pi, reg = self.mixer(Lhat_stack)
            reg_mixture = reg

        for block in self.transformer.h:
            x = block(x, bias_per_head=Bh, key_padding_mask=key_padding_mask)
        x = self.transformer.ln_f(x)

        # Global mean pool over non-padding positions
        valid  = key_padding_mask.float().unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        # Fuse biophysical features (RNA2PS + ENCORI) if provided
        if self.bio_proj is not None and bio_features is not None:
            bio = bio_features.to(pooled.device, dtype=pooled.dtype)
            bio_out = self.bio_proj(bio)             # (B, bio_hidden)
            pooled  = torch.cat([pooled, bio_out], dim=-1)  # (B, n_embd + bio_hidden)

        logits = self.head(pooled)
        loss   = None
        if targets is not None:
            loss = (
                F.cross_entropy(logits, targets, label_smoothing=self.label_smooth)
                + reg_mixture * 1.0
            )
        return logits, loss

    def configure_optimizers(self, lr, betas=(0.9, 0.95), weight_decay=None):
        if weight_decay is None:
            weight_decay = self.weight_decay
        decay, no_decay = set(), set()
        for name, module in self.named_modules():
            for pname, _ in module.named_parameters(recurse=False):
                full = f"{name}.{pname}" if name else pname
                if pname.endswith("bias"):
                    no_decay.add(full)
                elif isinstance(module, nn.LayerNorm):
                    no_decay.add(full)
                else:
                    decay.add(full)
        for emb in ["transformer.wte.weight", "transformer.wpe.weight"]:
            decay.discard(emb)
            no_decay.add(emb)
        param_dict  = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)],    "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
