"""
RNAPhaseek Hybrid Model Configuration
========================================
Configures the RNA-FM + FEGSTrans adapter hybrid classifier.

Backbone: multimolecule/rnafm  (640-dim, 12 layers, 100M params)
Adapter : N × FEGSTrans blocks operating on RNA-FM hidden states (640-dim)
Head    : biophysical fusion + linear classifier

Alternatives to RNA-FM (change `backbone` field):
  "multimolecule/rnabert"                  120-dim, 6 layers, lighter
  "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"  512-dim, DNA/RNA
"""

from dataclasses import dataclass

from .species_registry import N_SPECIES as _REGISTRY_N_SPECIES

# ── Backbone constants ────────────────────────────────────────────────────────
RNA_FM_MODEL    = "multimolecule/rnafm"
RNA_FM_DIM      = 640    # hidden size; must match the chosen backbone
RNA_FM_N_LAYERS = 12     # total encoder layers in the backbone

# ── Sequence limits ───────────────────────────────────────────────────────────
# RNA-FM tokenizer adds [CLS] + [EOS], so total token length = nucleotides + 2.
# Keep at 1022 so total never exceeds 1024 (RNA-FM's positional embedding limit).
MAX_NUCLEOTIDES = 1022

# ── FEGS ──────────────────────────────────────────────────────────────────────
TOPK_M = 10

# ── Biophysical ───────────────────────────────────────────────────────────────
from Functions.RNA_biophysical.biophysical_features import N_FEATURES
BIO_DIM = N_FEATURES   # RNA2PS + ENCORI + complexity + repeat + self-complementarity (38)

# ── Data paths ────────────────────────────────────────────────────────────────
SRC_POS   = "Data/processed/fegs_topk_pos"
SRC_NEG   = "Data/processed/fegs_topk_neg"
FASTA_POS = "Data/raw/all_positives_dedup.fasta"
FASTA_NEG = "Data/raw/negatives_ensembl.fasta"
BIO_POS   = "Data/splits/biophys_pos.npy"
BIO_NEG   = "Data/splits/biophys_neg.npy"
BIO_NORM  = "Data/splits/biophys_norm_stats.npz"
BEST_CKPT  = "model/hybrid_best.pt"
FINAL_CKPT = "model/hybrid_final.pt"


@dataclass
class HybridTrainArgs:
    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone:           str   = RNA_FM_MODEL
    freeze_backbone:    bool  = True   # keep frozen while data < ~1500
    unfreeze_last_n:    int   = 0      # unfreeze last N backbone layers; 0=all frozen
                                       # recommended: 2 once you have >1500 positives

    # ── FEGSTrans adapter (on top of RNA-FM) ──────────────────────────────────
    n_adapter_layers:   int   = 2      # 2 is a good starting point
    n_heads:            int   = 8      # 640 / 8 = 80-dim per head
    attn_pdrop:         float = 0.10
    resid_pdrop:        float = 0.10

    # ── FEGS ──────────────────────────────────────────────────────────────────
    topk_m:             int   = TOPK_M

    # ── Biophysical ───────────────────────────────────────────────────────────
    bio_dim:            int   = BIO_DIM

    # ── Multi-species (off by default for back-compat with existing checkpoints) ─
    n_species:          int   = _REGISTRY_N_SPECIES   # tied to species_registry.SPECIES_TO_ID
    species_dim:        int   = 32     # per-species embedding dim concatenated at the head
    use_species_embed:  bool  = False  # set True for the multi-species training run

    # ── Domain-adversarial organism invariance (item 1; off by default = v6-safe) ─
    adv_organism:       bool  = False  # add gradient-reversal organism head
    adv_lambda:         float = 1.0    # gradient-reversal strength
    n_organisms:        int   = 2      # yeast vs non-yeast

    # ── Sequence ──────────────────────────────────────────────────────────────
    max_nucleotides:    int   = MAX_NUCLEOTIDES

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:         int   = 4      # halved from 8 to cap peak memory on MPS
    num_workers:        int   = 0      # macOS: MUST be 0
    fp16_bias:          bool  = False  # fp16 overflows when learned beta * Lhat > 65k -> NaN logits
    epochs:             int   = 60
    lr:                 float = 2e-4   # adapter + head learning rate
    backbone_lr:        float = 5e-6   # only used when unfreeze_last_n > 0
    weight_decay:       float = 0.01
    warmup_frac:        float = 0.10
    label_smooth:       float = 0.05
    patience:           int   = 12     # AUROC-based early stopping

    # ── Paths ─────────────────────────────────────────────────────────────────
    src_pos:    str = SRC_POS
    src_neg:    str = SRC_NEG
    fasta_pos:  str = FASTA_POS
    fasta_neg:  str = FASTA_NEG
    bio_pos:    str = BIO_POS
    bio_neg:    str = BIO_NEG
    bio_norm:   str = BIO_NORM
    best_ckpt:  str = BEST_CKPT
    final_ckpt: str = FINAL_CKPT
