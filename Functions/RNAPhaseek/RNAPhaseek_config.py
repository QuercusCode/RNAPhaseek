"""
RNAPhaseek Model Configuration
================================
Mirrors Phaseek_v3_config.py but tuned for RNA:
  - Larger block_size (1024) to handle longer ncRNAs / lncRNAs
  - Slightly wider model (n_embd=256) to compensate for simpler nucleotide alphabet
  - Same training hyper-parameters as Phaseek v3
"""

from dataclasses import dataclass, field

# ── Data paths (override at run-time or via CLI) ──────────────────────────────
SRC_POS = "Data/processed/fegs_topk_pos"
SRC_NEG = "Data/processed/fegs_topk_neg"

# Biophysical feature arrays (RNA2PS + ENCORI, 26 features each)
BIO_POS  = "Data/splits/biophys_pos.npy"
BIO_NEG  = "Data/splits/biophys_neg.npy"
BIO_NORM = "Data/splits/biophys_norm_stats.npz"
from Functions.RNA_biophysical.biophysical_features import N_FEATURES
BIO_DIM  = N_FEATURES   # must match N_FEATURES in biophysical_features.py (38)

# ── FEGS graph bias ───────────────────────────────────────────────────────────
TOPK_M  = 10      # top-k RNA-FEGS motif matrices used as graph bias

# ── Sequence / model dimensions ───────────────────────────────────────────────
SEQ_LEN  = 1024   # max token length; lncRNAs can be long
N_LAYERS = 6
D_MODEL  = 256    # wider than Phaseek (192) to compensate for smaller alphabet
N_HEADS  = 8      # must divide D_MODEL evenly

# ── Data loading ──────────────────────────────────────────────────────────────
BATCH_SIZE   = 8   # smaller batch because sequences can be longer
NUM_WORKERS  = 2
PREFETCH     = 2
FP16_BIAS    = True

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS       = 25
LR           = 6e-4
WEIGHT_DECAY = 0.1
WARMUP_FRAC  = 0.05
LABEL_SMOOTH = 0.05

# ── Checkpoints ───────────────────────────────────────────────────────────────
BEST_CKPT  = "model/rna_phaseek_best.pt"
FINAL_CKPT = "model/rna_phaseek_final.pt"


@dataclass
class TrainArgs:
    src_pos:    str   = SRC_POS
    src_neg:    str   = SRC_NEG
    bio_pos:    str   = BIO_POS
    bio_neg:    str   = BIO_NEG
    bio_norm:   str   = BIO_NORM
    bio_dim:    int   = BIO_DIM
    topk_m:     int   = TOPK_M
    seq_len:    int   = SEQ_LEN
    n_layers:   int   = N_LAYERS
    d_model:    int   = D_MODEL
    n_heads:    int   = N_HEADS
    batch_size: int   = BATCH_SIZE
    num_workers:int   = NUM_WORKERS
    prefetch:   int   = PREFETCH
    fp16_bias:  bool  = FP16_BIAS
    epochs:     int   = EPOCHS
    lr:         float = LR
    weight_decay: float = WEIGHT_DECAY
    warmup_frac: float  = WARMUP_FRAC
    label_smooth: float = LABEL_SMOOTH
    best_ckpt:  str   = BEST_CKPT
    final_ckpt: str   = FINAL_CKPT
