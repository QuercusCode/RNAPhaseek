"""
Config for the full-sequence (multi-window) hybrid model.
"""

from dataclasses import dataclass

RNA_FM_MODEL    = "multimolecule/rnafm"
RNA_FM_DIM      = 640
RNA_FM_N_LAYERS = 12

from Functions.RNA_biophysical.biophysical_features import N_FEATURES
BIO_DIM = N_FEATURES   # RNA2PS + ENCORI + complexity + repeat + self-complementarity (38)

# Data paths -- reuse the unified positives/negatives that the original hybrid
# trained on, so val splits stay directly comparable to Phase 1's 0.7707.
FASTA_POS = "Data/raw/all_positives_dedup.fasta"
FASTA_NEG = "Data/raw/negatives_ensembl.fasta"
BIO_POS   = "Data/splits/biophys_pos.npy"
BIO_NEG   = "Data/splits/biophys_neg.npy"
BIO_NORM  = "Data/splits/biophys_norm_stats.npz"

BEST_CKPT  = "model/hybrid_fullseq_best.pt"
FINAL_CKPT = "model/hybrid_fullseq_final.pt"


@dataclass
class HybridFullSeqArgs:
    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone:           str   = RNA_FM_MODEL

    # ── Adapter ──────────────────────────────────────────────────────────────
    n_adapter_layers:   int   = 2
    n_heads:            int   = 8
    attn_pdrop:         float = 0.10
    resid_pdrop:        float = 0.10

    # ── Biophysical ──────────────────────────────────────────────────────────
    bio_dim:            int   = BIO_DIM

    # ── Windowing ────────────────────────────────────────────────────────────
    # RNA-FM tokenizer adds [CLS]+[EOS], so the model sees `window`+2 tokens.
    # Stride 512 gives 50% overlap. max_windows caps memory for very long RNAs.
    window:             int   = 1022      # nucleotides per window
    stride:             int   = 512       # 50% overlap
    max_windows:        int   = 32        # caps coverage to window + (max_windows-1)*stride nt
                                          # = 1022 + 31*512 = 17094 nt max coverage
                                          # (covers 99% of training RNAs; only the longest
                                          # NEAT1-class positives, ~1%, get truncated)

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:         int   = 1         # B=1, max_windows=32 -> up to 32 windows / batch
                                          # (B=2 OOM'd on jetsam with max_windows=32)
    num_workers:        int   = 0         # macOS: MUST be 0
    epochs:             int   = 60
    lr:                 float = 2e-4
    weight_decay:       float = 0.01
    warmup_frac:        float = 0.10
    label_smooth:       float = 0.05
    patience:           int   = 12

    # ── Paths ─────────────────────────────────────────────────────────────────
    fasta_pos:  str = FASTA_POS
    fasta_neg:  str = FASTA_NEG
    bio_pos:    str = BIO_POS
    bio_neg:    str = BIO_NEG
    bio_norm:   str = BIO_NORM
    best_ckpt:  str = BEST_CKPT
    final_ckpt: str = FINAL_CKPT
