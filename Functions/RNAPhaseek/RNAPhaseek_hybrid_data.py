"""
RNAPhaseek Hybrid — Dataset & DataLoader
==========================================
Loads raw RNA sequences from FASTA files and tokenises them with the
RNA-FM tokenizer (single nucleotide per token).  FEGS matrices are loaded
from pre-computed .npz files and aligned to the tokenized sequence (padded
for [CLS] and [EOS] special tokens at positions 0 and L+1).

Key differences from RNAPhaseek_data.py:
  - Input sequences are raw strings, not BPE-encoded int arrays.
  - Tokenization happens inside the collate function (batched, on-CPU).
  - FEGS matrices are inserted at positions [1:L+1, 1:L+1] in a
    (topk_m, T_tok, T_tok) tensor to align with [CLS]+sequence+[EOS].
  - Variable-length sequences are padded to the batch maximum inside
    collate_fn, not pre-padded to a global SEQ_LEN.
"""

from __future__ import annotations

import os
import zipfile
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .RNAPhaseek_data  import load_topk_unpadded   # reuse FEGS loader
from .RNAPhaseek_utils import LRUCache

_lru = LRUCache(max_items=128)   # 128 * ~40 MB max = ~5 GB cap (was 4096 → 160+ GB → OOM)


# ── FASTA reader ─────────────────────────────────────────────────────────────

def read_fasta(path: str) -> list[tuple[str, str]]:
    """Return list of (header, sequence) pairs from a FASTA file."""
    records: list[tuple[str, str]] = []
    header, parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts).upper().replace("T", "U")))
                header = line[1:]
                parts  = []
            elif line:
                parts.append(line)
    if header is not None:
        records.append((header, "".join(parts).upper().replace("T", "U")))
    return records


# ── FEGS loading (raw, unpadded to SEQ_LEN) ──────────────────────────────────

def load_fegs_raw(npz_path: str, L: int, topk: int) -> np.ndarray:
    """
    Load FEGS matrices for a sequence of length L.
    Returns (topk, L, L) float32 array (standardised within the L×L region).
    Uses an LRU cache keyed by (path, L).
    """
    key = (npz_path, L, topk)
    cached = _lru.get(key)
    if cached is not None:
        return cached
    arr = load_topk_unpadded(npz_path, ell=L, T=L, topk=topk)   # (topk, L, L)
    _lru.put(key, arr)
    return arr


# ── Dataset ───────────────────────────────────────────────────────────────────

class HybridRNADataset(Dataset):
    """
    Each item: (seq_string, npz_path, label, bio_row)
    Sequences are raw RNA strings (ACGU).  No tokenisation here —
    tokenisation is batched in collate_fn for efficiency.
    """

    def __init__(
        self,
        sequences:   list[str],
        npz_paths:   list[str],
        labels:      np.ndarray,
        bio_array:   Optional[np.ndarray] = None,
        max_nucleotides: int = 1022,
    ):
        assert len(sequences) == len(npz_paths) == len(labels)
        self.seqs    = [s[:max_nucleotides] for s in sequences]
        self.paths   = npz_paths
        self.labels  = labels.astype(np.int64)
        self.bio     = bio_array
        self.max_len = max_nucleotides

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        bio = self.bio[idx] if self.bio is not None else None
        return (
            self.seqs[idx],
            self.paths[idx],
            int(self.labels[idx]),
            bio,
        )


# ── Collate function ─────────────────────────────────────────────────────────

def make_collate_fn(tokenizer, topk_m: int, fp16_bias: bool = False):
    """
    Returns a collate_fn closed over the RNA-FM tokenizer and topk_m.

    For each batch:
      1. Tokenize all sequences (adds [CLS], [EOS], pads to batch max length)
      2. Build Lhat_padded: [B, topk_m, T_tok, T_tok]
         The FEGS matrix for sequence i (L_i nucleotides) is inserted at
         positions [1:L_i+1, 1:L_i+1] so it aligns with the nucleotide tokens.
         Position 0 = [CLS], position L_i+1 = [EOS]; both get zero bias.
    """
    bias_dtype = torch.float16 if fp16_bias else torch.float32

    def collate_fn(batch):
        seqs, paths, labels, bios = zip(*batch)
        B = len(seqs)

        # ── Tokenise ─────────────────────────────────────────────────────────
        enc = tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        token_ids  = enc["input_ids"]           # [B, T_tok]
        att_mask   = enc["attention_mask"]       # [B, T_tok]
        T_tok      = token_ids.shape[1]

        # ── FEGS bias ─────────────────────────────────────────────────────────
        Lhat_padded = torch.zeros(B, topk_m, T_tok, T_tok, dtype=bias_dtype)
        for i, (seq, path) in enumerate(zip(seqs, paths)):
            L = len(seq)                          # nucleotide count (no special tokens)
            if not os.path.exists(path):
                continue
            try:
                raw = load_fegs_raw(path, L=min(L, T_tok - 2), topk=topk_m)
            except (zipfile.BadZipFile, OSError, ValueError, KeyError) as e:
                # Corrupt .npz — skip this example's bias rather than crash training.
                print(f"[WARN] Skipping corrupt .npz: {path} ({type(e).__name__}: {e})", flush=True)
                continue
            # raw: (topk_m, L_use, L_use) where L_use = min(L, T_tok-2)
            L_use = raw.shape[1]
            # Insert at [1 : L_use+1, 1 : L_use+1]  (offset by 1 for [CLS])
            Lhat_padded[i, :, 1:L_use + 1, 1:L_use + 1] = torch.tensor(raw, dtype=bias_dtype)

        # ── Bio features ──────────────────────────────────────────────────────
        if bios[0] is not None:
            Bio = torch.tensor(np.stack(bios, axis=0), dtype=torch.float32)
        else:
            Bio = None

        Labels = torch.tensor(labels, dtype=torch.long)

        return token_ids, att_mask, Lhat_padded, Bio, Labels

    return collate_fn


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_dataloaders(
    seqs_tr:  list[str],
    seqs_va:  list[str],
    paths_tr: list[str],
    paths_va: list[str],
    y_tr:     np.ndarray,
    y_va:     np.ndarray,
    tokenizer,
    args,       # HybridTrainArgs
    bio_tr:     Optional[np.ndarray] = None,
    bio_va:     Optional[np.ndarray] = None,
):
    train_ds = HybridRNADataset(seqs_tr, paths_tr, y_tr, bio_tr, args.max_nucleotides)
    val_ds   = HybridRNADataset(seqs_va, paths_va, y_va, bio_va, args.max_nucleotides)

    # Weighted sampler to balance positive/negative ratio
    counts  = np.bincount(y_tr, minlength=2).astype(float)
    weights = np.array([1.0 / counts[int(c)] for c in y_tr], dtype=float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    collate = make_collate_fn(tokenizer, topk_m=args.topk_m, fp16_bias=args.fp16_bias)
    pin     = False   # MPS does not support pin_memory

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
        collate_fn=collate,
    )
    return train_loader, val_loader
