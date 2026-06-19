"""
Data pipeline for the full-sequence hybrid model.

Each RNA is sliced into overlapping windows of `window` nucleotides at stride
`stride`, capped at `max_windows`. The collate function builds:

  token_ids       : (B, N_w_max, T_tok)   long  -- RNA-FM token IDs, padded
  attention_mask  : (B, N_w_max, T_tok)   long  -- 1=real token, 0=padding
  window_mask     : (B, N_w_max)          bool  -- True=real window, False=padding
  bio_features    : (B, 26)               float -- per-RNA biophysical features
  labels          : (B,)                  long
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


VALID_NT = set("ACGUN")


def _normalise(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    return re.sub(r"[^AUGCN]", "N", seq)


def read_fasta(path: str) -> list[tuple[str, str]]:
    records, hdr, parts = [], None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr is not None:
                    records.append((hdr, _normalise("".join(parts))))
                hdr, parts = line[1:], []
            else:
                parts.append(line)
        if hdr is not None:
            records.append((hdr, _normalise("".join(parts))))
    return records


def slice_windows(seq: str, window: int, stride: int, max_windows: int) -> list[str]:
    """Slice a sequence into overlapping windows, capped at max_windows."""
    L = len(seq)
    if L == 0:
        return ["A"]   # degenerate; should not happen for real RNA
    if L <= window:
        return [seq]
    windows = []
    start = 0
    while start < L and len(windows) < max_windows:
        chunk = seq[start : start + window]
        windows.append(chunk)
        if start + window >= L:
            break
        start += stride
    return windows


class FullSeqRNADataset(Dataset):
    """One item = one RNA (full sequence + label + bio row)."""

    def __init__(
        self,
        sequences:  list[str],
        labels:     np.ndarray,
        bio_array:  Optional[np.ndarray] = None,
        max_total_nt: int = 22000,   # safety cap on the longest RNA we'll keep
    ):
        assert len(sequences) == len(labels)
        # Cap extreme lengths so a single 200kb RNA can't blow up memory.
        self.seqs   = [s[:max_total_nt] for s in sequences]
        self.labels = labels.astype(np.int64)
        self.bio    = bio_array

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        bio = self.bio[idx] if self.bio is not None else None
        return self.seqs[idx], int(self.labels[idx]), bio


def make_collate_fn(tokenizer, window: int, stride: int, max_windows: int):
    """Returns a collate_fn that windows + tokenizes a batch of RNAs."""

    def collate_fn(batch):
        seqs, labels, bios = zip(*batch)
        B = len(seqs)

        # Window every sequence, track per-RNA window count.
        all_windows: list[list[str]] = []
        counts: list[int] = []
        for seq in seqs:
            ws = slice_windows(seq, window, stride, max_windows)
            all_windows.append(ws)
            counts.append(len(ws))
        N_w_max = max(counts)

        # Pad each sample's window list with a single "A" placeholder so all
        # samples have N_w_max windows. The placeholder tokens will be ignored
        # via window_mask (False for padding windows).
        flat_windows: list[str] = []
        for ws in all_windows:
            flat_windows.extend(ws + ["A"] * (N_w_max - len(ws)))

        # One tokenizer call for the whole (B*N_w_max) flat batch.
        enc = tokenizer(
            flat_windows,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=window + 2,   # CLS + seq + EOS
        )
        T_tok = enc["input_ids"].shape[1]

        token_ids = enc["input_ids"].reshape(B, N_w_max, T_tok)
        att_mask  = enc["attention_mask"].reshape(B, N_w_max, T_tok)

        # Window-validity mask (True for real windows).
        window_mask = torch.zeros(B, N_w_max, dtype=torch.bool)
        for i, n in enumerate(counts):
            window_mask[i, :n] = True

        # Biophysical features (per-RNA).
        if bios[0] is not None:
            bio_t = torch.tensor(np.stack(bios, axis=0), dtype=torch.float32)
        else:
            bio_t = None

        labels_t = torch.tensor(labels, dtype=torch.long)
        return token_ids, att_mask, window_mask, bio_t, labels_t

    return collate_fn


def make_dataloaders(
    seqs_tr:  list[str],
    seqs_va:  list[str],
    y_tr:     np.ndarray,
    y_va:     np.ndarray,
    tokenizer,
    args,
    bio_tr:   Optional[np.ndarray] = None,
    bio_va:   Optional[np.ndarray] = None,
):
    train_ds = FullSeqRNADataset(seqs_tr, y_tr, bio_tr)
    val_ds   = FullSeqRNADataset(seqs_va, y_va, bio_va)

    # Weighted sampler to balance pos/neg in training.
    counts  = np.bincount(y_tr, minlength=2).astype(float)
    weights = np.array([1.0 / counts[int(c)] for c in y_tr], dtype=float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    collate = make_collate_fn(tokenizer, args.window, args.stride, args.max_windows)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=False,
        collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
        collate_fn=collate,
    )
    return train_loader, val_loader
