"""Utility functions for RNAPhaseek training pipeline."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device() -> str:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS")
        return "mps"
    print("Using CPU")
    return "cpu"


def true_len_ell(seq_ids: np.ndarray) -> int:
    """Return number of non-padding tokens (pad token = 0)."""
    return int((seq_ids != 0).sum())


def standardize_and_pad(M: np.ndarray, ell: int, T: int) -> np.ndarray:
    """
    Crop or zero-pad a (L, L) FEGS eigenvalue matrix to (T, T),
    then clip to actual sequence length ell×ell and standardise.
    """
    L = M.shape[0]
    out = np.zeros((T, T), dtype=np.float32)
    end = min(L, T, ell)
    if end > 0:
        out[:end, :end] = M[:end, :end]
    # Z-score normalise over non-zero region
    region = out[:ell, :ell]
    mu, std = region.mean(), region.std()
    if std > 1e-6:
        out[:ell, :ell] = (region - mu) / std
    return out


def list_npz_sorted(directory: str) -> list:
    """Return sorted list of .npz file paths in directory."""
    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".npz") and not f.startswith("._")  # skip macOS ._ AppleDouble files (exFAT)
    ]
    return files


class LRUCache:
    """Simple LRU cache backed by an ordered dict."""
    def __init__(self, max_items: int = 256):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.max   = max_items

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max:
            self.cache.popitem(last=False)
