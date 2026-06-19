"""
RNAPhaseek Dataset & DataLoader
=================================
Ported from Phaseek_v3_data.py.
Loads BPE-tokenised RNA sequences + pre-computed RNA-FEGS .npz files.
"""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .RNAPhaseek_utils import true_len_ell, standardize_and_pad, LRUCache

TOPK_M   = 10
SEQ_LEN  = 1024
FP16_BIAS = False  # fp16 overflows when learned beta * Lhat > 65k -> NaN logits

_lru = LRUCache(max_items=128)   # 128 x ~22 MB = ~3 GB cap (was 2048 -> up to 45 GB OOM risk)


def load_topk_unpadded(npz_path: str, ell: int, T: int, topk: int = TOPK_M) -> np.ndarray:
    """Load top-k RNA-FEGS eigenvalue matrices from a .npz file and pad to (T, T)."""
    try:
      z_file = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        print(f"\n[WARN] Skipping corrupt .npz: {npz_path} ({e})")
        return np.zeros((topk, T, T), dtype=np.float32)
    with z_file as z:
        mats   = []
        m_keys = sorted([k for k in z.files if k.startswith("M") and k[1:].isdigit()],
                        key=lambda k: int(k[1:]))
        for k in m_keys[:topk]:
            M = np.asarray(z[k], dtype=np.float32)
            if M.ndim == 2:
                M = np.nan_to_num(M)
                mats.append(M)
        # fallback: unnamed arrays
        if not mats:
            for k in z.files:
                M = np.asarray(z[k], dtype=np.float32)
                if M.ndim == 2 and M.size > 0:
                    mats.append(np.nan_to_num(M))
            if len(mats) > topk:
                scores = [float(np.linalg.norm(M, ord="fro")) for M in mats]
                mats   = [mats[i] for i in np.argsort(scores)[::-1][:topk]]
        while len(mats) < topk:
            mats.append(np.zeros((1, 1), dtype=np.float32))

    stack = np.stack(
        [standardize_and_pad(M, ell=ell, T=T) for M in mats[:topk]], axis=0
    )   # (topk, T, T)
    return stack


def build_Lhat_stack(npz_path: str, ell: int, T: int = SEQ_LEN, topk: int = TOPK_M) -> np.ndarray:
    key    = (npz_path, T)
    cached = _lru.get(key)
    if cached is not None:
        return cached
    stack  = load_topk_unpadded(npz_path, ell=ell, T=T, topk=topk)
    _lru.put(key, stack)
    return stack


class RNASeqTopkDataset(Dataset):
    def __init__(
        self,
        seq_array:   np.ndarray,
        label_array: np.ndarray,
        npz_paths:   np.ndarray,
        bio_array:   np.ndarray = None,   # (N, 26) biophysical features, optional
    ):
        assert seq_array.shape[0] == label_array.shape[0] == npz_paths.shape[0]
        self.seq   = seq_array.astype(np.int64, copy=False)
        self.lab   = label_array.astype(np.int64, copy=False)
        self.paths = npz_paths.astype(str)
        self.bio   = bio_array  # may be None

    def __len__(self): return self.seq.shape[0]

    def __getitem__(self, idx):
        bio = self.bio[idx] if self.bio is not None else None
        return self.seq[idx], int(self.lab[idx]), self.paths[idx], bio


def collate_fn(batch):
    seqs, labs, paths, bios = zip(*batch)
    seqs   = np.stack(seqs, axis=0)
    ells   = [true_len_ell(seqs[i]) for i in range(seqs.shape[0])]
    stacks = [build_Lhat_stack(paths[i], ell=ells[i], T=SEQ_LEN, topk=TOPK_M)
              for i in range(len(paths))]
    Lhat   = torch.tensor(
        np.stack(stacks, axis=0),
        dtype=torch.float16 if FP16_BIAS else torch.float32,
    )
    # Biophysical features (RNA2PS + ENCORI)
    if bios[0] is not None:
        Bio = torch.tensor(np.stack(bios, axis=0), dtype=torch.float32)
    else:
        Bio = None
    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(labs, dtype=torch.long),
        Lhat,
        Bio,
    )


def make_dataloaders(
    Xseq_tr, Xseq_va,
    paths_tr, paths_va,
    y_tr, y_va,
    batch_size, num_workers, prefetch, device,
    bio_tr=None, bio_va=None,
) -> Tuple[DataLoader, DataLoader]:
    pin = device == "cuda"

    train_ds = RNASeqTopkDataset(Xseq_tr, y_tr, paths_tr, bio_array=bio_tr)
    val_ds   = RNASeqTopkDataset(Xseq_va, y_va, paths_va, bio_array=bio_va)

    counts  = np.bincount(y_tr, minlength=2).astype(float)
    weights = np.array([1.0 / counts[c] for c in y_tr], dtype=float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # prefetch_factor is only valid when num_workers > 0 (PyTorch ≥ 2.x)
    pf = prefetch if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0), prefetch_factor=pf,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=(num_workers > 0), prefetch_factor=pf,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
