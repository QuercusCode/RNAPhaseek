"""
RNA BPE Tokenizer Training & Sequence Encoding
================================================
Trains a Byte-Pair Encoding (BPE) tokenizer on all RNA sequences in the
dataset, then encodes every split (train/val/test) into zero-padded integer
arrays that the RNAPhaseek DataLoader expects.

What this script produces
--------------------------
  model/rna_bpe_tokenizer.json    — full HuggingFace tokenizer config
  model/rna_bpe_vocab.json        — token → id mapping  (used by runner + generator)
  Data/splits/pos_seq_encoded.npy — (N_pos, SEQ_LEN) int32 array
  Data/splits/neg_seq_encoded.npy — (N_neg, SEQ_LEN) int32 array
  Data/splits/train_pos_encoded.npy  etc.  — per-split arrays

BPE on RNA sequences
---------------------
The 4-character RNA alphabet (A, U, G, C) is too small for meaningful BPE
merges at the character level. Instead each nucleotide is first separated by
a space so BPE sees individual characters as initial units:

    "AUGCAUGC"  →  "A U G C A U G C"

BPE then merges the most frequent adjacent pairs (AU, GC, AUG, …) up to
vocab_size = 512, giving the model k-mer-level tokens without losing the
ability to fall back to single nucleotides. Single nucleotides A, U, G, C
are always present as tokens (required by the generator's SeqProp logic).

Special tokens
--------------
  0  [PAD]   — padding
  1  [UNK]   — unknown / degenerate nucleotide
  2  [BOS]   — beginning of sequence  (not inserted during encoding)
  3  [EOS]   — end of sequence        (not inserted during encoding)

Usage
-----
  # Full pipeline: train tokenizer + encode all splits
  python Functions/tokenizer_training.py

  # Just train the tokenizer (no encoding)
  python Functions/tokenizer_training.py --train_only

  # Just encode (tokenizer already trained)
  python Functions/tokenizer_training.py --encode_only

  # Inspect vocab
  python Functions/tokenizer_training.py --show_vocab --top_k 30
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from Bio import SeqIO

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── HuggingFace tokenizers ─────────────────────────────────────────────────────
try:
    from tokenizers import Tokenizer, AddedToken
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    _TOKENIZERS_OK = True
except ImportError:
    _TOKENIZERS_OK = False
    print("[!] 'tokenizers' library not found.  pip install tokenizers")

# ── Paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR   = ROOT / "Data"  / "splits"
MODEL_DIR    = ROOT / "model"
TOK_JSON     = MODEL_DIR / "rna_bpe_tokenizer.json"
VOCAB_JSON   = MODEL_DIR / "rna_bpe_vocab.json"

SEQ_LEN      = 1024
VOCAB_SIZE   = 512

# Special tokens (fixed IDs, must match wte padding_idx=0)
PAD_TOK = "[PAD]"
UNK_TOK = "[UNK]"
BOS_TOK = "[BOS]"
EOS_TOK = "[EOS]"
SPECIAL_TOKENS = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK]

RNA_BASES = "AUGC"


# =============================================================================
# Helpers
# =============================================================================

def _normalise(seq: str) -> str:
    """Uppercase, T→U, keep only AUGC (replace others with the UNK character N)."""
    seq = seq.upper().replace("T", "U")
    return "".join(c if c in RNA_BASES else "N" for c in seq)


def _to_spaced(seq: str) -> str:
    """
    Convert 'AUGCAUGC' → 'A U G C A U G C'
    This makes each nucleotide an independent initial BPE unit.
    """
    return " ".join(seq)


def collect_fasta_sequences(paths: list) -> list:
    """
    Load sequences from one or more FASTA files.
    Returns list of normalised RNA strings.
    """
    seqs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  [skip] not found: {p}")
            continue
        for rec in SeqIO.parse(str(p), "fasta"):
            s = _normalise(str(rec.seq))
            if len(s) >= 10:
                seqs.append(s)
    return seqs


# =============================================================================
# Step 1 — Train BPE tokenizer
# =============================================================================

def train_tokenizer(
    train_seqs: list,
    vocab_size:  int  = VOCAB_SIZE,
    out_json:    Path = TOK_JSON,
    vocab_json:  Path = VOCAB_JSON,
) -> "Tokenizer":
    """
    Train a BPE tokenizer on the supplied RNA sequences and save it.

    Parameters
    ----------
    train_seqs : list of str  (normalised RNA sequences)
    vocab_size : int          (target vocabulary size, incl. special tokens)
    out_json   : Path         (full tokenizer config output)
    vocab_json : Path         (simple token→id dict for fast lookup)
    """
    if not _TOKENIZERS_OK:
        raise RuntimeError("Install tokenizers:  pip install tokenizers")

    print(f"\nTraining BPE tokenizer on {len(train_seqs):,} sequences "
          f"(target vocab_size={vocab_size}) ...")

    # ── Build tokenizer ───────────────────────────────────────────────────────
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOK))
    tokenizer.pre_tokenizer = Whitespace()   # splits on spaces → individual chars

    trainer = BpeTrainer(
        vocab_size          = vocab_size,
        special_tokens      = SPECIAL_TOKENS,
        min_frequency       = 2,
        show_progress       = True,
        initial_alphabet    = list(RNA_BASES) + ["N"],   # ensure single-nt tokens
    )

    # Feed spaced sequences to the trainer (via an iterator)
    def _seq_iter():
        for s in train_seqs:
            yield _to_spaced(s)

    tokenizer.train_from_iterator(_seq_iter(), trainer=trainer,
                                   length=len(train_seqs))

    # ── Verify single-nucleotide tokens are present ───────────────────────────
    vocab = tokenizer.get_vocab()
    for nt in RNA_BASES:
        if nt not in vocab:
            print(f"  [!] WARNING: nucleotide '{nt}' not in vocab — adding manually")
            tokenizer.add_tokens([nt])
            vocab = tokenizer.get_vocab()

    # ── Save ──────────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_json))
    print(f"Tokenizer saved → {out_json}")

    # Save simple vocab dict (token → id) — used by runner.py and generator.py
    with open(vocab_json, "w") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)
    print(f"Vocab dict saved → {vocab_json}  ({len(vocab)} tokens)")

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nVocab statistics:")
    print(f"  Total tokens : {len(vocab)}")
    print(f"  Special tokens: {SPECIAL_TOKENS}")
    for nt in RNA_BASES:
        print(f"  '{nt}' → id {vocab.get(nt, 'MISSING')}")

    return tokenizer


# =============================================================================
# Step 2 — Encode sequences
# =============================================================================

def encode_sequences(
    seqs:       list,
    tokenizer:  "Tokenizer",
    seq_len:    int  = SEQ_LEN,
    pad_id:     int  = 0,
    show_stats: bool = True,
) -> np.ndarray:
    """
    Encode a list of RNA strings to a (N, seq_len) int32 array.

    Each sequence is:
      1. Converted to spaced format ("A U G C …")
      2. BPE tokenised → variable-length list of token IDs
      3. Truncated to seq_len (if longer)
      4. Zero-padded on the right (if shorter)

    Parameters
    ----------
    seqs       : list of normalised RNA strings
    tokenizer  : trained HuggingFace Tokenizer
    seq_len    : maximum sequence length in tokens
    pad_id     : padding token ID (default 0 = [PAD])

    Returns
    -------
    np.ndarray of shape (N, seq_len), dtype int32
    """
    N   = len(seqs)
    out = np.full((N, seq_len), fill_value=pad_id, dtype=np.int32)

    lengths = []
    for i, seq in enumerate(tqdm(seqs, desc="Encoding", ncols=90, leave=False)):
        spaced  = _to_spaced(seq[:seq_len * 2])   # rough pre-truncation
        enc     = tokenizer.encode(spaced)
        ids     = enc.ids[:seq_len]
        out[i, :len(ids)] = ids
        lengths.append(len(ids))

    if show_stats:
        lengths = np.array(lengths)
        print(f"  Encoded {N:,} sequences")
        print(f"  Token length — min:{lengths.min()}  "
              f"mean:{lengths.mean():.1f}  "
              f"median:{np.median(lengths):.0f}  "
              f"max:{lengths.max()}  "
              f"truncated:{(lengths >= seq_len).sum()}")

    return out


# =============================================================================
# Step 3 — Encode and save all splits
# =============================================================================

def encode_and_save_splits(
    tokenizer:   "Tokenizer",
    splits_dir:  Path = SPLITS_DIR,
    seq_len:     int  = SEQ_LEN,
):
    """
    For every (train/val/test) × (pos/neg) FASTA pair, encode sequences
    and save as .npy arrays next to the FASTA files.

    Also saves combined pos_seq_encoded.npy and neg_seq_encoded.npy
    (all splits merged) for convenience when calling RNAPhaseek_train.py.
    """
    splits = ["train", "val", "test"]
    labels = ["pos", "neg"]

    all_pos, all_neg = [], []

    for split in splits:
        for label in labels:
            fasta = splits_dir / f"{split}_{label}.fasta"
            npy   = splits_dir / f"{split}_{label}_encoded.npy"

            if not fasta.exists():
                print(f"  [skip] {fasta.name} not found")
                continue

            seqs = collect_fasta_sequences([fasta])
            if not seqs:
                print(f"  [skip] {fasta.name} is empty")
                continue

            print(f"\nEncoding {split}/{label} ({len(seqs):,} sequences)...")
            arr = encode_sequences(seqs, tokenizer, seq_len=seq_len)
            np.save(str(npy), arr)
            print(f"  Saved → {npy}  shape={arr.shape}")

            if label == "pos":
                all_pos.append(arr)
            else:
                all_neg.append(arr)

    # Save merged arrays (used directly by RNAPhaseek_train.py)
    if all_pos:
        merged = np.vstack(all_pos)
        out    = splits_dir / "pos_seq_encoded.npy"
        np.save(str(out), merged)
        print(f"\nMerged positive set saved → {out}  shape={merged.shape}")

    if all_neg:
        merged = np.vstack(all_neg)
        out    = splits_dir / "neg_seq_encoded.npy"
        np.save(str(out), merged)
        print(f"Merged negative set saved  → {out}  shape={merged.shape}")


# =============================================================================
# Utilities
# =============================================================================

def load_tokenizer(tok_json: Path = TOK_JSON) -> "Tokenizer":
    """Load a previously saved tokenizer."""
    if not _TOKENIZERS_OK:
        raise RuntimeError("Install tokenizers:  pip install tokenizers")
    return Tokenizer.from_file(str(tok_json))


def show_vocab(tok_json: Path = VOCAB_JSON, top_k: int = 30):
    """Print the most common k-mer tokens learned by BPE."""
    with open(tok_json) as f:
        vocab = json.load(f)

    special = set(SPECIAL_TOKENS)
    kmer_tokens = {t: i for t, i in vocab.items()
                   if t not in special and not t.startswith("##")}

    by_length = sorted(kmer_tokens.items(), key=lambda x: (-len(x[0]), x[1]))
    print(f"\nTop-{top_k} longest k-mer tokens learned by BPE:")
    print(f"{'Token':<20} {'ID':>6}")
    print("─" * 28)
    for tok, tid in by_length[:top_k]:
        print(f"{tok:<20} {tid:>6}")

    all_lengths = [len(t) for t in kmer_tokens]
    if all_lengths:
        print(f"\nk-mer length distribution:")
        for k in range(1, max(all_lengths) + 1):
            cnt = all_lengths.count(k)
            bar = "█" * (cnt // max(1, max(all_lengths.count(l)
                          for l in all_lengths) // 40))
            print(f"  {k}-mer : {cnt:>5}  {bar}")


def verify_roundtrip(tokenizer: "Tokenizer", n: int = 5):
    """Sanity-check: encode then decode a few RNA sequences."""
    test_seqs = [
        "AUGCAUGCAUGCGGGGAUGCAUGCGGGG",
        "AUGUUUAAAGGGUUUCCCAAAGGG",
        "GCGCGCGCAUAUAUAUGGGGCCCC",
    ][:n]
    print("\nRoundtrip verification:")
    for seq in test_seqs:
        spaced  = _to_spaced(seq)
        enc     = tokenizer.encode(spaced)
        decoded = tokenizer.decode(enc.ids).replace(" ", "")
        ok      = "✓" if decoded == seq else "✗"
        print(f"  {ok}  {seq[:30]:<30} → {len(enc.ids)} tokens")


# =============================================================================
# CLI
# =============================================================================

def _collect_all_fasta(splits_dir: Path) -> list:
    """Collect all .fasta sequences from splits dir for tokenizer training."""
    fastas = list(splits_dir.glob("*.fasta"))
    if not fastas:
        # Fallback: check parent Data/ dir
        fastas = list(splits_dir.parent.glob("**/*.fasta"))
    return fastas


def main():
    p = argparse.ArgumentParser(
        description="Train RNA BPE tokenizer and encode sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vocab_size",   type=int,  default=VOCAB_SIZE)
    p.add_argument("--seq_len",      type=int,  default=SEQ_LEN)
    p.add_argument("--splits_dir",   type=str,  default=str(SPLITS_DIR))
    p.add_argument("--tok_json",     type=str,  default=str(TOK_JSON))
    p.add_argument("--vocab_json",   type=str,  default=str(VOCAB_JSON))
    p.add_argument("--train_only",   action="store_true",
                   help="Only train tokenizer, skip encoding")
    p.add_argument("--encode_only",  action="store_true",
                   help="Only encode sequences (tokenizer must already exist)")
    p.add_argument("--show_vocab",   action="store_true",
                   help="Print learned k-mer vocabulary and exit")
    p.add_argument("--top_k",        type=int,  default=30,
                   help="Number of tokens to show with --show_vocab")
    args = p.parse_args()

    splits_dir = Path(args.splits_dir)
    tok_json   = Path(args.tok_json)
    vocab_json = Path(args.vocab_json)

    # ── Show vocab mode ───────────────────────────────────────────────────────
    if args.show_vocab:
        if not vocab_json.exists():
            print(f"[!] Vocab not found: {vocab_json}. Train the tokenizer first.")
            sys.exit(1)
        show_vocab(vocab_json, top_k=args.top_k)
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    if not args.encode_only:
        fastas = _collect_all_fasta(splits_dir)
        if not fastas:
            print(f"[!] No FASTA files found under {splits_dir}")
            print("    Run data_preparation.py first to build the splits.")
            sys.exit(1)

        print(f"Found {len(fastas)} FASTA files for tokenizer training:")
        for f in sorted(fastas):
            print(f"  {f.name}")

        all_seqs = collect_fasta_sequences(fastas)
        print(f"\nTotal sequences for tokenizer training: {len(all_seqs):,}")

        if not all_seqs:
            print("[!] No sequences found. Check your FASTA files.")
            sys.exit(1)

        tokenizer = train_tokenizer(
            all_seqs,
            vocab_size  = args.vocab_size,
            out_json    = tok_json,
            vocab_json  = vocab_json,
        )
        verify_roundtrip(tokenizer)
    else:
        if not tok_json.exists():
            print(f"[!] Tokenizer not found: {tok_json}")
            print("    Run without --encode_only first.")
            sys.exit(1)
        print(f"Loading existing tokenizer from {tok_json} ...")
        tokenizer = load_tokenizer(tok_json)

    # ── Encode ────────────────────────────────────────────────────────────────
    if not args.train_only:
        print("\nEncoding all splits...")
        encode_and_save_splits(
            tokenizer   = tokenizer,
            splits_dir  = splits_dir,
            seq_len     = args.seq_len,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
