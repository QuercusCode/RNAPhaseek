"""
predict_hybrid.py
==================
End-to-end LLPS prediction for new (unlabeled) RNA sequences using a trained
RNAPhaseek hybrid model.

  - Reads a FASTA of RNA sequences
  - Computes FEGS bias matrices on the fly
  - Computes 26-dim RNA2PS+ENCORI biophysical features on the fly
  - Loads a hybrid checkpoint (Phase 1 or Phase 2)
  - Writes a TSV: header, length, prob_llps, call (using the saved threshold)

Usage
-----
  # Phase 1 (best AUROC, balanced):
  python predict_hybrid.py --fasta my_rnas.fasta \
      --ckpt model/phase1/hybrid_best.pt \
      --thresh_file model/phase1/hybrid_best_thresh.npy \
      --out predictions.tsv

  # Phase 2 (best sensitivity, screening):
  python predict_hybrid.py --fasta my_rnas.fasta \
      --ckpt model/hybrid_best.pt \
      --thresh_file model/hybrid_best_thresh.npy \
      --out predictions.tsv

  # Override the saved threshold (e.g. for stricter calls):
  python predict_hybrid.py --fasta my_rnas.fasta --ckpt model/phase1/hybrid_best.pt \
      --thresh 0.50 --out predictions.tsv

The script writes its own .npz files for FEGS into a temp dir; pass --keep_temp
if you want to inspect them afterwards.
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import multimolecule  # noqa: F401 — registers RnaFmModel/RnaTokenizer

from transformers import AutoTokenizer

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Functions.RNAPhaseek.RNAPhaseek_hybrid       import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import load_fegs_raw
from Functions.RNAPhaseek.RNAPhaseek_utils         import setup_device
from Functions.precompute_fegs                     import process_fasta  as fegs_process
from Functions.RNA_FEGS.RNA_FEGS_feature_extraction import DEFAULT_MOTIF_GROUPS
from Functions.RNA_biophysical                     import RNABiophysicalExtractor

VALID_NT = set("ACGUN")


def normalise(s: str) -> str:
    s = s.upper().replace("T", "U")
    s = re.sub(r"[^AUGCN]", "N", s)
    return s


def read_fasta(path: str):
    records = []
    hdr, parts = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr is not None:
                    records.append((hdr, normalise("".join(parts))))
                hdr, parts = line[1:], []
            else:
                parts.append(line)
        if hdr is not None:
            records.append((hdr, normalise("".join(parts))))
    return records


def compute_biophys(seqs):
    extractor = RNABiophysicalExtractor(normalize=False)
    return np.stack([extractor._compute_one(s) for s in seqs], axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Predict LLPS probabilities for unlabeled RNA sequences")
    p.add_argument("--fasta",        required=True, help="Input FASTA of RNA sequences")
    p.add_argument("--ckpt",         required=True, help="Hybrid model checkpoint .pt")
    p.add_argument("--out",          default="predictions.tsv", help="Output TSV path")
    p.add_argument("--thresh",       type=float, default=None,
                   help="Decision threshold for binary call (default: use saved F1-optimal)")
    p.add_argument("--thresh_file",  default="",
                   help=".npy with saved {f1: float, mcc: float} thresholds")
    p.add_argument("--bio_norm",     default="Data/splits/biophys_norm_stats.npz",
                   help="Z-score normalisation stats (mean/std) from training set")
    p.add_argument("--backbone",     default="multimolecule/rnafm")
    p.add_argument("--n_adapter_layers", type=int, default=2)
    p.add_argument("--n_heads",      type=int, default=8)
    p.add_argument("--topk_m",       type=int, default=10)
    p.add_argument("--max_nucleotides", type=int, default=1022)
    p.add_argument("--batch_size",   type=int, default=4)
    # Sliding-window inference for sequences longer than 1022 nt.
    # When --window_mode is "first" (default), only the first 1022 nt are used (model's training regime).
    # When "slide", the RNA is sliced into overlapping windows; per-window scores are aggregated.
    p.add_argument("--window_mode",  choices=["first", "slide"], default="first",
                   help="'first' = score only first 1022 nt (default, matches training). "
                        "'slide' = score overlapping 1022-nt windows, then aggregate via --window_pool.")
    p.add_argument("--window_stride", type=int, default=512,
                   help="Stride between consecutive windows (only used with --window_mode slide)")
    p.add_argument("--window_pool",  choices=["max", "mean"], default="max",
                   help="How to aggregate per-window probabilities. 'max' (default) flags an RNA as "
                        "LLPS+ if ANY window scores high; 'mean' averages.")
    p.add_argument("--no_bio",       action="store_true", help="Skip biophysical branch")
    p.add_argument("--keep_temp",    action="store_true", help="Don't delete temp FEGS files")
    args = p.parse_args()

    # ── 1. Load FASTA ─────────────────────────────────────────────────────────
    records = read_fasta(args.fasta)
    if not records:
        print(f"No sequences in {args.fasta}", file=sys.stderr)
        sys.exit(1)
    headers = [h for h, _ in records]
    seqs    = [s for _, s in records]
    print(f"Loaded {len(seqs)} sequences from {args.fasta}")

    # ── 1b. Sliding-window expansion (optional) ───────────────────────────────
    # In 'slide' mode, we expand each RNA into overlapping 1022-nt windows,
    # run the model on each, then aggregate per-RNA. The model itself only
    # ever sees 1022-nt inputs (its training regime).
    if args.window_mode == "slide":
        window = args.max_nucleotides
        stride = args.window_stride
        flat_seqs    = []
        flat_parent  = []   # parent index in `seqs` for each window
        for i, s in enumerate(seqs):
            if len(s) <= window:
                flat_seqs.append(s)
                flat_parent.append(i)
            else:
                start = 0
                while start < len(s):
                    flat_seqs.append(s[start : start + window])
                    flat_parent.append(i)
                    if start + window >= len(s):
                        break
                    start += stride
        print(f"Sliding-window mode: {len(seqs)} RNAs -> {len(flat_seqs)} windows "
              f"(window={window}, stride={stride}, pool='{args.window_pool}')")
        # Build a new FASTA in a tmp dir that FEGS can read.
        expanded_dir = tempfile.mkdtemp(prefix="rnaphaseek_windows_")
        expanded_fasta = os.path.join(expanded_dir, "windows.fasta")
        with open(expanded_fasta, "w") as f:
            for w, p in zip(flat_seqs, flat_parent):
                f.write(f">parent_{p}\n{w}\n")
        # For FEGS + forward, replace seqs with flat_seqs and remember parent.
        fegs_fasta = expanded_fasta
        forward_seqs = flat_seqs
    else:
        flat_parent  = list(range(len(seqs)))
        fegs_fasta   = args.fasta
        forward_seqs = seqs
        expanded_dir = None

    # ── 2. Resolve threshold ──────────────────────────────────────────────────
    if args.thresh is not None:
        thresh = float(args.thresh)
        thresh_source = "user --thresh"
    elif args.thresh_file and os.path.exists(args.thresh_file):
        th = np.load(args.thresh_file, allow_pickle=True).item()
        thresh = float(th.get("f1", 0.5))
        thresh_source = f"F1-optimal from {args.thresh_file}"
    else:
        thresh = 0.5
        thresh_source = "default 0.5"
    print(f"Decision threshold: {thresh:.3f}  ({thresh_source})")

    # ── 3. Compute FEGS (and biophysical) on the fly ──────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="rnaphaseek_predict_")
    try:
        fegs_dir = Path(tmpdir) / "fegs"
        print(f"Computing FEGS matrices -> {fegs_dir}/ ...")
        fegs_pairs = fegs_process(
            fasta_path   = Path(fegs_fasta),
            out_dir      = fegs_dir,
            topk         = args.topk_m,
            seq_len      = 1024,
            workers      = max(1, os.cpu_count() - 2),
            overwrite    = True,
            motif_groups = DEFAULT_MOTIF_GROUPS,
            start_idx    = 0,
        )
        npz_paths = sorted(str(p) for p in fegs_dir.glob("*.npz"))
        assert len(npz_paths) == len(forward_seqs), \
            f"FEGS count {len(npz_paths)} != forward_seqs {len(forward_seqs)}"

        bio_features = None
        if not args.no_bio:
            print(f"Computing biophysical features ...")
            # Compute biophys on the FULL parent sequence (matches training).
            # In slide mode, replicate per-parent biophys to each window of that parent.
            bio_raw_parent = compute_biophys(seqs)   # (N_parent, 26)
            if os.path.exists(args.bio_norm):
                ns = np.load(args.bio_norm)
                m  = ns["mean"].astype(np.float32)
                s  = ns["std"].astype(np.float32).clip(min=1e-8)
                bio_parent_n = (bio_raw_parent - m) / s
            else:
                print(f"  [WARN] {args.bio_norm} not found -- skipping biophysical normalisation")
                bio_parent_n = bio_raw_parent
            # Replicate to flat (per-window) layout
            bio_features = np.stack([bio_parent_n[p] for p in flat_parent], axis=0)

        # ── 4. Load model ─────────────────────────────────────────────────────
        device = setup_device()
        print(f"Loading model on {device} ...")
        model_args = HybridTrainArgs(
            backbone           = args.backbone,
            freeze_backbone    = True,
            unfreeze_last_n    = 0,
            n_adapter_layers   = args.n_adapter_layers,
            n_heads            = args.n_heads,
            topk_m             = args.topk_m,
            bio_dim            = 26 if bio_features is not None else 0,
            batch_size         = args.batch_size,
            num_workers        = 0,
            fp16_bias          = False,
            max_nucleotides    = args.max_nucleotides,
        )
        model = RNAFMHybridClassifier(model_args).to(device)
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

        # ── 5. Batched forward ────────────────────────────────────────────────
        flat_probs = np.zeros(len(forward_seqs), dtype=np.float32)
        for start in range(0, len(forward_seqs), args.batch_size):
            end       = min(start + args.batch_size, len(forward_seqs))
            batch_seqs = [s[:args.max_nucleotides] for s in forward_seqs[start:end]]
            enc       = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(device)
            att_mask  = enc["attention_mask"].to(device)
            T_tok     = input_ids.shape[1]

            # Build Lhat_padded for this batch
            Lhat_padded = torch.zeros(end - start, args.topk_m, T_tok, T_tok, dtype=torch.float32)
            for i, (seq, path) in enumerate(zip(batch_seqs, npz_paths[start:end])):
                L_use = min(len(seq), T_tok - 2)
                if L_use < 1: continue
                try:
                    raw = load_fegs_raw(path, L=L_use, topk=args.topk_m)
                except Exception as e:
                    print(f"  [WARN] bad FEGS for sample {start+i}: {e}")
                    continue
                L_use = raw.shape[1]
                Lhat_padded[i, :, 1:L_use+1, 1:L_use+1] = torch.tensor(raw, dtype=torch.float32)
            Lhat_padded = Lhat_padded.to(device)

            bio_tensor = None
            if bio_features is not None:
                bio_tensor = torch.tensor(bio_features[start:end], dtype=torch.float32, device=device)

            with torch.no_grad():
                logits, _ = model(input_ids, att_mask, labels=None,
                                  Lhat_stack=Lhat_padded, bio_features=bio_tensor)
                p = torch.softmax(logits, dim=-1)[:, 1]
                p = torch.where(torch.isfinite(p), p, torch.full_like(p, 0.5))
            flat_probs[start:end] = p.cpu().numpy()
            print(f"  {end}/{len(forward_seqs)} done")

    finally:
        if not args.keep_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)
            if expanded_dir is not None:
                shutil.rmtree(expanded_dir, ignore_errors=True)

    # ── 6. Aggregate per-RNA (slide mode) + write output TSV ──────────────────
    n_rna = len(seqs)
    probs = np.zeros(n_rna, dtype=np.float32)
    n_windows = np.zeros(n_rna, dtype=np.int32)
    best_window_idx = np.full(n_rna, -1, dtype=np.int32)   # which window had max prob
    parent_arr = np.array(flat_parent)
    for j, prob in enumerate(flat_probs):
        i = int(parent_arr[j])
        n_windows[i] += 1
        if args.window_pool == "max":
            if prob > probs[i] or best_window_idx[i] < 0:
                probs[i] = prob
                best_window_idx[i] = j
        else:   # mean
            probs[i] += prob   # accumulate, divide after the loop
    if args.window_pool == "mean":
        probs = probs / np.maximum(n_windows, 1).astype(np.float32)

    calls = (probs >= thresh).astype(int)
    with open(args.out, "w") as f:
        if args.window_mode == "slide":
            f.write("header\tlength\tn_windows\tprob_llps\tbest_window_pos\tcall\n")
            for i, (h, s, p, c) in enumerate(zip(headers, seqs, probs, calls)):
                # best_window_pos is the nucleotide start of the best window (slide mode + max pool)
                if args.window_pool == "max" and best_window_idx[i] >= 0:
                    # Count how many windows precede this parent
                    parent_windows = [j for j in range(len(flat_parent)) if flat_parent[j] == i]
                    w_idx = parent_windows.index(int(best_window_idx[i]))
                    pos = w_idx * args.window_stride
                else:
                    pos = -1
                f.write(f"{h}\t{len(s)}\t{n_windows[i]}\t{p:.4f}\t{pos}\t{c}\n")
        else:
            f.write("header\tlength\tprob_llps\tcall\n")
            for h, s, p, c in zip(headers, seqs, probs, calls):
                f.write(f"{h}\t{len(s)}\t{p:.4f}\t{c}\n")
    print(f"\nWrote {n_rna} predictions -> {args.out}")
    print(f"Positive calls (prob >= {thresh:.2f}): {int(calls.sum())} / {n_rna}")


if __name__ == "__main__":
    main()
