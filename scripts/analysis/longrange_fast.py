"""DECISIVE long-range experiment (parallelized). Same question as longrange_decisive.py
— does looking beyond RNA-FM's first-1022-nt window improve discrimination of long
(>1022 nt) RNA-self-LLPS sequences? — but the ViennaRNA MFE folding (the bottleneck) is
spread across CPU cores instead of run serially.

Strategies on a balanced long subset (pos vs neg, all >1022 nt):
  production_fullseq : v6 as shipped (deep streams see first 1022nt)            [baseline]
  first_window       : v6 on the standalone first-1022nt window
  max_window         : v6 per tiled window -> MAX     (best single LLPS module)
  mean_window        : v6 mean over windows           (soft MIL)
  noisy_OR           : 1 - prod(1 - p_window)         (MIL "any window")
  v7_MIL             : trained tiling + attention pooling  (learned cross-window)

Caveat: v6/v7 trained on these seqs -> absolute AUROC optimistic; the DELTAS are the
signal. Non-first windows are content truncated away in training, so a max/MIL gain there
reflects real generalization, not memorization.
"""
import os, sys, json, time, tempfile, shutil, multiprocessing as mp
import numpy as np
sys.path.insert(0, os.getcwd())

LONG, WIN, STRIDE, CAP = 1022, 1022, 512, 16
W = max(4, mp.cpu_count() - 4)

# ── parallel biophysical worker (ViennaRNA fold is the bottleneck) ──
_EXT = None
def _init_worker():
    global _EXT
    from Functions.RNA_biophysical import RNABiophysicalExtractor
    _EXT = RNABiophysicalExtractor(normalize=False)
def _bio_one(s):
    return _EXT._compute_one(s)
def parallel_bio(seqs, tag=""):
    t = time.time()
    with mp.Pool(W, initializer=_init_worker) as p:
        bio = np.stack(p.map(_bio_one, seqs, chunksize=4)).astype(np.float32)
    print(f"  [bio {tag}] {len(seqs)} seqs folded on {W} cores in {time.time()-t:.0f}s", flush=True)
    return bio


def norm(s):
    return "".join(c for c in s.upper().replace("T", "U") if not c.isspace())
def windows(s):
    if len(s) <= WIN:
        return [s]
    out, i = [], 0
    while i < len(s) and len(out) < CAP:
        out.append(s[i:i + WIN])
        if i + WIN >= len(s):
            break
        i += STRIDE
    return out


def score_with_bio(scorer, seqs, bio_array, bs=16):
    """v6 forward with PRECOMPUTED (parallel) bio — mirrors RNAPhaseekScorer.score()."""
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import HybridRNADataset, make_collate_fn
    from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted
    from Functions.precompute_fegs import process_fasta
    d = Path(tempfile.mkdtemp(prefix="lrf_")); fa = d / "in.fasta"
    with open(fa, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">s{i}\n{s}\n")
    process_fasta(fa, d, topk=10, seq_len=1024, overwrite=True, workers=8)
    paths = list_npz_sorted(str(d))
    bio_n = ((bio_array - scorer.m) / scorer.sd).astype(np.float32)
    ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), bio_n, scorer.args.max_nucleotides)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=make_collate_fn(scorer.tok, topk_m=10))
    probs = []
    with torch.no_grad():
        for tk, at, Lh, bi, _ in ld:
            tk = tk.to(scorer.device); at = at.to(scorer.device); Lh = Lh.to(scorer.device)
            bi = bi.to(scorer.device) if bi is not None else None
            lg, _ = scorer.model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    shutil.rmtree(d, ignore_errors=True)
    return np.concatenate(probs)


def main():
    import torch, multimolecule  # noqa
    import paths  # noqa
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta
    from Functions.RNAPhaseek.RNAPhaseek_utils import setup_device

    pos = [norm(s) for _, s in read_fasta("Data/raw/multispecies/strict_pool_v5_positives.fasta")]
    neg = [norm(s) for _, s in read_fasta("Data/raw/multispecies/strict_pool_v5_negatives_all.fasta")]
    pos_long = [s for s in pos if len(s) > LONG]
    neg_long = [s for s in neg if len(s) > LONG]
    rng = np.random.RandomState(0)
    n = min(len(pos_long), len(neg_long))
    pos_sub = [pos_long[i] for i in rng.choice(len(pos_long), n, replace=False)]
    neg_sub = [neg_long[i] for i in rng.choice(len(neg_long), n, replace=False)] if len(neg_long) > n else neg_long
    seqs = pos_sub + neg_sub
    y = np.array([1] * len(pos_sub) + [0] * len(neg_sub))
    print(f"long subset: {len(pos_sub)} pos + {len(neg_sub)} neg = {len(seqs)} (>{LONG} nt) | {W} cores", flush=True)

    from rnaphaseek import RNAPhaseekScorer
    scorer = RNAPhaseekScorer(quiet=True)

    # ── stage 1: production full-seq baseline ──
    bio_full = parallel_bio(seqs, "fullseq")
    print("stage 1/3: v6 full-seq (production) ...", flush=True)
    prod_full = score_with_bio(scorer, seqs, bio_full)

    # ── stage 2: tiled windows ──
    flat, owner = [], []
    for gi, s in enumerate(seqs):
        for w in windows(s):
            flat.append(w); owner.append(gi)
    print(f"stage 2/3: v6 on {len(flat)} tiled windows ...", flush=True)
    bio_win = parallel_bio(flat, "windows")
    wp = score_with_bio(scorer, flat, bio_win)
    byseq = defaultdict(list)
    for p, gi in zip(wp, owner):
        byseq[gi].append(float(p))
    first = np.array([byseq[gi][0] for gi in range(len(seqs))])
    maxw = np.array([max(byseq[gi]) for gi in range(len(seqs))])
    meanw = np.array([float(np.mean(byseq[gi])) for gi in range(len(seqs))])
    noisyor = np.array([1 - float(np.prod([1 - p for p in byseq[gi]])) for gi in range(len(seqs))])

    # ── stage 3: v7 MIL ──
    mil = None
    try:
        print("stage 3/3: v7 MIL ...", flush=True)
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq import RNAFMHybridFullSeq
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_config import HybridFullSeqArgs
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_data import FullSeqRNADataset, make_collate_fn as fs_collate
        from transformers import AutoTokenizer
        from torch.utils.data import DataLoader
        device = setup_device()
        a7 = HybridFullSeqArgs(bio_dim=38)
        tok = AutoTokenizer.from_pretrained(a7.backbone, trust_remote_code=True)
        m7 = RNAFMHybridFullSeq(a7).to(device).eval()
        m7.load_state_dict(torch.load("model/strict_eval_v7_mil/final_model.pt", map_location=device, weights_only=True))
        nz = np.load("model/strict_eval_v7_mil/norm_stats.npz")
        bio_n7 = ((bio_full - nz["mean"]) / nz["std"]).astype(np.float32)
        coll = fs_collate(tok, a7.window, a7.stride, a7.max_windows)
        ld = DataLoader(FullSeqRNADataset(seqs, np.zeros(len(seqs), int), bio_n7),
                        batch_size=1, shuffle=False, collate_fn=coll)
        out = []
        with torch.no_grad():
            for tk, at, wm, bi, _ in ld:
                tk = tk.to(device); at = at.to(device); wm = wm.to(device)
                bi = bi.to(device) if bi is not None else None
                lg, _ = m7(tk, at, wm, labels=None, bio_features=bi)
                fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
                out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        mil = np.concatenate(out)
    except Exception as e:
        print(f"!! v7 MIL failed: {e}", flush=True)

    # ── results ──
    print("\n================ DECISIVE LONG-RANGE RESULT (AUROC, long subset) ================")
    strategies = [("production_fullseq (shipped)", prod_full), ("first_window (standalone)", first),
                  ("max_window (best module)", maxw), ("mean_window", meanw), ("noisy_OR (any window)", noisyor)]
    if mil is not None:
        strategies.append(("v7_MIL (learned attention)", mil))
    base = roc_auc_score(y, prod_full)
    res = {}
    for name, s in strategies:
        au = roc_auc_score(y, s); res[name] = au
        print(f"  {name:<32} AUROC={au:.4f}   delta vs shipped={au - base:+.4f}")

    print("\n--- where does the LLPS signal live? (long positives; leakage-robust) ---")
    npos = int((y == 1).sum())
    n_nonfirst = sum(1 for gi in range(len(seqs)) if y[gi] == 1 and len(byseq[gi]) > 1 and int(np.argmax(byseq[gi])) != 0)
    n_strong = sum(1 for gi in range(len(seqs)) if y[gi] == 1 and maxw[gi] - first[gi] > 0.10)
    print(f"  best window is NOT the first:    {n_nonfirst}/{npos} ({100*n_nonfirst/npos:.0f}%)")
    print(f"  max-window beats first by >0.10: {n_strong}/{npos} ({100*n_strong/npos:.0f}%)")
    print("  mean P by window position (positives):")
    maxc = max(len(byseq[gi]) for gi in range(len(seqs)))
    for wpos in range(min(maxc, 6)):
        vals = [byseq[gi][wpos] for gi in range(len(seqs)) if y[gi] == 1 and len(byseq[gi]) > wpos]
        print(f"    window #{wpos} (nt {wpos*STRIDE:>5}-{wpos*STRIDE+WIN:>5}): mean P={np.mean(vals):.3f} (n={len(vals)})")

    json.dump({"auroc": res, "n_pos": npos, "n_neg": int((y == 0).sum()),
               "best_window_nonfirst_frac": n_nonfirst / npos, "max_beats_first_by_0.1_frac": n_strong / npos,
               "window": WIN, "stride": STRIDE, "cap": CAP},
              open("outputs/longrange_decisive.json", "w"), indent=2)
    print("\nsaved -> outputs/longrange_decisive.json")


if __name__ == "__main__":
    main()
