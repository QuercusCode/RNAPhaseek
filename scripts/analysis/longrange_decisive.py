"""DECISIVE long-range experiment: does looking beyond RNA-FM's first-1022-nt window
improve discrimination of long (>1022 nt) RNA-self-LLPS sequences? I.e. does the
1022-nt cap actually cost us anything — and would a long-context v8 backbone help?

On a balanced long subset (pos vs neg, all >1022 nt), compare:
  production_fullseq : v6 as shipped (deep streams see first 1022nt; biophys global)  [baseline]
  first_window       : v6 on the standalone first-1022nt window
  max_window         : v6 on each tiled window -> MAX            (best single LLPS module)
  mean_window        : v6 mean over windows                      (soft MIL)
  noisy_OR           : 1 - prod(1 - p_window)                    (MIL-style "any window")
  v7_MIL             : trained tiling + attention pooling        (learned cross-window)

Caveat: v6/v7 trained on these seqs -> absolute AUROC is OPTIMISTIC. The signal is the
DELTA across strategies; crucially, non-first windows are content the model NEVER saw in
training (truncated away), so a max/MIL gain there reflects real generalization, not memory.

NB: must run under `if __name__ == '__main__'` — FEGS precompute uses multiprocessing spawn.
"""
import os, sys, json, gc
import numpy as np, torch, multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # noqa  project path bootstrap
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta
from Functions.RNAPhaseek.RNAPhaseek_utils import setup_device
from Functions.RNA_biophysical import RNABiophysicalExtractor

LONG, WIN, STRIDE, CAP = 1022, 1022, 512, 16


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


def main():
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
    print(f"long subset: {len(pos_sub)} pos + {len(neg_sub)} neg = {len(seqs)} (all >{LONG} nt)", flush=True)

    # ── v6: production-fullseq baseline + per-window tiling ──
    from rnaphaseek import RNAPhaseekScorer
    sc = RNAPhaseekScorer(quiet=True)
    print("scoring full sequences with v6 (production behavior) ...", flush=True)
    prod_full = sc.score(seqs)

    flat, owner = [], []
    for gi, s in enumerate(seqs):
        for w in windows(s):
            flat.append(w); owner.append(gi)
    print(f"scoring {len(flat)} tiled windows with v6 ...", flush=True)
    wp = sc.score(flat)
    byseq = defaultdict(list)
    for p, gi in zip(wp, owner):
        byseq[gi].append(float(p))
    first = np.array([byseq[gi][0] for gi in range(len(seqs))])
    maxw = np.array([max(byseq[gi]) for gi in range(len(seqs))])
    meanw = np.array([float(np.mean(byseq[gi])) for gi in range(len(seqs))])
    noisyor = np.array([1 - float(np.prod([1 - p for p in byseq[gi]])) for gi in range(len(seqs))])
    del sc; gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass

    # ── v7 MIL (learned tiling + attention pooling) ──
    mil = None
    try:
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq import RNAFMHybridFullSeq
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_config import HybridFullSeqArgs
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_data import FullSeqRNADataset, make_collate_fn as fs_collate
        from torch.utils.data import DataLoader
        device = setup_device()
        a7 = HybridFullSeqArgs(bio_dim=38)
        tok = AutoTokenizer.from_pretrained(a7.backbone, trust_remote_code=True)
        m7 = RNAFMHybridFullSeq(a7).to(device).eval()
        m7.load_state_dict(torch.load("model/strict_eval_v7_mil/final_model.pt", map_location=device, weights_only=True))
        ext = RNABiophysicalExtractor(normalize=False)
        bio = np.stack([ext._compute_one(s) for s in seqs]).astype(np.float32)
        nz = np.load("model/strict_eval_v7_mil/norm_stats.npz")
        bio_n = ((bio - nz["mean"]) / nz["std"]).astype(np.float32)
        coll = fs_collate(tok, a7.window, a7.stride, a7.max_windows)
        ld = DataLoader(FullSeqRNADataset(seqs, np.zeros(len(seqs), int), bio_n),
                        batch_size=1, shuffle=False, collate_fn=coll)
        print(f"scoring {len(seqs)} seqs with v7 MIL ...", flush=True)
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
        print(f"!! v7 MIL scoring failed: {e}", flush=True)

    # ── Results ──
    print("\n================ DECISIVE LONG-RANGE RESULT (AUROC on long subset) ================")
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
    print(f"  best window is NOT the first:      {n_nonfirst}/{npos} ({100*n_nonfirst/npos:.0f}%)")
    print(f"  max-window beats first by >0.10:   {n_strong}/{npos} ({100*n_strong/npos:.0f}%)")
    print("  mean P by window position (positives):")
    maxc = max(len(byseq[gi]) for gi in range(len(seqs)))
    for wpos in range(min(maxc, 6)):
        vals = [byseq[gi][wpos] for gi in range(len(seqs)) if y[gi] == 1 and len(byseq[gi]) > wpos]
        print(f"    window #{wpos} (nt {wpos*STRIDE:>5}-{wpos*STRIDE+WIN:>5}): mean P={np.mean(vals):.3f}  (n={len(vals)})")

    json.dump({"auroc": res, "n_pos": npos, "n_neg": int((y == 0).sum()),
               "best_window_nonfirst_frac": n_nonfirst / npos, "max_beats_first_by_0.1_frac": n_strong / npos,
               "window": WIN, "stride": STRIDE, "cap": CAP},
              open("outputs/longrange_decisive.json", "w"), indent=2)
    print("\nsaved -> outputs/longrange_decisive.json")


if __name__ == "__main__":
    main()
