"""HELD-OUT confirmation of the long-range MIL gain (task 2).

Both v6 and v7 were compared on the SAME locked test that NEITHER trained on:
  - v6_orgbalanced : standard hybrid (TRUNCATES to first 1022nt), trained on the dev split
  - v7_mil         : attention-MIL over <=1022nt windows (full-length), same dev split
Re-derives the exact locked test (GroupShuffleSplit test_size=0.15, random_state=999 — the
v7 definition), scores both, and restricts to the LONG (>1022nt) subset. This is the
leakage-free MIL-vs-truncation delta (the in-sample experiment gave +0.029).

Self-consistency: full-test AUROC should reproduce the saved values (v6~0.827, v7~0.852).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.getcwd())


def main():
    import torch, multimolecule  # noqa
    import paths  # noqa
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from run_v7_mil import build_v5_seqs
    from longrange_fast import score_with_bio
    from rnaphaseek import RNAPhaseekScorer
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq import RNAFMHybridFullSeq
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_config import HybridFullSeqArgs
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_data import FullSeqRNADataset, make_collate_fn as fs_collate
    from Functions.RNAPhaseek.RNAPhaseek_utils import setup_device

    seqs, y, bio_pre, is_struct = build_v5_seqs()          # bio_pre = RAW precomputed biophysics
    groups = np.load("Data/splits/cluster_groups_v5.npy")
    all_idx = np.arange(len(y))
    _, test_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=999).split(all_idx, y, groups))
    test_idx = np.array(test_idx)
    test_seqs = [seqs[i] for i in test_idx]
    yt = y[test_idx]
    bio_test = bio_pre[test_idx].astype(np.float32)
    lengths = np.array([len(s) for s in test_seqs])
    long_mask = lengths > 1022
    print(f"locked test (held out by BOTH models): {len(test_idx)} seqs | "
          f"pos={int(yt.sum())} neg={int((yt == 0).sum())} | "
          f"long(>1022nt)={int(long_mask.sum())} (pos={int(yt[long_mask].sum())} neg={int((yt[long_mask] == 0).sum())})",
          flush=True)

    # ── v6_orgbalanced (TRUNCATION), held-out ──
    print("scoring locked test with v6_orgbalanced (truncation) ...", flush=True)
    sc6 = RNAPhaseekScorer("model/strict_eval_v6_orgbalanced/final_model.pt",
                           "model/strict_eval_v6_orgbalanced/norm_stats.npz", quiet=True)
    p6 = score_with_bio(sc6, test_seqs, bio_test)
    del sc6
    import gc; gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass

    # ── v7_mil (full-length), held-out ──
    print("scoring locked test with v7 MIL (full-length) ...", flush=True)
    device = setup_device()
    a7 = HybridFullSeqArgs(bio_dim=38)
    tok = AutoTokenizer.from_pretrained(a7.backbone, trust_remote_code=True)
    m7 = RNAFMHybridFullSeq(a7).to(device).eval()
    m7.load_state_dict(torch.load("model/strict_eval_v7_mil/final_model.pt", map_location=device, weights_only=True))
    nz7 = np.load("model/strict_eval_v7_mil/norm_stats.npz")
    bio_n7 = ((bio_test - nz7["mean"]) / nz7["std"]).astype(np.float32)
    coll = fs_collate(tok, a7.window, a7.stride, a7.max_windows)
    ld = DataLoader(FullSeqRNADataset(test_seqs, np.zeros(len(test_seqs), int), bio_n7),
                    batch_size=1, shuffle=False, collate_fn=coll)
    out = []
    with torch.no_grad():
        for tk, at, wm, bi, _ in ld:
            tk = tk.to(device); at = at.to(device); wm = wm.to(device)
            bi = bi.to(device) if bi is not None else None
            lg, _ = m7(tk, at, wm, labels=None, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
            out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    p7 = np.concatenate(out)

    # ── results ──
    def auc(lab, p):
        return roc_auc_score(lab, p) if len(set(lab)) > 1 else float("nan")
    print("\n================ HELD-OUT MIL vs TRUNCATION (locked test) ================")
    print("FULL locked test (self-consistency vs saved):")
    print(f"  v6_orgbalanced (truncation): {auc(yt, p6):.4f}   (saved ~0.827)")
    print(f"  v7_mil (full-length):        {auc(yt, p7):.4f}   (saved ~0.852)")
    lm = long_mask
    a6, a7v = auc(yt[lm], p6[lm]), auc(yt[lm], p7[lm])
    print("\nLONG (>1022nt) HELD-OUT subset  <-- THE ANSWER:")
    print(f"  v6_orgbalanced (truncation): {a6:.4f}")
    print(f"  v7_mil (full-length):        {a7v:.4f}")
    print(f"  delta (MIL - truncation):    {a7v - a6:+.4f}   (in-sample experiment gave +0.029)")
    sm = ~long_mask
    print(f"\nSHORT (<=1022nt) subset (MIL≈truncation expected): "
          f"v6={auc(yt[sm], p6[sm]):.4f}  v7={auc(yt[sm], p7[sm]):.4f}  (n={int(sm.sum())})")

    json.dump({"n_test": int(len(test_idx)), "n_long": int(lm.sum()),
               "full_v6_trunc": float(auc(yt, p6)), "full_v7_mil": float(auc(yt, p7)),
               "long_v6_trunc": float(a6), "long_v7_mil": float(a7v), "long_delta": float(a7v - a6)},
              open("outputs/longrange_heldout.json", "w"), indent=2)
    print("\nsaved -> outputs/longrange_heldout.json")


if __name__ == "__main__":
    main()
