"""ITEM 6 — uncertainty / abstention. Does an uncertainty signal flag the known
out-of-distribution failure (scrambled kissing-loops A_bar/B_bar that v6 confidently
mis-scored ~0.92) so the model could ABSTAIN instead of guessing?

Two signals, no retraining:
  - MC-dropout: keep the model's Dropout(0.1) active, run N stochastic forward passes
    -> per-seq mean P and STD (epistemic uncertainty of one model).
  - Ensemble disagreement: score with 3 independently-trained models (v6_production,
    v6_orgbalanced, v7_mil) -> STD across models.
Test groups: KL_pos (27 real LLPS), KL_neg (A_bar/B_bar OOD), v6 designs (confident
in-distribution), random (confident negatives). High uncertainty on KL_neg = success.
"""
import os, sys, json, tempfile, shutil, random
import numpy as np
sys.path.insert(0, os.getcwd())
from pathlib import Path

_EXT = None
def _init():
    global _EXT
    from Functions.RNA_biophysical import RNABiophysicalExtractor
    _EXT = RNABiophysicalExtractor(normalize=False)
def _bio_one(s):
    return _EXT._compute_one(s)


def load_fa(f):
    recs = []; h = None; s = ""
    for ln in open(f):
        ln = ln.rstrip()
        if ln.startswith(">"):
            if h: recs.append((h, s))
            h = ln[1:]; s = ""
        elif ln: s += ln
    if h: recs.append((h, s))
    return recs
def norm(s):
    return "".join(c for c in s.upper().replace("T", "U") if not c.isspace())


def main():
    import torch, multimolecule, multiprocessing as mp  # noqa
    import paths  # noqa
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import HybridRNADataset, make_collate_fn
    from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device
    from Functions.precompute_fegs import process_fasta
    from rnaphaseek import RNAPhaseekScorer
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq import RNAFMHybridFullSeq
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_config import HybridFullSeqArgs
    from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_data import FullSeqRNADataset, make_collate_fn as fs_collate

    # ── assemble labelled test set ──
    items = []  # (name, group, seq)
    for h, s in load_fa("Data/raw/multispecies/external/external_deleaked.fasta"):
        nm = h.split("|")[2]
        grp = "KL_neg (A_bar/B_bar)" if "label=neg" in h else "KL_pos (real LLPS)"
        items.append((nm, grp, norm(s)))
    for h, s in load_fa("outputs/designs/designed_ga_v6.fasta")[:10]:
        items.append((h.split("_P")[0], "design (in-dist)", norm(s)))
    rng = random.Random(0)
    for i in range(10):
        items.append((f"random_{i}", "random", "".join(rng.choice("ACGU") for _ in range(200))))
    names = [x[0] for x in items]; groups = [x[1] for x in items]; seqs = [x[2] for x in items]
    print(f"test set: {len(seqs)} seqs | groups: {sorted(set(groups))}", flush=True)

    # ── biophysics (parallel, once) + FEGS (once) ──
    with mp.Pool(min(16, mp.cpu_count() - 2), initializer=_init) as p:
        bio = np.stack(p.map(_bio_one, seqs)).astype(np.float32)
    d = Path(tempfile.mkdtemp(prefix="unc_")); fa = d / "in.fasta"
    with open(fa, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">s{i}\n{s}\n")
    process_fasta(fa, d, topk=10, seq_len=1024, overwrite=True, workers=8)
    paths_npz = list_npz_sorted(str(d))
    device = setup_device()

    def v6_pass(sc, n_pass=1, mc=False):
        bio_n = ((bio - sc.m) / sc.sd).astype(np.float32)
        ds = HybridRNADataset(seqs, paths_npz, np.zeros(len(seqs), int), bio_n, sc.args.max_nucleotides)
        ld = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=make_collate_fn(sc.tok, topk_m=10))
        sc.model.eval()
        if mc:
            for m in sc.model.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
        allp = []
        for _ in range(n_pass):
            ps = []
            with torch.no_grad():
                for tk, at, Lh, bi, _ in ld:
                    tk = tk.to(device); at = at.to(device); Lh = Lh.to(device)
                    bi = bi.to(device) if bi is not None else None
                    lg, _ = sc.model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
                    fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
                    ps.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
            allp.append(np.concatenate(ps))
        return np.array(allp)

    # MC-dropout on v6 production
    sc6 = RNAPhaseekScorer(quiet=True)
    mc = v6_pass(sc6, n_pass=30, mc=True)
    p6_mc_mean, p6_mc_std = mc.mean(0), mc.std(0)
    p6 = v6_pass(sc6, n_pass=1, mc=False)[0]

    # ensemble members
    sc6b = RNAPhaseekScorer("model/strict_eval_v6_orgbalanced/final_model.pt",
                            "model/strict_eval_v6_orgbalanced/norm_stats.npz", quiet=True)
    p6b = v6_pass(sc6b, 1, False)[0]
    a7 = HybridFullSeqArgs(bio_dim=38)
    tok = AutoTokenizer.from_pretrained(a7.backbone, trust_remote_code=True)
    m7 = RNAFMHybridFullSeq(a7).to(device).eval()
    m7.load_state_dict(torch.load("model/strict_eval_v7_mil/final_model.pt", map_location=device, weights_only=True))
    nz7 = np.load("model/strict_eval_v7_mil/norm_stats.npz")
    bio_n7 = ((bio - nz7["mean"]) / nz7["std"]).astype(np.float32)
    ld7 = DataLoader(FullSeqRNADataset(seqs, np.zeros(len(seqs), int), bio_n7),
                     batch_size=1, shuffle=False, collate_fn=fs_collate(tok, a7.window, a7.stride, a7.max_windows))
    out = []
    with torch.no_grad():
        for tk, at, wm, bi, _ in ld7:
            tk = tk.to(device); at = at.to(device); wm = wm.to(device)
            bi = bi.to(device) if bi is not None else None
            lg, _ = m7(tk, at, wm, labels=None, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
            out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    p7 = np.concatenate(out)
    ens = np.vstack([p6, p6b, p7]); ens_std = ens.std(0)
    shutil.rmtree(d, ignore_errors=True)

    # ── report ──
    print("\n================ UNCERTAINTY by group ================")
    print(f"{'group':<24}{'n':>3} {'meanP':>7} {'MCdrop_std':>11} {'ens_std':>9}")
    order = ["KL_pos (real LLPS)", "design (in-dist)", "random", "KL_neg (A_bar/B_bar)"]
    res = {}
    for g in order:
        idx = [i for i in range(len(seqs)) if groups[i] == g]
        if not idx: continue
        mp_ = p6[idx].mean(); mcs = p6_mc_std[idx].mean(); ens = ens_std[idx].mean()
        res[g] = {"n": len(idx), "meanP": float(mp_), "mcdrop_std": float(mcs), "ens_std": float(ens)}
        print(f"{g:<24}{len(idx):>3} {mp_:>7.3f} {mcs:>11.4f} {ens:>9.4f}")

    print("\n--- the OOD failures, per-sequence ---")
    print(f"{'name':<14}{'P(v6)':>7}{'MCstd':>8}{'ens_std':>9}  (v6 alone was confidently wrong)")
    for i in range(len(seqs)):
        if groups[i].startswith("KL_neg"):
            print(f"{names[i]:<14}{p6[i]:>7.3f}{p6_mc_std[i]:>8.4f}{ens_std[i]:>9.4f}")

    # does uncertainty separate KL_neg from confident in-dist (designs)?
    kln = [i for i in range(len(seqs)) if groups[i].startswith("KL_neg")]
    des = [i for i in range(len(seqs)) if groups[i].startswith("design")]
    print("\n--- separation: KL_neg vs confident designs ---")
    print(f"  MC-dropout std: KL_neg={p6_mc_std[kln].mean():.4f}  vs  design={p6_mc_std[des].mean():.4f}")
    print(f"  ensemble  std : KL_neg={ens_std[kln].mean():.4f}  vs  design={ens_std[des].mean():.4f}")
    json.dump({"by_group": res,
               "kl_neg": [{"name": names[i], "p6": float(p6[i]), "mc_std": float(p6_mc_std[i]),
                           "ens_std": float(ens_std[i])} for i in kln]},
              open("outputs/uncertainty_ood.json", "w"), indent=2)
    print("\nsaved -> outputs/uncertainty_ood.json")


if __name__ == "__main__":
    main()
