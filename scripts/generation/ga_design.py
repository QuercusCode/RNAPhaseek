"""
Genetic-algorithm de novo LLPS RNA design.

Unlike SeqProp (gradient through a bio-zero proxy), the GA scores every
candidate with the FULL v3 pipeline (real 33-dim biophysical features + FEGS
structure), and is population-based so it explores diverse solutions rather
than collapsing to a single attractor.

  initialize random population
  for each generation:
    score all (full model) -> fitness = P(LLPS)
    keep elites; breed the rest via tournament-select + 2-point crossover + mutation
  report best + diverse designs, convergence curve, composition.

Run:
  python ga_design.py
"""
import os, sys, json, random, tempfile
import numpy as np, torch
import multimolecule  # noqa
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils         import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                     import RNABiophysicalExtractor
from Functions.precompute_fegs                     import process_fasta

FINAL = "model/strict_eval_v3aug/final_model.pt"
BASES = "ACGU"
# ── GA hyperparameters ──
POP, GEN, L = 64, 40, 200
ELITE, MUT, TOURN = 10, 0.04, 3
SEED = 0


def recover_norm():
    pos = read_fasta("Data/raw/multispecies/strict_pool_v3_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v3_negatives_all.fasta")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
    bio = np.vstack([np.load("Data/splits/biophys_strict_v3_pos.npy"),
                     np.load("Data/splits/biophys_strict_v3_neg.npy")]).astype(np.float32)
    dev, _ = train_test_split(np.arange(len(y)), test_size=0.15, random_state=999, stratify=y)
    f_tr, _ = train_test_split(dev, test_size=0.15, random_state=7, stratify=y[dev])
    btr = np.vstack([bio[f_tr], np.load("Data/splits/biophys_synth_train.npy").astype(np.float32)])
    return btr.mean(0), btr.std(0).clip(min=1e-8)


def main():
    set_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    device = setup_device()
    args = HybridTrainArgs(bio_dim=33, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
    model = RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(FINAL, map_location=device, weights_only=True))
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    m, sd = recover_norm(); ext = RNABiophysicalExtractor(normalize=False)

    def score_batch(seqs):
        if not seqs: return np.array([])
        d = Path(tempfile.mkdtemp(prefix="fegs_ga_"))
        fa = d / "s.fasta"
        with open(fa, "w") as f:
            for i, s in enumerate(seqs): f.write(f">s{i}\n{s}\n")
        process_fasta(fa, d, topk=10, seq_len=1024, overwrite=True, workers=4)
        paths = list_npz_sorted(str(d))
        bio = np.stack([ext._compute_one(s) for s in seqs]).astype(np.float32)
        ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), (bio - m) / sd, args.max_nucleotides)
        ld = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
        out = []
        with torch.no_grad():
            for tk, at, Lh, bi, _ in ld:
                tk = tk.to(device); at = at.to(device); Lh = Lh.to(device)
                bi = bi.to(device) if bi is not None else None
                lg, _ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
                fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
                out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        import shutil; shutil.rmtree(d, ignore_errors=True)
        return np.concatenate(out)

    def mutate(s):
        return "".join(random.choice(BASES) if random.random() < MUT else c for c in s)
    def crossover(a, b):
        i, j = sorted(random.sample(range(1, L), 2))
        return a[:i] + b[i:j] + a[j:]
    def tournament(pop, fit):
        cand = random.sample(range(len(pop)), TOURN)
        return pop[max(cand, key=lambda i: fit[i])]

    pop = ["".join(random.choice(BASES) for _ in range(L)) for _ in range(POP)]
    cache = {}; history = []
    for g in range(GEN):
        new = [s for s in pop if s not in cache]
        for s, p in zip(new, score_batch(new)): cache[s] = float(p)
        fit = [cache[s] for s in pop]
        order = sorted(range(POP), key=lambda i: -fit[i])
        best, mean = fit[order[0]], float(np.mean(fit))
        history.append((g, best, mean))
        print(f"gen {g:>2}/{GEN}: best={best:.4f} mean={mean:.4f} unique={len(set(pop))}", flush=True)
        elites = [pop[i] for i in order[:ELITE]]
        newpop = list(elites)
        while len(newpop) < POP:
            newpop.append(mutate(crossover(tournament(pop, fit), tournament(pop, fit))))
        pop = newpop

    # Final: top diverse designs
    final = sorted(set(pop), key=lambda s: -cache.get(s, 0))
    top = final[:10]
    print("\n=== TOP GA DESIGNS (full-model P(LLPS)) ===")
    print(f"{'P':>6} {'GC%':>4} {'A%':>4} {'C%':>4} {'G%':>4} {'U%':>4}  preview")
    for s in top:
        c = Counter(s); n = len(s)
        print(f"{cache[s]:>6.3f} {100*(c['G']+c['C'])/n:>4.0f} "
              f"{100*c['A']/n:>4.0f} {100*c['C']/n:>4.0f} {100*c['G']/n:>4.0f} {100*c['U']/n:>4.0f}  {s[:46]}")

    # composition of top-20 vs seqprop reference
    pool = final[:20]; cc = Counter("".join(pool)); tn = sum(cc.values())
    print(f"\nGA top-20 mean composition: A={100*cc['A']/tn:.0f}% C={100*cc['C']/tn:.0f}% "
          f"G={100*cc['G']/tn:.0f}% U={100*cc['U']/tn:.0f}%")

    with open("outputs/designs/designed_ga.fasta", "w") as f:
        for i, s in enumerate(top): f.write(f">ga_design_{i}_P{cache[s]:.3f}\n{s}\n")
    # convergence figure
    h = np.array(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(h[:, 0], h[:, 1], "-o", ms=3, color="#27ae60", label="best in population")
    ax.plot(h[:, 0], h[:, 2], "-o", ms=3, color="#95a5a6", label="population mean")
    ax.set_xlabel("generation"); ax.set_ylabel("full-model P(LLPS)")
    ax.set_title("Figure 13 — Genetic-algorithm convergence (full v3 model)", fontweight="bold")
    ax.legend(frameon=False); ax.grid(alpha=0.25, ls="--")
    fig.savefig("report_assets/fig13_ga_convergence.png", dpi=140, bbox_inches="tight"); plt.close()
    json.dump({"history": history, "top": [{"seq": s, "P": cache[s]} for s in top]},
              open("model/strict_eval_v3aug/ga_summary.json", "w"), indent=2)
    print("\nSaved -> designed_ga.fasta, report_assets/fig13_ga_convergence.png")


if __name__ == "__main__":
    main()
