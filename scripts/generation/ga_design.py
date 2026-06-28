"""
Genetic-algorithm de novo design against the RNAPhaseek model.

Fitness uses the full pipeline (38-dim biophysics including the 5
self-complementarity features + FEGS + forward), so the GA evolves
structurally-grounded designs — verified afterwards by the
design-vs-scramble structure-dependence check.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python ga_design.py
"""
import os, sys, json, random, tempfile
import numpy as np, torch
import multimolecule  # noqa
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils         import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                     import RNABiophysicalExtractor
from Functions.precompute_fegs                     import process_fasta

FINAL = "model/final_model.pt"
NORM  = "model/norm_stats.npz"
BASES = "ACGU"
POP, GEN, L = 64, 40, 200
ELITE, MUT, TOURN = 10, 0.04, 3
SEED = 0


def main():
    set_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    device = setup_device()
    args = HybridTrainArgs(bio_dim=38, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
    model = RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(FINAL, map_location=device, weights_only=True))
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    nz = np.load(NORM); m, sd = nz["mean"].astype(np.float32), nz["std"].astype(np.float32)
    ext = RNABiophysicalExtractor(normalize=False)

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
        i, j = sorted(random.sample(range(1, L), 2)); return a[:i] + b[i:j] + a[j:]
    def tournament(pop, fit):
        cand = random.sample(range(len(pop)), TOURN); return pop[max(cand, key=lambda i: fit[i])]

    pop = ["".join(random.choice(BASES) for _ in range(L)) for _ in range(POP)]
    cache, history = {}, []
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

    final = sorted(set(pop), key=lambda s: -cache.get(s, 0))
    top = final[:10]
    print("\n=== TOP GA DESIGNS (full-model P(LLPS)) ===")
    print(f"{'P':>6} {'GC%':>4} {'A%':>4} {'C%':>4} {'G%':>4} {'U%':>4}  preview")
    for s in top:
        c = Counter(s); n = len(s)
        print(f"{cache[s]:>6.3f} {100*(c['G']+c['C'])/n:>4.0f} {100*c['A']/n:>4.0f} {100*c['C']/n:>4.0f} "
              f"{100*c['G']/n:>4.0f} {100*c['U']/n:>4.0f}  {s[:46]}")

    os.makedirs("outputs/designs", exist_ok=True)
    with open("outputs/designs/designed_ga.fasta", "w") as f:
        for i, s in enumerate(top): f.write(f">ga_design_{i}_P{cache[s]:.3f}\n{s}\n")
    h = np.array(history)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(h[:, 0], h[:, 1], "-o", ms=3, color="#16a085", label="best in population")
    ax.plot(h[:, 0], h[:, 2], "-o", ms=3, color="#95a5a6", label="population mean")
    ax.set_xlabel("generation"); ax.set_ylabel("full-model P(LLPS)")
    ax.set_title("Genetic-algorithm convergence on the RNAPhaseek model", fontweight="bold")
    ax.legend(frameon=False); ax.grid(alpha=0.25, ls="--")
    os.makedirs("report_assets", exist_ok=True)
    fig.savefig("report_assets/fig_ga_convergence.png", dpi=140, bbox_inches="tight"); plt.close()
    os.makedirs("outputs/designs", exist_ok=True)
    json.dump({"history": history, "top": [{"seq": s, "P": cache[s]} for s in top]},
              open("outputs/designs/ga_summary.json", "w"), indent=2)
    print("\nSaved -> outputs/designs/designed_ga.fasta, report_assets/fig_ga_convergence.png")


if __name__ == "__main__":
    main()
