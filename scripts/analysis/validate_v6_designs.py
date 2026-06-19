"""Structure-dependence validation of the v6-GA designs, judged by the v6 production
model. Each design is scored against its composition-matched scrambles; Delta>0 means
the score is structure-driven (trustworthy), not a composition artifact.
Controls: real LLPS positives (expect Delta>0) and random RNA (expect ~0, low score)."""
import os, sys, json, random, tempfile
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # project path bootstrap (see paths.py)
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils         import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                     import RNABiophysicalExtractor
from Functions.precompute_fegs                     import process_fasta
from generate_structural_negatives_v4              import ae_dishuffle, mono_shuffle

FINAL = "model/strict_eval_v6_production/final_model.pt"
NORM  = "model/strict_eval_v6_production/norm_stats.npz"
K = 3


def scramble(s):
    out = []
    for k in range(K):
        rng = random.Random(100 + k)
        out.append(ae_dishuffle(s, rng) or mono_shuffle(s, rng))
    return out


def main():
    set_seed(42); device = setup_device()
    ext = RNABiophysicalExtractor(normalize=False)
    tok = AutoTokenizer.from_pretrained("multimolecule/rnafm", trust_remote_code=True)
    nz = np.load(NORM); m, sd = nz["mean"].astype(np.float32), nz["std"].astype(np.float32)
    args = HybridTrainArgs(bio_dim=38, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
    model = RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(FINAL, map_location=device, weights_only=True))

    groups = []   # (label, [design+scrambles])
    for _, s in read_fasta("outputs/designs/designed_ga_v6.fasta"):
        groups.append(("v6_GA", [s] + scramble(s)))
    if os.path.exists("outputs/designs/designed_den_v6.fasta"):
        for _, s in read_fasta("outputs/designs/designed_den_v6.fasta"):
            groups.append(("v6_DEN", [s] + scramble(s)))
    rng = random.Random(7)
    shortpos = [s for _, s in read_fasta("Data/raw/multispecies/strict_pool_v5_positives.fasta") if 100 <= len(s) <= 400]
    for s in rng.sample(shortpos, 12):
        groups.append(("real_pos", [s] + scramble(s)))
    for i in range(8):
        rr = random.Random(50 + i)
        s = "".join(rr.choice("ACGU") for _ in range(200))
        groups.append(("random", [s] + scramble(s)))

    flat, owner = [], []
    for gi, (_, seqs) in enumerate(groups):
        for j, s in enumerate(seqs):
            flat.append(s); owner.append((gi, j == 0))
    d = Path(tempfile.mkdtemp(prefix="fegs_v6val_")); fa = d / "s.fasta"
    with open(fa, "w") as f:
        for i, s in enumerate(flat): f.write(f">s{i}\n{s}\n")
    process_fasta(fa, d, topk=10, seq_len=1024, overwrite=True, workers=4)
    paths = list_npz_sorted(str(d))
    bio = np.stack([ext._compute_one(s) for s in flat]).astype(np.float32)
    ds = HybridRNADataset(flat, paths, np.zeros(len(flat), int), (bio - m) / sd, args.max_nucleotides)
    ld = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
    probs = []
    with torch.no_grad():
        for tk, at, Lh, bi, _ in ld:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device); bi = bi.to(device) if bi is not None else None
            lg, _ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)

    rows = {}
    for gi, (lab, seqs) in enumerate(groups):
        idxs = [i for i, (g, _) in enumerate(owner) if g == gi]
        di = [i for i in idxs if owner[i][1]][0]; sc = [i for i in idxs if not owner[i][1]]
        rows.setdefault(lab, []).append((probs[di], probs[sc].mean()))
    print(f"\n{'group':<12}{'n':>4}{'design':>9}{'scramble':>10}{'Delta':>8}")
    print("-"*45)
    summ = {}
    for lab in ["v6_GA", "v6_DEN", "real_pos", "random"]:
        if lab not in rows: continue
        r = rows[lab]; dm = np.mean([x[0] for x in r]); sm = np.mean([x[1] for x in r])
        print(f"{lab:<12}{len(r):>4}{dm:>9.3f}{sm:>10.3f}{dm-sm:>+8.3f}")
        summ[lab] = {"design": float(dm), "scramble": float(sm), "delta": float(dm-sm), "n": len(r)}
    json.dump(summ, open("model/strict_eval_v6_production/design_structure_dependence.json", "w"), indent=2)
    print("\nDelta>0 => v6 designs are structure-driven (trustworthy), not composition artifacts.")


if __name__ == "__main__":
    main()
