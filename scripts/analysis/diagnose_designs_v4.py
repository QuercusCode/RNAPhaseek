"""
Structure-dependence diagnostic: are the existing de novo designs structurally
grounded, or composition artifacts? Judge each design by whether the STRUCTURE-AWARE
v4 model scores it higher than its own composition-matched scramble.

For every design d: make K composition-matched scrambles (di-shuffle, mono fallback),
score d and its scrambles with BOTH models, and report the structure-dependence
  delta = P(design) - mean P(scrambles).
v4 (reads self-complementarity) should give delta > 0 for a structurally-real design;
v3 (composition-blind) should give delta ~ 0 regardless. Controls:
  * real LLPS positives (short)  -> expect delta_v4 > 0 (true structure)
  * random sequences             -> expect delta_v4 ~ 0 and low absolute score

Efficiency: columns 0-32 of the width-38 vector ARE the original v3 features
(new features were appended at the end), so one featurization scores both models.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python diagnose_designs_v4.py
"""
import os, sys, json, random, tempfile
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # project path bootstrap (see paths.py)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils         import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                     import RNABiophysicalExtractor
from Functions.precompute_fegs                     import process_fasta
from generate_structural_negatives_v4              import ae_dishuffle, mono_shuffle, struct_metrics

V3 = "model/strict_eval_v3aug/final_model.pt"
V4 = "model/strict_eval_v4/final_model.pt"
K_SCRAMBLE = 3


def v3_norm():
    pos = np.load("Data/splits/biophys_strict_v3_pos.npy"); neg = np.load("Data/splits/biophys_strict_v3_neg.npy")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
    bio = np.vstack([pos, neg]).astype(np.float32)
    dev, _ = train_test_split(np.arange(len(y)), test_size=0.15, random_state=999, stratify=y)
    f_tr, _ = train_test_split(dev, test_size=0.15, random_state=7, stratify=y[dev])
    btr = np.vstack([bio[f_tr], np.load("Data/splits/biophys_synth_train.npy").astype(np.float32)])
    return btr.mean(0), btr.std(0).clip(min=1e-8)


def scramble(seq, n=K_SCRAMBLE):
    out = []
    for k in range(n):
        rng = random.Random(1000 + k)
        c = ae_dishuffle(seq, rng) or mono_shuffle(seq, rng)
        out.append(c)
    return out


def main():
    set_seed(42); device = setup_device()
    ext = RNABiophysicalExtractor(normalize=False)
    tok = AutoTokenizer.from_pretrained("multimolecule/rnafm", trust_remote_code=True)

    # ── assemble design groups + controls ──
    groups = []  # (method, parent_seq, [design + scrambles...])
    design_files = [("GA", "outputs/designs/designed_ga.fasta"), ("DEN", "outputs/designs/designed_den.fasta"),
                    ("SeqProp", "outputs/designs/designed_v3_seqprop.fasta")]
    if os.path.exists("outputs/designs/designed_ga_v4.fasta"):
        design_files.append(("GA_v4", "outputs/designs/designed_ga_v4.fasta"))
    for meth, fa in design_files:
        for _, s in read_fasta(fa):
            groups.append((meth, s, [s] + scramble(s)))
    # positive control: short real LLPS positives
    rng = random.Random(7)
    shortpos = [s for _, s in read_fasta("Data/raw/multispecies/strict_pool_v3_positives.fasta")
                if 100 <= len(s) <= 400]
    for s in rng.sample(shortpos, min(20, len(shortpos))):
        groups.append(("real_pos", s, [s] + scramble(s)))
    # random-sequence floor
    for i in range(10):
        rr = random.Random(500 + i)
        s = "".join(rr.choice("ACGU") for _ in range(200))
        groups.append(("random", s, [s] + scramble(s)))

    # flatten -> one FEGS + bio pass
    flat, owner = [], []   # owner[i] = (group_idx, is_design)
    for gi, (_, _, seqs) in enumerate(groups):
        for j, s in enumerate(seqs):
            flat.append(s); owner.append((gi, j == 0))
    print(f"{len(groups)} groups, {len(flat)} sequences (designs + {K_SCRAMBLE} scrambles each)")

    d = Path(tempfile.mkdtemp(prefix="fegs_diag_"))
    fa = d / "s.fasta"
    with open(fa, "w") as f:
        for i, s in enumerate(flat): f.write(f">s{i}\n{s}\n")
    process_fasta(fa, d, topk=10, seq_len=1024, overwrite=True, workers=4)
    paths = list_npz_sorted(str(d))
    bio38 = np.stack([ext._compute_one(s) for s in flat]).astype(np.float32)

    def score(model_path, bio_in, bio_dim):
        args = HybridTrainArgs(bio_dim=bio_dim, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
        model = RNAFMHybridClassifier(args).to(device).eval()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        ds = HybridRNADataset(flat, paths, np.zeros(len(flat), int), bio_in, args.max_nucleotides)
        ld = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
        probs = []
        with torch.no_grad():
            for tk, at, Lh, bi, _ in ld:
                tk = tk.to(device); at = at.to(device); Lh = Lh.to(device)
                bi = bi.to(device) if bi is not None else None
                lg, _ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
                fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
                probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        del model
        return np.concatenate(probs)

    m4 = np.load("model/strict_eval_v4/norm_stats.npz"); mean4, sd4 = m4["mean"], m4["std"]
    mean3, sd3 = v3_norm()
    p4 = score(V4, (bio38 - mean4) / sd4, 38)
    p3 = score(V3, (bio38[:, :33] - mean3) / sd3, 33)

    # ── per-group structure-dependence ──
    rows = {}
    for gi, (meth, _, seqs) in enumerate(groups):
        idxs = [i for i, (g, _) in enumerate(owner) if g == gi]
        di = [i for i in idxs if owner[i][1]][0]
        sc = [i for i in idxs if not owner[i][1]]
        rec = rows.setdefault(meth, [])
        rec.append(dict(d4=p4[di], s4=p4[sc].mean(), d3=p3[di], s3=p3[sc].mean()))

    print(f"\n{'group':<10}{'n':>4}{'v4_design':>11}{'v4_scram':>10}{'Δ_v4':>8}{'v3_design':>11}{'v3_scram':>10}{'Δ_v3':>8}")
    print("-" * 72)
    summ = {}
    for meth in ["GA", "GA_v4", "DEN", "SeqProp", "real_pos", "random"]:
        r = rows.get(meth, [])
        if not r: continue
        d4 = np.mean([x["d4"] for x in r]); s4 = np.mean([x["s4"] for x in r])
        d3 = np.mean([x["d3"] for x in r]); s3 = np.mean([x["s3"] for x in r])
        print(f"{meth:<10}{len(r):>4}{d4:>11.3f}{s4:>10.3f}{d4-s4:>+8.3f}{d3:>11.3f}{s3:>10.3f}{d3-s3:>+8.3f}")
        summ[meth] = dict(n=len(r), v4_design=float(d4), v4_scramble=float(s4), delta_v4=float(d4-s4),
                          v3_design=float(d3), v3_scramble=float(s3), delta_v3=float(d3-s3))
    json.dump(summ, open("model/strict_eval_v4/design_structure_dependence.json", "w"), indent=2)
    print("\nΔ>0 ⇒ score is structure-driven (trustworthy);  Δ≈0 ⇒ composition artifact.")
    print("Saved -> model/strict_eval_v4/design_structure_dependence.json")


if __name__ == "__main__":
    main()
