"""
Re-score de novo designed RNAs with the FULL v3 pipeline (real biophysical
features + FEGS), not the bio-zero proxy the generator optimizes against.
Also characterizes what the model 'designed'.

Run after generation:
  python rescore_designs.py designed_v3_seqprop.fasta
"""
import os, sys, json, tempfile
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
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

DESIGNS = sys.argv[1] if len(sys.argv) > 1 else "outputs/designs/designed_v3_seqprop.fasta"
FINAL = "model/strict_eval_v3aug/final_model.pt"

def recover_v3_norm():
    """Norm stats of the v3 final model (real f_tr + synthetic augmentation)."""
    pos = read_fasta("Data/raw/multispecies/strict_pool_v3_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v3_negatives_all.fasta")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
    bio = np.vstack([np.load("Data/splits/biophys_strict_v3_pos.npy"),
                     np.load("Data/splits/biophys_strict_v3_neg.npy")]).astype(np.float32)
    dev, _ = train_test_split(np.arange(len(y)), test_size=0.15, random_state=999, stratify=y)
    f_tr, _ = train_test_split(dev, test_size=0.15, random_state=7, stratify=y[dev])
    bio_tr = np.vstack([bio[f_tr], np.load("Data/splits/biophys_synth_train.npy").astype(np.float32)])
    return bio_tr.mean(0), bio_tr.std(0).clip(min=1e-8)

def main():
    set_seed(42); device = setup_device()
    recs = read_fasta(DESIGNS)
    seqs = [s for _, s in recs]; names = [h for h, _ in recs]
    print(f"Re-scoring {len(seqs)} designs from {DESIGNS}")

    fegs_dir = Path(tempfile.mkdtemp(prefix="fegs_designs_"))
    process_fasta(Path(DESIGNS), fegs_dir, topk=10, seq_len=1024, overwrite=True, workers=2)
    paths = list_npz_sorted(str(fegs_dir))

    ext = RNABiophysicalExtractor(normalize=False)
    bio = np.stack([ext._compute_one(s) for s in seqs]).astype(np.float32)
    m, sd = recover_v3_norm()
    bio_n = (bio - m) / sd

    args = HybridTrainArgs(bio_dim=33, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
    model = RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(FINAL, map_location=device, weights_only=True))
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), bio_n, args.max_nucleotides)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
    probs = []
    with torch.no_grad():
        for tk, at, Lh, bi, _ in loader:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device)
            bi = bi.to(device) if bi is not None else None
            lg, _ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)

    # Characterize
    print(f"\n{'#':>3} {'fullP':>6} {'L':>4} {'GC%':>4} {'maxrun':>6} {'tand':>4} {'tri':>4} {'topbase':>7}  motif/sequence preview")
    print("-"*100)
    order = np.argsort(-probs)
    from collections import Counter
    for i in order:
        s = seqs[i]; f = bio[i]
        L = len(s); gc = 100*sum(c in "GC" for c in s)/L
        c = Counter(s); topbase = max(c, key=c.get); topfrac = 100*c[topbase]/L
        # longest mono run
        run=1; mr=1
        for j in range(1,L):
            if s[j]==s[j-1]: run+=1; mr=max(mr,run)
            else: run=1
        print(f"{i:>3} {probs[i]:>6.3f} {L:>4} {gc:>4.0f} {int(f[26]):>6} {int(f[28]):>4} {int(f[31]):>4} {topbase}:{topfrac:>3.0f}%  {s[:46]}")

    print(f"\nFull-model P(LLPS): mean={probs.mean():.3f}  range=[{probs.min():.3f},{probs.max():.3f}]")
    print(f"Designs scoring >0.5 on FULL model: {int((probs>0.5).sum())}/{len(probs)}")
    json.dump({"designs": [{"name": names[i], "seq": seqs[i], "full_model_prob": float(probs[i])} for i in order]},
              open("model/strict_eval_v3aug/designs_rescored.json","w"), indent=2)
    print("Saved -> model/strict_eval_v3aug/designs_rescored.json")

if __name__ == "__main__":
    main()
