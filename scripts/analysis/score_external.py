"""External validation: score the de-leaked external designed-RNA-condensate set
with the FROZEN v3 model (no retraining, no threshold re-fit)."""
import os, sys, json, tempfile
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils         import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                     import RNABiophysicalExtractor
from Functions.precompute_fegs                     import process_fasta

EXT = "Data/raw/multispecies/external/external_deleaked.fasta"
FINAL = "model/strict_eval_v3aug/final_model.pt"

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
    set_seed(42); device = setup_device()
    recs = read_fasta(EXT)
    seqs = [s for _, s in recs]
    labels = np.array([1 if "label=pos" in h else 0 for h, _ in recs])
    names = [h.split("|")[2] if "|" in h else h[:20] for h, _ in recs]
    srcs  = [h.split("|")[1] if "|" in h else "?" for h, _ in recs]
    print(f"External set: {len(seqs)} seqs ({int(labels.sum())} pos / {int((labels==0).sum())} neg)")

    d = Path(tempfile.mkdtemp(prefix="fegs_ext_"))
    process_fasta(Path(EXT), d, topk=10, seq_len=1024, overwrite=True, workers=2)
    paths = list_npz_sorted(str(d))
    ext = RNABiophysicalExtractor(normalize=False)
    bio = np.stack([ext._compute_one(s) for s in seqs]).astype(np.float32)
    m, sd = recover_norm()

    args = HybridTrainArgs(bio_dim=33, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
    model = RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(FINAL, map_location=device, weights_only=True))
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), (bio - m) / sd, args.max_nucleotides)
    ld = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
    probs = []
    with torch.no_grad():
        for tk, at, Lh, bi, _ in ld:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device); bi = bi.to(device) if bi is not None else None
            lg, _ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)

    pos_p = probs[labels == 1]; neg_p = probs[labels == 0]
    print(f"\n=== EXTERNAL VALIDATION (frozen v3 model) ===")
    print(f"POSITIVES (n={len(pos_p)}): mean P(LLPS)={pos_p.mean():.3f}  median={np.median(pos_p):.3f}  "
          f"recall@0.5={100*(pos_p>=0.5).mean():.0f}%  ({int((pos_p>=0.5).sum())}/{len(pos_p)} called LLPS)")
    print(f"NEGATIVES (n={len(neg_p)}): scores={[f'{x:.3f}' for x in neg_p]}  "
          f"rejected@0.5={int((neg_p<0.5).sum())}/{len(neg_p)}")
    if len(set(labels)) > 1 and (labels==1).sum() and (labels==0).sum():
        try: print(f"AUROC (note: only {int((labels==0).sum())} negatives) = {roc_auc_score(labels, probs):.3f}")
        except Exception as e: print(f"AUROC n/a: {e}")
    print(f"\nPer-sequence (sorted by score):")
    print(f"{'P':>6} {'label':>5} {'src':>12} {'len':>4}  name")
    for i in np.argsort(-probs):
        print(f"{probs[i]:>6.3f} {'pos' if labels[i] else 'NEG':>5} {srcs[i]:>12} {len(seqs[i]):>4}  {names[i]}")
    json.dump({"n_pos": int(labels.sum()), "n_neg": int((labels==0).sum()),
               "pos_mean_prob": float(pos_p.mean()), "pos_recall_0.5": float((pos_p>=0.5).mean()),
               "neg_scores": neg_p.tolist(),
               "per_seq": [{"name": names[i], "src": srcs[i], "label": int(labels[i]), "prob": float(probs[i])}
                           for i in range(len(seqs))]},
              open("model/strict_eval_v3aug/external_validation.json", "w"), indent=2)
    print("\nSaved -> model/strict_eval_v3aug/external_validation.json")

if __name__ == "__main__":
    main()
