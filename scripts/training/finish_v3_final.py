"""
Finish the v3 eval: the 5-fold CV completed (saved in cv_progress.json) but the
final model + locked test were killed mid-training. This re-trains ONLY the final
model on all dev and scores the locked test, reusing the eval module's functions.
"""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed

set_seed(42); device = setup_device()
OUT = "model/strict_eval_v3aug"

# Load v3 pool into the eval module's global G
pos = read_fasta("Data/raw/multispecies/strict_pool_v3_positives.fasta")
neg = read_fasta("Data/raw/multispecies/strict_pool_v3_negatives_all.fasta")
pp = list_npz_sorted("Data/processed/fegs_topk_strict_v3_pos")
npn = list_npz_sorted("Data/processed/fegs_topk_strict_v3_neg")
npos = min(len(pos), len(pp)); nneg = min(len(neg), len(npn))
E.G["seqs"]  = [s for _, s in pos[:npos]] + [s for _, s in neg[:nneg]]
E.G["hdrs"]  = [h for h, _ in pos[:npos]] + [h for h, _ in neg[:nneg]]
E.G["paths"] = list(pp[:npos]) + list(npn[:nneg])
E.G["y"]     = np.concatenate([np.ones(npos), np.zeros(nneg)]).astype(int)
E.G["bio"]   = np.vstack([np.load("Data/splits/biophys_strict_v3_pos.npy"),
                          np.load("Data/splits/biophys_strict_v3_neg.npy")]).astype(np.float32)
N = len(E.G["y"])

args = HybridTrainArgs(bio_dim=33, use_species_embed=False, unfreeze_last_n=2,
                       freeze_backbone=False, epochs=30, patience=6,
                       lr=1e-4, backbone_lr=5e-6, weight_decay=0.03)
tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

# Augmentation (same as the v3 run)
meta = json.load(open("Data/splits/synthetic_train_meta.json"))
ar = read_fasta(meta["fasta"]); ap = list_npz_sorted(meta["fegs_dir"]); na = min(len(ar), len(ap))
E.G["aug"] = {"seqs": [s for _, s in ar[:na]], "paths": list(ap[:na]),
              "y": np.array(meta["labels"][:na], dtype=int),
              "bio": np.load(meta["bio"]).astype(np.float32)[:na]}

# Reproduce the EXACT splits the eval used
all_idx = np.arange(N)
dev_idx, test_idx = train_test_split(all_idx, test_size=0.15, random_state=999, stratify=E.G["y"])
f_tr, f_va = train_test_split(dev_idx, test_size=0.15, random_state=7, stratify=E.G["y"][dev_idx])
print(f"N={N} dev={len(dev_idx)} locked_test={len(test_idx)} final_train={len(f_tr)} final_val={len(f_va)}", flush=True)

# Train the final model on all dev (inner val for early-stopping), score locked test
model, fm, fsd, _ = E.train_model(f_tr, f_va, args, device, tok, tag="final")
os.makedirs(OUT, exist_ok=True)
torch.save(model.state_dict(), f"{OUT}/final_model.pt")
tpr, tlb, thd = E.score_with(model, test_idx, fm, fsd, args, device, tok)
au = roc_auc_score(tlb, tpr); pr = average_precision_score(tlb, tpr)
acc = accuracy_score(tlb, (tpr >= 0.5).astype(int))
print(f"\n*** LOCKED-TEST AUROC = {au:.4f}  PR-AUC = {pr:.4f}  acc@0.5 = {acc*100:.1f}%  (n={len(tlb)}) ***", flush=True)
E.diagnostic(tpr, tlb, thd, "Locked test")

cv = json.load(open(f"{OUT}/cv_progress.json"))["fold_scores"]
json.dump({"cv_mean_auroc": float(np.mean(cv)), "cv_std_auroc": float(np.std(cv)), "cv_fold_scores": cv,
           "locked_test_auroc": float(au), "locked_test_prauc": float(pr),
           "n_test": int(len(tlb)), "n_dev": int(len(dev_idx)),
           "config": {"unfreeze_last_n": 2, "bio_dim": 33, "pool": "v3_retiered_experimental"}},
          open(f"{OUT}/eval_summary.json", "w"), indent=2)
print(f"Saved -> {OUT}/eval_summary.json", flush=True)
