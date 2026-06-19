"""Find where ERNIE-RNA hybrid training first produces non-finite loss/weights.
Mirrors run_v6_cv.train_orgbalanced setup but runs a handful of steps with per-step diagnostics."""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts/training")
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils import list_npz_sorted, setup_device, set_seed
from run_v5_final import build_pool_v5

set_seed(42); device = setup_device(); SP = "Data/splits"
build_pool_v5()
groups = np.load(f"{SP}/cluster_groups_v5.npy"); yeast = np.load(f"{SP}/is_yeast_v5.npy")
y = E.G["y"]; N = len(y); all_idx = np.arange(N)
args = HybridTrainArgs(backbone="multimolecule/ernierna", bio_dim=38, use_species_embed=False,
                       unfreeze_last_n=0, freeze_backbone=True, epochs=30, patience=6, lr=1e-4, backbone_lr=5e-6, weight_decay=0.03)
tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
meta = json.load(open(f"{SP}/synthetic_train_meta.json"))
ar = read_fasta(meta["fasta"]); apa = list_npz_sorted(meta["fegs_dir"]); na = min(len(ar), len(apa))
aug = {"seqs": [s for _, s in ar[:na]], "paths": list(apa[:na]),
       "y": np.array(meta["labels"][:na], dtype=int), "bio": np.load(f"{SP}/biophys_v4_synth.npy").astype(np.float32)[:na]}
dev_idx, _ = next(GroupShuffleSplit(1, test_size=0.15, random_state=999).split(all_idx, y, groups))
sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
tr_rel, _ = next(sgkf.split(dev_idx, y[dev_idx], groups[dev_idx])); tr = dev_idx[tr_rel]

TARGET = {0: 0.50, 1: 0.25, 2: 0.25}
seqs_tr = [E.G["seqs"][i] for i in tr] + aug["seqs"]
ptr = [E.G["paths"][i] for i in tr] + aug["paths"]
ytr = np.concatenate([y[tr], aug["y"]])
bio_raw = np.vstack([E.G["bio"][tr], aug["bio"]]); m = bio_raw.mean(0); sd = bio_raw.std(0).clip(min=1e-8)
bio_tr = (bio_raw - m) / sd
grp = np.empty(len(ytr), int); yk = yeast[tr]
for i in range(len(tr)): grp[i] = 0 if y[tr][i] == 0 else (1 if yk[i] else 2)
for j in range(len(aug["y"])): grp[len(tr)+j] = 0 if aug["y"][j] == 0 else 2
cnt = {g: max(int((grp == g).sum()), 1) for g in TARGET}; w = np.array([TARGET[g]/cnt[g] for g in grp], float)
collate = make_collate_fn(tok, topk_m=args.topk_m, fp16_bias=args.fp16_bias)
loader = DataLoader(HybridRNADataset(seqs_tr, ptr, ytr, bio_tr, args.max_nucleotides),
                    batch_size=args.batch_size, sampler=WeightedRandomSampler(w, len(w), replacement=True),
                    num_workers=0, collate_fn=collate, drop_last=True)

model = E.init_model(args, device); opt = model.configure_optimizers(args)
sched = E.make_scheduler(opt, args.epochs * len(loader), args.warmup_frac)
print(f"START debug: {len(loader)} steps/epoch, lr={args.lr} bblr={args.backbone_lr}", flush=True)
model.train(); skipped = 0; nonfinite_loss = 0; NSTEPS = 150
for step, (tk, at, Lh, bi, yb) in enumerate(loader):
    tk = tk.to(device); at = at.to(device); Lh = Lh.to(device); yb = yb.to(device); bi = bi.to(device)
    _, loss = model(tk, at, labels=yb, Lhat_stack=Lh, bio_features=bi)
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        nonfinite_loss += 1; skipped += 1; continue          # forward guard should make this rare
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isfinite(gnorm):
        opt.step()
    else:
        skipped += 1                                          # non-finite grad from ERNIE overflow -> skip
    sched.step()
    pfin = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    if not pfin:
        print(f"[step {step}] PARAMS WENT NON-FINITE despite skip (gnorm={float(gnorm)})", flush=True); break
    if (step + 1) % 25 == 0:
        print(f"[step {step+1}] loss={float(loss):.4f} gradnorm={float(gnorm):7.2f} params_ok=True skipped_so_far={skipped}", flush=True)
    if step >= NSTEPS:
        break
# final: eval-mode validation forward finite? (this is what was all-NaN before)
model.eval()
fin_val = 0; tot = 0
with torch.no_grad():
    for vi, (tk, at, Lh, bi, yb) in enumerate(val_loader if False else [next(iter(loader))]):
        tk=tk.to(device); at=at.to(device); Lh=Lh.to(device); bi=bi.to(device)
        lg,_ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
        fin_val += int(torch.isfinite(lg).all()); tot += 1
print(f"\nDONE: {NSTEPS}+ steps, skipped={skipped} ({nonfinite_loss} nonfinite-loss), "
      f"final params finite={all(bool(torch.isfinite(p).all()) for p in model.parameters())}, "
      f"eval-forward finite={fin_val}/{tot}", flush=True)
