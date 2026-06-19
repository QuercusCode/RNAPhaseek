"""
v7 experiment: tiling + attention-MIL to recover the 74% of long-positive sequence
truncated by RNA-FM's 1022-nt window. Uses the existing (previously-abandoned) fullseq
attention-MIL model — each RNA is sliced into <=1022-nt windows (stride 512, <=32
windows), every window encoded by frozen RNA-FM + the adapter, then ATTENTION-POOLED
over windows (the model learns which windows carry LLPS signal), fused with the
width-38 biophysics, classified.

Organism-balanced (P(neg)=.5/yeast-pos=.25/nonyeast-pos=.25) so it STACKS on v6's win and
isolates the MIL-vs-truncation effect. Same v5 pool, cluster-grouped dev/test split, seeds.
Trains one final model, scores the SAME locked test -> overall + yeast + non-yeast AUROC
vs v5 (0.860/0.714) and v6 (0.827/0.749).

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python run_v7_mil.py
"""
import os, sys, json
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
import paths  # project path bootstrap (see paths.py)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
import Functions.RNAPhaseek.RNAPhaseek_hybrid_eval as E
from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq import RNAFMHybridFullSeq
from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_config import HybridFullSeqArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_data import (
    FullSeqRNADataset, make_collate_fn as fs_collate)
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta
from Functions.RNAPhaseek.RNAPhaseek_utils import setup_device, set_seed
from run_v5_final import subset_auroc

OUT = "model/strict_eval_v7_mil"
SP = "Data/splits"
TARGET = {0: 0.50, 1: 0.25, 2: 0.25}


def build_v5_seqs():
    pos = read_fasta("Data/raw/multispecies/strict_pool_v5_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v5_negatives_all.fasta")
    sneg = read_fasta("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
    seqs = [s for _, s in pos] + [s for _, s in neg] + [s for _, s in sneg]
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg)), np.zeros(len(sneg))]).astype(int)
    bio = np.vstack([np.load(f"{SP}/biophys_v5_pos.npy"), np.load(f"{SP}/biophys_v5_neg.npy"),
                     np.load(f"{SP}/biophys_v4_structneg.npy")]).astype(np.float32)
    is_struct = np.array([False]*(len(pos)+len(neg)) + [True]*len(sneg))
    return seqs, y, bio, is_struct


@torch.no_grad()
def score(model, seqs, bio_n, device, collate, bs=1):
    ds = FullSeqRNADataset(seqs, np.zeros(len(seqs), int), bio_n)
    ld = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    probs = []
    for tk, at, wm, bi, _ in ld:
        tk = tk.to(device); at = at.to(device); wm = wm.to(device)
        bi = bi.to(device) if bi is not None else None
        lg, _ = model(tk, at, wm, labels=None, bio_features=bi)
        fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
        probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    set_seed(42); device = setup_device(); os.makedirs(OUT, exist_ok=True)
    seqs, y, bio, is_struct = build_v5_seqs()
    groups = np.load(f"{SP}/cluster_groups_v5.npy"); yeast = np.load(f"{SP}/is_yeast_v5.npy")
    N = len(y); all_idx = np.arange(N)
    args = HybridFullSeqArgs(bio_dim=38, epochs=30, patience=6)
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    collate = fs_collate(tok, args.window, args.stride, args.max_windows)

    # synth aug (short repeats; full sequence = 1 window)
    meta = json.load(open(f"{SP}/synthetic_train_meta.json"))
    ar = read_fasta(meta["fasta"]); na = len(ar)
    aug_seqs = [s for _, s in ar]; aug_y = np.array(meta["labels"][:na], dtype=int)
    aug_bio = np.load(f"{SP}/biophys_v4_synth.npy").astype(np.float32)[:na]

    dev_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=999).split(all_idx, y, groups))
    f_tr, f_va = next(GroupShuffleSplit(1, test_size=0.15, random_state=7).split(dev_idx, y[dev_idx], groups[dev_idx]))
    f_tr, f_va = dev_idx[f_tr], dev_idx[f_va]

    tr_seqs = [seqs[i] for i in f_tr] + aug_seqs
    ytr = np.concatenate([y[f_tr], aug_y])
    bio_tr_raw = np.vstack([bio[f_tr], aug_bio])
    m = bio_tr_raw.mean(0); sd = bio_tr_raw.std(0).clip(min=1e-8)
    bio_tr = (bio_tr_raw - m) / sd
    grp = np.empty(len(ytr), int); yk = yeast[f_tr]
    for i in range(len(f_tr)): grp[i] = 0 if y[f_tr][i] == 0 else (1 if yk[i] else 2)
    for j in range(len(aug_y)): grp[len(f_tr)+j] = 0 if aug_y[j] == 0 else 2
    cnt = {g: max(int((grp == g).sum()), 1) for g in TARGET}
    w = np.array([TARGET[g]/cnt[g] for g in grp], float)
    print(f"train: neg={cnt[0]} yeast-pos={cnt[1]} nonyeast-pos={cnt[2]} | windows/RNA capped at {args.max_windows}", flush=True)

    train_ds = FullSeqRNADataset(tr_seqs, ytr, bio_tr)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    va_seqs = [seqs[i] for i in f_va]; bio_va = (bio[f_va] - m) / sd
    val_loader = DataLoader(FullSeqRNADataset(va_seqs, y[f_va], bio_va), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)

    model = RNAFMHybridFullSeq(args).to(device)
    opt = model.configure_optimizers(args)
    sched = E.make_scheduler(opt, args.epochs * max(1, len(train_loader)), args.warmup_frac)
    best_au, best_state, pat = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        for tk, at, wm, bi, yb in train_loader:
            tk = tk.to(device); at = at.to(device); wm = wm.to(device)
            yb = yb.to(device); bi = bi.to(device) if bi is not None else None
            _, loss = model(tk, at, wm, labels=yb, bio_features=bi)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        # val AUROC
        model.eval()
        vp = score(model, va_seqs, bio_va, device, collate)
        vau = roc_auc_score(y[f_va], vp)
        if vau > best_au + 1e-4:
            best_au = vau; pat = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        print(f"    [v7-mil] ep{ep+1}/{args.epochs} val={vau:.4f} (best={max(best_au,0):.4f} pat {pat}/{args.patience})", flush=True)
        if pat >= args.patience:
            print(f"    [v7-mil] early stop @ ep{ep+1}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"{OUT}/final_model.pt")
    np.savez(f"{OUT}/norm_stats.npz", mean=m, std=sd)

    # locked test
    model.eval()
    tp = score(model, [seqs[i] for i in test_idx], (bio[test_idx]-m)/sd, device, collate)
    tlb = y[test_idx]
    tau = roc_auc_score(tlb, tp); tacc = accuracy_score(tlb, (tp >= 0.5).astype(int))
    tny, tnyn = subset_auroc(tp, tlb, ~yeast[test_idx])
    tyy, tyn = subset_auroc(tp, tlb, yeast[test_idx])
    v5 = json.load(open("model/strict_eval_v5/eval_summary.json"))
    v6 = json.load(open("model/strict_eval_v6_orgbalanced/eval_summary.json"))
    print(f"\n*** v7 MIL (organism-balanced) LOCKED TEST ***", flush=True)
    print(f"  overall AUROC   = {tau:.4f}   (v5={v5['locked_test_auroc']:.3f} v6={v6['locked_test_auroc']:.3f})")
    print(f"  yeast-pos AUROC = {tyy}   (n={tyn})")
    print(f"  NON-YEAST AUROC = {tny}   (n={tnyn})   v5={v5['locked_test_nonyeast_auroc']:.3f} v6={v6['locked_test_nonyeast_auroc']:.3f}")
    print(f"  acc@0.5         = {tacc*100:.1f}%")
    json.dump({"locked_test_auroc": float(tau), "locked_test_acc": float(tacc),
               "locked_test_yeast_auroc": tyy, "locked_test_nonyeast_auroc": tny,
               "v5_locked_auroc": v5["locked_test_auroc"], "v5_locked_nonyeast": v5["locked_test_nonyeast_auroc"],
               "v6_locked_auroc": v6["locked_test_auroc"], "v6_locked_nonyeast": v6["locked_test_nonyeast_auroc"],
               "model": "attention-MIL fullseq, organism-balanced, window=1022 stride=512 max_windows=32"},
              open(f"{OUT}/eval_summary.json", "w"), indent=2)
    print(f"Saved -> {OUT}/eval_summary.json", flush=True)


if __name__ == "__main__":
    main()
