"""
Diagnostic for the strict-pool model.

Answers: WHERE does the model fail?
  - Easy task:  positives vs matched-transcriptome negatives (val-held-out AUROC)
  - Hard task:  positives vs adversarial hard negatives (sub-threshold repeats,
                TERRA, composition shuffles)
  - Full-pool mean-prob breakdown by group/category (diagnostic; includes train
    data so it's optimistic, but reveals the failure PATTERN)

Run:
  python -m Functions.RNAPhaseek.diagnose_strict
"""
import os
import sys

import numpy as np
import torch
import multimolecule  # noqa: F401 — registers RNA-FM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from Functions.RNAPhaseek.RNAPhaseek_hybrid          import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config   import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data     import read_fasta, make_collate_fn, HybridRNADataset
from Functions.RNAPhaseek.RNAPhaseek_utils           import list_npz_sorted, setup_device, set_seed
from Functions.RNAPhaseek.species_registry           import species_id_for, label_for

FASTA_POS = "Data/raw/multispecies/strict_pool_positives.fasta"
FASTA_NEG = "Data/raw/multispecies/strict_pool_negatives_all.fasta"
SRC_POS   = "Data/processed/fegs_topk_strict_pos"
SRC_NEG   = "Data/processed/fegs_topk_strict_neg"
BIO_POS   = "Data/splits/biophys_strict_pos.npy"
BIO_NEG   = "Data/splits/biophys_strict_neg.npy"
CKPT      = "model/strict/hybrid_strict_best.pt"


def pos_category(hdr: str) -> str:
    h = hdr.lower()
    if "repeat_expansion" in h or "(cag)" in h or "(cug)" in h or "(cgg)" in h or "(ggggcc)" in h or "(g4c2)" in h:
        return "repeat/designed"
    if "lncrna_scaffold" in h: return "lncRNA"
    if "viral" in h or "pmid:" in h and "virus" in h: return "viral"
    if "mapped_utr" in h or "utr" in h: return "UTR"
    if "g4_telomeric" in h: return "G4"
    return "other"


def neg_subtype(hdr: str) -> str:
    h = hdr.lower()
    if "hardneg_terra" in h: return "hard:TERRA"
    if "hardneg_shuffle" in h: return "hard:shuffle"
    if "hardneg_" in h: return "hard:subthreshold"
    return "matched"


def score(model, seqs, paths, bio, device, tokenizer, args):
    """Return predicted P(LLPS) for each record."""
    y_dummy = np.zeros(len(seqs), dtype=np.int64)
    ds = HybridRNADataset(seqs, paths, y_dummy, bio, args.max_nucleotides)
    collate = make_collate_fn(tokenizer, topk_m=args.topk_m)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
    out = []
    with torch.no_grad():
        for token_ids, att, Lhat, biob, _ in loader:
            token_ids = token_ids.to(device); att = att.to(device); Lhat = Lhat.to(device)
            biob = biob.to(device) if biob is not None else None
            logits, _ = model(token_ids, att, labels=None, Lhat_stack=Lhat, bio_features=biob)
            finite = torch.isfinite(logits).all(-1, keepdim=True)
            logits = torch.where(finite, logits, torch.zeros_like(logits))
            p = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            out.append(p)
    return np.concatenate(out)


def main():
    set_seed(42)
    device = setup_device()

    # ── Load records (FASTA order, matching trainer) ──
    pos_recs = read_fasta(FASTA_POS); neg_recs = read_fasta(FASTA_NEG)
    pos_paths = list_npz_sorted(SRC_POS); neg_paths = list_npz_sorted(SRC_NEG)
    n_pos = min(len(pos_recs), len(pos_paths)); n_neg = min(len(neg_recs), len(neg_paths))
    pos_recs, pos_paths = pos_recs[:n_pos], pos_paths[:n_pos]
    neg_recs, neg_paths = neg_recs[:n_neg], neg_paths[:n_neg]

    pos_seqs = [s for _, s in pos_recs]; pos_hdrs = [h for h, _ in pos_recs]
    neg_seqs = [s for _, s in neg_recs]; neg_hdrs = [h for h, _ in neg_recs]

    all_seqs  = pos_seqs + neg_seqs
    all_paths = list(pos_paths) + list(neg_paths)
    all_hdrs  = pos_hdrs + neg_hdrs
    y = np.concatenate([np.ones(len(pos_seqs)), np.zeros(len(neg_seqs))]).astype(int)
    bio = np.vstack([np.load(BIO_POS), np.load(BIO_NEG)]).astype(np.float32)

    # ── Replicate the trainer's val split (random_state=42, stratify=y) ──
    (s_tr, s_va, p_tr, p_va, h_tr, h_va,
     y_tr, y_va, b_tr, b_va) = train_test_split(
        all_seqs, all_paths, all_hdrs, y, bio,
        test_size=0.15, random_state=42, stratify=y)
    # Bio normalisation: trainer computed from the train fold
    m = b_tr.mean(0); sd = b_tr.std(0).clip(min=1e-8)
    b_va_n = (b_va - m) / sd

    # ── Model ──
    args = HybridTrainArgs(bio_dim=26, use_species_embed=False)
    model = RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

    # ── Score the VAL set (clean held-out) ──
    probs_va = score(model, s_va, p_va, b_va_n, device, tokenizer, args)
    y_va = np.array(y_va)
    sub_va = np.array([neg_subtype(h) if yy == 0 else "positive" for h, yy in zip(h_va, y_va)])

    print("\n" + "=" * 64)
    print("VAL SET (held-out) — n =", len(y_va))
    print("=" * 64)
    n_hard_va = int(((sub_va == "hard:TERRA") | (sub_va == "hard:shuffle") | (sub_va == "hard:subthreshold")).sum())
    print(f"  positives={int((y_va==1).sum())}  matched_neg={int((sub_va=='matched').sum())}  hard_neg={n_hard_va}")

    # Easy task: positives vs matched negatives only
    easy = (sub_va == "positive") | (sub_va == "matched")
    if (y_va[easy] == 1).sum() and (y_va[easy] == 0).sum():
        au_easy = roc_auc_score(y_va[easy], probs_va[easy])
        pr_easy = average_precision_score(y_va[easy], probs_va[easy])
        print(f"\n  EASY task  (pos vs matched-transcriptome neg):  AUROC={au_easy:.4f}  PR-AUC={pr_easy:.4f}")
    # Overall val
    au_all = roc_auc_score(y_va, probs_va)
    print(f"  FULL  val  (pos vs all neg):                     AUROC={au_all:.4f}")

    # ── Full-pool mean-prob breakdown (diagnostic; includes train data) ──
    probs_pos = score(model, pos_seqs, list(pos_paths), (bio[:len(pos_seqs)] - m) / sd, device, tokenizer, args)
    probs_neg = score(model, neg_seqs, list(neg_paths), (bio[len(pos_seqs):] - m) / sd, device, tokenizer, args)

    print("\n" + "=" * 64)
    print("FULL-POOL mean P(LLPS) by group  (incl. train; optimistic but shows the pattern)")
    print("=" * 64)
    # positives by category
    pcat = np.array([pos_category(h) for h in pos_hdrs])
    print("  POSITIVES (should be HIGH):")
    for c in sorted(set(pcat)):
        msk = pcat == c
        print(f"    {c:<18} n={int(msk.sum()):<4} mean_prob={probs_pos[msk].mean():.3f}")
    print(f"    {'ALL POS':<18} n={len(probs_pos):<4} mean_prob={probs_pos.mean():.3f}")

    # negatives by subtype
    nsub = np.array([neg_subtype(h) for h in neg_hdrs])
    print("\n  NEGATIVES (should be LOW):")
    for c in sorted(set(nsub)):
        msk = nsub == c
        print(f"    {c:<18} n={int(msk.sum()):<4} mean_prob={probs_neg[msk].mean():.3f}")

    # ── Key diagnostic: can it separate pos from EACH neg type? (full pool) ──
    print("\n" + "=" * 64)
    print("SEPARABILITY (full pool): positives vs each negative type")
    print("=" * 64)
    for c in sorted(set(nsub)):
        msk = nsub == c
        if msk.sum() < 3:
            print(f"    pos vs {c:<18} n_neg={int(msk.sum())}  (too few)")
            continue
        yy = np.concatenate([np.ones(len(probs_pos)), np.zeros(int(msk.sum()))])
        pp = np.concatenate([probs_pos, probs_neg[msk]])
        au = roc_auc_score(yy, pp)
        print(f"    pos vs {c:<18} n_neg={int(msk.sum()):<4} AUROC={au:.4f}")


if __name__ == "__main__":
    main()
