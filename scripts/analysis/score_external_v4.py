"""
External validation re-test for RNAPhaseek v4 (frozen, byte-identical to the v3
external test except the checkpoint, bio_dim=38, and the new norm stats).

Adds the red-team's required guardrails:
  * determinism CANARY: assert _compute_one reproduces the train-time stored vector,
    so train-fit and external-eval featurization cannot silently diverge.
  * feature ABLATION: re-score with the 5 new self-complementarity columns (33-37)
    zeroed. If A-bar/B-bar still drop, the gain came from the backbone learning
    structure (from the structural negatives), not the explicit feature column.
  * per-sequence POSITIVE MARGIN vs v3: every external positive must stay high
    (not just recall) — guards against global score deflation.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python score_external_v4.py
"""
import os, sys, json, tempfile
import numpy as np, torch
import multimolecule  # noqa
sys.path.insert(0, os.getcwd())
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pathlib import Path
from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data   import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils         import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                     import RNABiophysicalExtractor
from Functions.precompute_fegs                     import process_fasta

EXT   = "Data/raw/multispecies/external/external_deleaked.fasta"
FINAL = "model/strict_eval_v4/final_model.pt"
NORM  = "model/strict_eval_v4/norm_stats.npz"
CANARY = "Data/splits/biophys_v4_canary.json"
V3JSON = "model/strict_eval_v3aug/external_validation.json"


def canary_check(ext):
    c = json.load(open(CANARY))
    got = ext._compute_one(c["seq"]).astype(np.float32)
    ref = np.array(c["vec"], dtype=np.float32)
    assert got.shape == ref.shape == (c["n_features"],), f"canary dim {got.shape}"
    md = float(np.max(np.abs(got - ref)))
    assert md < 1e-4, f"CANARY MISMATCH max|delta|={md:.2e} — featurization diverged from train time!"
    print(f"canary OK (max|delta|={md:.2e}, {c['n_features']} dims)")


def score(model, device, tok, args, paths, seqs, bio_norm):
    ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), bio_norm, args.max_nucleotides)
    ld = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
    probs = []
    with torch.no_grad():
        for tk, at, Lh, bi, _ in ld:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device)
            bi = bi.to(device) if bi is not None else None
            lg, _ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def report(tag, probs, labels, names):
    pos_p = probs[labels == 1]; neg_p = probs[labels == 0]
    print(f"\n=== {tag} ===")
    print(f"POSITIVES (n={len(pos_p)}): mean={pos_p.mean():.3f} median={np.median(pos_p):.3f} "
          f"recall@0.5={100*(pos_p>=0.5).mean():.0f}% ({int((pos_p>=0.5).sum())}/{len(pos_p)})")
    print(f"NEGATIVES (n={len(neg_p)}): scores={[f'{x:.3f}' for x in neg_p]} "
          f"rejected@0.5={int((neg_p<0.5).sum())}/{len(neg_p)}")
    return pos_p, neg_p


def main():
    set_seed(42); device = setup_device()
    recs = read_fasta(EXT)
    seqs = [s for _, s in recs]
    labels = np.array([1 if "label=pos" in h else 0 for h, _ in recs])
    names = [h.split("|")[2] if "|" in h else h[:20] for h, _ in recs]
    print(f"External set: {len(seqs)} seqs ({int(labels.sum())} pos / {int((labels==0).sum())} neg)")

    ext = RNABiophysicalExtractor(normalize=False)
    canary_check(ext)

    d = Path(tempfile.mkdtemp(prefix="fegs_extv4_"))
    process_fasta(Path(EXT), d, topk=10, seq_len=1024, overwrite=True, workers=2)
    paths = list_npz_sorted(str(d))
    bio = np.stack([ext._compute_one(s) for s in seqs]).astype(np.float32)
    assert bio.shape[1] == 38, bio.shape

    nz = np.load(NORM); m, sd = nz["mean"].astype(np.float32), nz["std"].astype(np.float32)
    args = HybridTrainArgs(bio_dim=38, use_species_embed=False, unfreeze_last_n=2, freeze_backbone=False)
    model = RNAFMHybridClassifier(args).to(device).eval()
    model.load_state_dict(torch.load(FINAL, map_location=device, weights_only=True))
    tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)

    # FULL v4
    bio_n = (bio - m) / sd
    p_full = score(model, device, tok, args, paths, seqs, bio_n)
    report("EXTERNAL v4 (full, frozen)", p_full, labels, names)

    # ABLATION: zero the 5 new self-complementarity columns (33-37) -> training-mean neutral
    bio_abl = bio_n.copy(); bio_abl[:, 33:38] = 0.0
    p_abl = score(model, device, tok, args, paths, seqs, bio_abl)
    report("EXTERNAL v4 (self-comp features ABLATED)", p_abl, labels, names)

    # AUROC (note small n_neg)
    try:
        au_full = roc_auc_score(labels, p_full); au_abl = roc_auc_score(labels, p_abl)
        print(f"\nAUROC full={au_full:.3f}  ablated={au_abl:.3f}  (n_neg={int((labels==0).sum())}, low power)")
    except Exception as e:
        au_full = au_abl = None; print("AUROC n/a:", e)

    # Per-sequence margin vs v3
    v3 = {}
    if os.path.exists(V3JSON):
        for r in json.load(open(V3JSON)).get("per_seq", []):
            v3[r["name"]] = r["prob"]
    print(f"\n{'name':<22}{'lbl':>4}{'v3':>8}{'v4':>8}{'Δ':>8}{'v4_abl':>9}")
    order = np.argsort(-p_full)
    for i in order:
        v3p = v3.get(names[i], float('nan'))
        print(f"{names[i][:22]:<22}{'pos' if labels[i] else 'NEG':>4}{v3p:>8.3f}{p_full[i]:>8.3f}"
              f"{p_full[i]-v3p:>+8.3f}{p_abl[i]:>9.3f}")

    pos = labels == 1
    json.dump({"n_pos": int(labels.sum()), "n_neg": int((labels==0).sum()),
               "full":    {"pos_mean": float(p_full[pos].mean()), "pos_recall": float((p_full[pos]>=0.5).mean()),
                           "neg_scores": p_full[~pos].tolist(), "auroc": au_full},
               "ablated": {"pos_mean": float(p_abl[pos].mean()), "pos_recall": float((p_abl[pos]>=0.5).mean()),
                           "neg_scores": p_abl[~pos].tolist(), "auroc": au_abl},
               "per_seq": [{"name": names[i], "label": int(labels[i]),
                            "prob_v4": float(p_full[i]), "prob_v4_ablated": float(p_abl[i]),
                            "prob_v3": v3.get(names[i])} for i in range(len(seqs))]},
              open("model/strict_eval_v4/external_validation_v4.json", "w"), indent=2)
    print("\nSaved -> model/strict_eval_v4/external_validation_v4.json")


if __name__ == "__main__":
    main()
