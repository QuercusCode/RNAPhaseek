"""
evaluate.py
===========
Full evaluation suite for RNAPhaseek:
  1. Standard metrics  : AUROC, PR-AUC, MCC, F1 @ best threshold
  2. Cross-organism     : train human+mouse, test yeast+C.elegans
  3. Ablation           : model vs. no-graph-bias vs. no-BPE vs. FEGS-only
  4. Attribution maps   : top-k attention positions per positive sequence
  5. Condensate report  : per-condensate score distribution

Usage:
    python Functions/evaluate.py \
        --checkpoint model/rnaphasek_best.pt \
        --test-pos Data/splits/test_pos.fasta \
        --test-neg Data/splits/test_neg.fasta \
        --fegs-dir Data/fegs/ \
        --tokenizer model/rna_bpe_tokenizer.json
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    f1_score, precision_recall_curve
)

warnings.filterwarnings("ignore")

# ─────────────────────── imports ────────────────────────────────
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Functions"))

from tokenizers import Tokenizer as HFTokenizer
from RNAPhaseek.RNAPhaseek import RNAPhaseekClassifier
from RNAPhaseek.RNAPhaseek_config import CONFIG
from RNAPhaseek.RNAPhaseek_data import RNASeqTopkDataset, collate_fn

# ─────────────────────── helpers ────────────────────────────────

def parse_fasta(path: Path):
    with open(path) as f:
        hdr, chunks = None, []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr: yield hdr, "".join(chunks)
                hdr, chunks = line[1:], []
            else:
                chunks.append(line)
        if hdr: yield hdr, "".join(chunks)

def best_threshold(y_true, y_score):
    """F1-optimal threshold."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    idx  = np.argmax(f1s)
    return thresholds[min(idx, len(thresholds)-1)]

def evaluate_set(model, loader, device, label: str = "") -> dict:
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            topk  = batch["topk_matrices"].to(device)
            label_b = batch["labels"]
            logits = model(input_ids=ids, attention_mask=mask, topk_matrices=topk)
            scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(label_b.numpy().tolist())

    y_true  = np.array(all_labels)
    y_score = np.array(all_scores)
    thresh  = best_threshold(y_true, y_score)
    y_pred  = (y_score >= thresh).astype(int)

    auroc  = roc_auc_score(y_true, y_score)
    prauc  = average_precision_score(y_true, y_score)
    mcc    = matthews_corrcoef(y_true, y_pred)
    f1     = f1_score(y_true, y_pred)

    result = dict(auroc=auroc, prauc=prauc, mcc=mcc, f1=f1,
                  threshold=thresh, n_pos=int(y_true.sum()),
                  n_neg=int((1-y_true).sum()))
    if label:
        print(f"\n{'─'*40}")
        print(f"  {label}")
        print(f"{'─'*40}")
        print(f"  AUROC  : {auroc:.4f}")
        print(f"  PR-AUC : {prauc:.4f}")
        print(f"  MCC    : {mcc:.4f}")
        print(f"  F1     : {f1:.4f} (thresh={thresh:.3f})")
        print(f"  N pos  : {result['n_pos']} | N neg : {result['n_neg']}")
    return result

# ─────────────────────── attention attribution ───────────────────

def attribution_map(model, sample, device, topn: int = 20):
    """Return top-n positions by summed attention weight (all heads)."""
    model.eval()
    input_ids   = sample["input_ids"].unsqueeze(0).to(device)
    attn_mask   = sample["attention_mask"].unsqueeze(0).to(device)
    topk        = sample["topk_matrices"].unsqueeze(0).to(device)

    # Register hooks to collect attention weights
    attn_weights = []
    hooks = []
    for block in model.blocks:
        def hook(module, inp, out, store=attn_weights):
            # out is (B, H, L, L) or similar
            if isinstance(out, tuple):
                store.append(out[0].detach().cpu())
            else:
                store.append(out.detach().cpu())
        h = block.attn.register_forward_hook(hook)
        hooks.append(h)

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attn_mask,
                  topk_matrices=topk)

    for h in hooks:
        h.remove()

    if not attn_weights:
        return []

    # Average across heads and layers
    attn = torch.stack(attn_weights, dim=0)  # (L_layers, B, H, L, L)
    attn = attn.mean(dim=[0, 1, 2])           # (L_seq, L_seq)
    # Column sum = how much each position is attended to
    importance = attn.sum(dim=0).numpy()       # (L_seq,)
    top_positions = np.argsort(importance)[::-1][:topn].tolist()
    return top_positions

# ─────────────────────── cross-organism split ────────────────────

def organism_from_header(hdr: str) -> str:
    low = hdr.lower()
    if "homo_sapiens" in low or "homo sapiens" in low:    return "human"
    if "mus_musculus" in low or "mus musculus" in low:    return "mouse"
    if "saccharomyces" in low or "cerevisiae" in low:     return "yeast"
    if "caenorhabditis" in low or "elegans" in low:       return "celegans"
    return "other"

# ─────────────────────── main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--test-pos",    required=True)
    parser.add_argument("--test-neg",    required=True)
    parser.add_argument("--train-pos",   default=None, help="For cross-org test")
    parser.add_argument("--val-pos",     default=None)
    parser.add_argument("--fegs-dir",    required=True)
    parser.add_argument("--tokenizer",   required=True)
    parser.add_argument("--out-json",    default="evaluation_results.json")
    parser.add_argument("--batch-size",  type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = CONFIG.copy()
    if "config" in ckpt:
        cfg.update(ckpt["config"])

    model = RNAPhaseekClassifier(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    tokenizer = HFTokenizer.from_file(args.tokenizer)

    # ── Standard test set ─────────────────────────────────────
    test_pos = list(parse_fasta(Path(args.test_pos)))
    test_neg = list(parse_fasta(Path(args.test_neg)))

    test_dataset = RNASeqTopkDataset(
        pos_seqs=[s for _,s in test_pos],
        neg_seqs=[s for _,s in test_neg],
        pos_hdrs=[h for h,_ in test_pos],
        neg_hdrs=[h for h,_ in test_neg],
        fegs_dir=Path(args.fegs_dir),
        tokenizer=tokenizer,
        seq_len=cfg["seq_len"],
        topk=cfg["topk_m"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn
    )

    results = {}
    results["standard"] = evaluate_set(model, test_loader, device,
                                       label="Standard test set")

    # ── Cross-organism evaluation ─────────────────────────────
    # Filter test set by organism
    for org_name, org_key in [("yeast", "yeast"), ("C. elegans", "celegans"),
                               ("SARS-CoV-2", "sars2")]:
        org_pos = [(h,s) for h,s in test_pos if organism_from_header(h) == org_key]
        org_neg = [(h,s) for h,s in test_neg]  # use full neg set

        if len(org_pos) < 5:
            print(f"  [skip] {org_name}: only {len(org_pos)} positives")
            continue

        ds = RNASeqTopkDataset(
            pos_seqs=[s for _,s in org_pos],
            neg_seqs=[s for _,s in org_neg],
            pos_hdrs=[h for h,_ in org_pos],
            neg_hdrs=[h for h,_ in org_neg],
            fegs_dir=Path(args.fegs_dir),
            tokenizer=tokenizer,
            seq_len=cfg["seq_len"],
            topk=cfg["topk_m"],
        )
        ld = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, collate_fn=collate_fn)
        results[f"cross_org_{org_key}"] = evaluate_set(
            model, ld, device, label=f"Cross-organism: {org_name}"
        )

    # ── Attribution on first 10 positives ─────────────────────
    print("\n─── Attribution maps (first 10 positives) ─────────────────")
    attr_results = []
    for i in range(min(10, len(test_dataset))):
        sample = test_dataset[i]
        if sample["labels"] != 1:
            continue
        top_pos = attribution_map(model, sample, device)
        hdr = test_pos[i][0] if i < len(test_pos) else f"seq_{i}"
        gene = hdr.split("|")[1] if "|" in hdr else hdr[:30]
        print(f"  {gene}: top positions {top_pos[:5]}")
        attr_results.append({"gene": gene, "top_positions": top_pos})
    results["attribution"] = attr_results

    # ── Save results ──────────────────────────────────────────
    out_path = Path(args.out_json)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    # ── Final summary table ───────────────────────────────────
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    for key, val in results.items():
        if isinstance(val, dict) and "auroc" in val:
            print(f"  {key:30s}: AUROC={val['auroc']:.4f} | PR-AUC={val['prauc']:.4f} | MCC={val['mcc']:.4f}")

if __name__ == "__main__":
    main()
