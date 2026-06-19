"""
Deep Exploration Network (DEN) for DIVERSE de novo LLPS RNA design.

SeqProp and the GA both mode-collapsed to a single attractor. DEN trains a
GENERATOR network G(z) (z ~ N(0,I)) to output sequences that simultaneously
(a) maximize the v3 model's P(LLPS) and (b) are mutually DIVERSE — via a
pairwise-similarity penalty across each batch (Linder et al. 2020, Cell Systems).

Gradient-based, so fitness uses the differentiable RNA-FM+adapter proxy
(bio-zero), same as SeqProp; final designs are re-scored with the FULL model.

  python den_design.py
"""
import os, sys, json, tempfile
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import multimolecule  # noqa
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
from Functions.generator_hybrid import load_hybrid_for_generation, get_nt_embeddings, RNA_BASES
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta, HybridRNADataset, make_collate_fn
from Functions.RNAPhaseek.RNAPhaseek_utils    import list_npz_sorted, setup_device, set_seed
from Functions.RNA_biophysical                import RNABiophysicalExtractor
from Functions.precompute_fegs                import process_fasta

FINAL = "model/strict_eval_v3aug/final_model.pt"
L, ZDIM, BATCH, STEPS, LR, LAMBDA = 200, 64, 12, 400, 1e-3, 4.0


def main():
    set_seed(0); device = setup_device()
    model = load_hybrid_for_generation(FINAL, device)
    for p in model.parameters(): p.requires_grad_(False)
    W_NT, cls_emb, eos_emb = get_nt_embeddings(model, device)
    bio_zero = torch.zeros(1, model.args.bio_dim, device=device)

    class Gen(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(ZDIM, 256), nn.ReLU(),
                                  nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, L*4))
        def forward(s, z): return s.net(z).view(-1, L, 4)

    def batched_fitness(P_soft):           # (B,L,4) -> (B,)
        Bsz = P_soft.shape[0]
        E = P_soft @ W_NT                  # (B,L,640)
        ie = torch.cat([cls_emb.expand(Bsz, 1, -1), E, eos_emb.expand(Bsz, 1, -1)], 1)
        am = torch.ones(Bsz, L+2, dtype=torch.long, device=device)
        x = model.backbone(inputs_embeds=ie, attention_mask=am).last_hidden_state
        kpm = am.bool()
        for blk in model.adapter: x = blk(x, bias_per_head=None, key_padding_mask=kpm)
        x = model.adapter_ln(x)
        pooled = (x * am.float().unsqueeze(-1)).sum(1) / am.float().sum(1, keepdim=True)
        if model.bio_proj is not None:
            pooled = torch.cat([pooled, model.bio_proj(bio_zero.expand(Bsz, -1))], -1)
        return torch.softmax(model.head(pooled), -1)[:, 1]

    G = Gen().to(device)
    opt = torch.optim.Adam(G.parameters(), lr=LR)
    hist = []
    for step in range(STEPS):
        z = torch.randn(BATCH, ZDIM, device=device)
        logits = G(z)
        tau = max(0.4, 2.0 * (1 - step / STEPS))
        P = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        fit = batched_fitness(P)
        flat = F.normalize(P.reshape(BATCH, -1), dim=1)
        sim = flat @ flat.t()
        off = ~torch.eye(BATCH, dtype=torch.bool, device=device)
        div = sim[off].mean()                          # avg pairwise similarity
        loss = -fit.mean() + LAMBDA * div
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            hist.append((step, float(fit.mean()), float(div)))
            print(f"step {step:>3}/{STEPS}: fitness={fit.mean():.4f}  pairwise_sim={div:.3f}", flush=True)

    # ── Sample diverse designs, decode, dedup ──
    G.eval()
    with torch.no_grad():
        z = torch.randn(160, ZDIM, device=device)
        idx = G(z).argmax(-1)
    seqs = ["".join(RNA_BASES[i] for i in row.tolist()) for row in idx]
    seqs = list(dict.fromkeys(seqs))                   # dedup exact
    print(f"\nSampled {len(seqs)} unique sequences")

    # ── Re-score with the FULL model + measure diversity ──
    pos = read_fasta("Data/raw/multispecies/strict_pool_v3_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v3_negatives_all.fasta")
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)
    bioall = np.vstack([np.load("Data/splits/biophys_strict_v3_pos.npy"),
                        np.load("Data/splits/biophys_strict_v3_neg.npy")]).astype(np.float32)
    dev, _ = train_test_split(np.arange(len(y)), test_size=0.15, random_state=999, stratify=y)
    f_tr, _ = train_test_split(dev, test_size=0.15, random_state=7, stratify=y[dev])
    btr = np.vstack([bioall[f_tr], np.load("Data/splits/biophys_synth_train.npy").astype(np.float32)])
    m, sd = btr.mean(0), btr.std(0).clip(min=1e-8)
    ext = RNABiophysicalExtractor(normalize=False)
    tok = AutoTokenizer.from_pretrained(model.args.backbone, trust_remote_code=True)
    d = Path(tempfile.mkdtemp(prefix="fegs_den_")); fa = d / "s.fasta"
    with open(fa, "w") as f:
        for i, s in enumerate(seqs): f.write(f">s{i}\n{s}\n")
    process_fasta(fa, d, topk=10, seq_len=1024, overwrite=True, workers=4)
    paths = list_npz_sorted(str(d))
    bio = np.stack([ext._compute_one(s) for s in seqs]).astype(np.float32)
    ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), (bio - m) / sd, model.args.max_nucleotides)
    ld = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=make_collate_fn(tok, topk_m=10))
    probs = []
    with torch.no_grad():
        for tk, at, Lh, bi, _ in ld:
            tk = tk.to(device); at = at.to(device); Lh = Lh.to(device); bi = bi.to(device) if bi is not None else None
            lg, _ = model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
            fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)

    # Diversity = mean pairwise Hamming identity among top designs
    order = np.argsort(-probs)
    top = [seqs[i] for i in order[:15]]
    def ident(a, b): return sum(x == y for x, y in zip(a, b)) / len(a)
    ids = [ident(top[i], top[j]) for i in range(len(top)) for j in range(i+1, len(top))]
    mean_ident = float(np.mean(ids)) if ids else 1.0

    print(f"\n=== DEN designs: full-model P(LLPS) mean={probs.mean():.3f} max={probs.max():.3f} ===")
    print(f"Diversity: mean pairwise identity among top-15 = {mean_ident:.2f}  (lower=more diverse)")
    print(f"(SeqProp/GA top designs were ~0.9+ identical; DEN target is much lower)")
    print(f"\n{'P':>6} {'GC%':>4} {'U%':>4}  preview")
    for i in order[:10]:
        s = seqs[i]; c = Counter(s); n = len(s)
        print(f"{probs[i]:>6.3f} {100*(c['G']+c['C'])/n:>4.0f} {100*c['U']/n:>4.0f}  {s[:48]}")

    with open("outputs/designs/designed_den.fasta", "w") as f:
        for i in order[:15]: f.write(f">den_design_{i}_P{probs[i]:.3f}\n{seqs[i]}\n")
    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(h[:, 0], h[:, 1], "-o", ms=3, color="#8e44ad", label="mean fitness (proxy)")
    ax2 = ax.twinx(); ax2.plot(h[:, 0], h[:, 2], "-s", ms=3, color="#e67e22", label="pairwise similarity")
    ax.set_xlabel("DEN training step"); ax.set_ylabel("fitness", color="#8e44ad")
    ax2.set_ylabel("pairwise similarity (lower=diverse)", color="#e67e22")
    ax.set_title("Figure 14 — DEN training: fitness up, diversity maintained", fontweight="bold")
    fig.savefig("report_assets/fig14_den.png", dpi=140, bbox_inches="tight"); plt.close()
    json.dump({"mean_full_prob": float(probs.mean()), "max_full_prob": float(probs.max()),
               "mean_pairwise_identity_top15": mean_ident, "n_unique": len(seqs),
               "top": [{"seq": seqs[i], "P": float(probs[i])} for i in order[:15]]},
              open("model/strict_eval_v3aug/den_summary.json", "w"), indent=2)
    print("\nSaved -> designed_den.fasta, report_assets/fig14_den.png")


if __name__ == "__main__":
    main()
