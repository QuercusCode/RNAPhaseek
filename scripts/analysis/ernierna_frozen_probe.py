"""Zero-shot frozen probe: does ERNIE-RNA's representation encode the base-pairing competition
that v6/RNA-FM is blind to? NO fine-tuning, NO new data.

Mechanism under test: in the Williams G4 flank series, an LLPS-NEGATIVE construct has a 5'/3' flank
(C9 or U9) that Watson-Crick / wobble pairs the G-core and SEQUESTERS it -> no LLPS; the matched
LLPS-POSITIVE has an A9 flank that cannot pair the G-core -> core free -> LLPS. v6 (RNA-FM, no
pairing prior) scores both ~identically (0.90 vs 0.92). ERNIE-RNA injects a canonical base-pairing
attention bias (AU=2, CG=3, GU=0.8). If that bias is real signal, ERNIE-RNA's attention should put
HIGH flank<->core attention mass on the C9/U9 negatives and LOW on the A9 positives. RNA-FM (control)
should NOT separate them.

Decisive readout: AUROC of [flank<->core attention mass] predicting LLPS-NEGATIVE among the flanked
constructs, ERNIE-RNA vs RNA-FM. ERNIE >> RNA-FM = the representation contains the missing signal
(green light to integrate); both ~0.5 = the gap is a data problem, not a backbone problem.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/analysis/ernierna_frozen_probe.py
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from sklearn.metrics import roc_auc_score

# ── runtime patch: ernierna modeling uses input_embeds= (typo) vs transformers 5.x inputs_embeds= ──
import multimolecule.models.ernierna.modeling_ernierna as _ern
from transformers.masking_utils import create_bidirectional_mask as _cbm, create_causal_mask as _ccm
def _fix(fn):
    def w(*a, **k):
        if "input_embeds" in k: k["inputs_embeds"] = k.pop("input_embeds")
        return fn(*a, **k)
    return w
_ern.create_bidirectional_mask = _fix(_cbm); _ern.create_causal_mask = _fix(_ccm)

from multimolecule import RnaTokenizer, ErnieRnaModel, RnaFmModel
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta

FASTA = "Data/raw/multispecies/staging/v11_additions.fasta"
DEV = "cpu"   # 26 short seqs; CPU avoids MPS attention quirks

# flanked constructs: (name, label, flank_positions, core_positions) in 0-based SEQ coords.
# core (G3A2)4 = 20 nt; flank = 9 nt. label: 1 = LLPS-positive (core free), 0 = LLPS-negative (core sequestered)
def layout(name, seqlen):
    if name.startswith("5p"):   return list(range(0, 9)), list(range(9, seqlen))           # flank then core
    if name.startswith("3p"):   return list(range(seqlen-9, seqlen)), list(range(0, seqlen-9))  # core then flank
    if name.startswith("dual"): return list(range(0,9))+list(range(seqlen-9, seqlen)), list(range(9, seqlen-9))
    return None, None
# LLPS truth for the flank series (from the papers): A9 flanks condense, C9/U9 sequester -> soluble
FLANK_LLPS = {"5pA9":1,"5pmixed":1,"3pA9":1,"3pmixed":1,         # positives (core free)
              "5pC9":0,"3pC9":0,"3pU9":0,"dual_U9":0,             # negatives (core sequestered)
              "5pU9":0,"dual_mixed":0,"dual_A9":1}                # soft: U9 wobble->treat sequestered; dual_A9 free


def load_model(kind):
    tok = RnaTokenizer.from_pretrained(f"multimolecule/{kind}")
    M = ErnieRnaModel if kind == "ernierna" else RnaFmModel
    kw = dict(attn_implementation="eager")   # eager = returns attention weights (SDPA returns None)
    try:
        model = M.from_pretrained(f"multimolecule/{kind}", tokenizer=tok, **kw)
    except TypeError:
        model = M.from_pretrained(f"multimolecule/{kind}", **kw)
    return tok, model.to(DEV).eval()


def per_token(model, tok, seq):
    """per-nucleotide hidden states (L, D), special tokens stripped."""
    enc = tok(seq, return_tensors="pt").to(DEV)
    assert enc["input_ids"].shape[1] == len(seq) + 2, "unexpected [CLS]+seq+[EOS] layout"
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0]
    return h[1:1+len(seq)].cpu().numpy()                      # (L, D)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def core_perturbation(tokmat, core_idx, bare_core_repr):
    """how far the flank pushes the model's representation of the (identical) G-core away from the
    bare free core. High = the flank changed how the model 'sees' the core (i.e. sequestered it)."""
    core_repr = tokmat[core_idx].mean(0)
    return 1.0 - cos(core_repr, bare_core_repr)


def main():
    recs = read_fasta(FASTA)
    label = [h.split("|")[0] for h, _ in recs]
    name = [h.split("|")[2] for h, _ in recs]
    seq = {name[i]: recs[i][1] for i in range(len(recs))}
    print(f"probe set: {len(recs)} seqs from {FASTA}\nloading models (downloads ERNIE-RNA ~86M if needed)...")
    e_tok, e_model = load_model("ernierna")
    f_tok, f_model = load_model("rnafm")
    print("ERNIE-RNA + RNA-FM loaded (frozen).\n")

    # per-token reps for everything once
    e_tokmat = {n: per_token(e_model, e_tok, seq[n]) for n in seq}
    f_tokmat = {n: per_token(f_model, f_tok, seq[n]) for n in seq}
    e_bare = e_tokmat["G3A2_4"].mean(0); f_bare = f_tokmat["G3A2_4"].mean(0)   # bare free (G3A2)4 core

    # ── decisive test: does the flank perturb the model's representation of the (identical) G-core? ──
    rows = []
    for nm, llps in FLANK_LLPS.items():
        if nm not in seq: continue
        _, co = layout(nm, len(seq[nm]))
        rows.append({"name": nm, "llps_neg": int(llps == 0),
                     "ernie": core_perturbation(e_tokmat[nm], co, e_bare),
                     "rnafm": core_perturbation(f_tokmat[nm], co, f_bare)})
    y = np.array([r["llps_neg"] for r in rows])               # 1 = sequestered / LLPS-negative
    def auroc(key): return float(roc_auc_score(y, [r[key] for r in rows])) if len(set(y)) == 2 else float("nan")

    print("=== CORE-PERTURBATION  (1-cos of flanked-core vs bare free core; higher = flank changed how the model 'sees' the core) ===")
    print(f"{'construct':<12}{'LLPS':>6}{'ERNIE-RNA':>11}{'RNA-FM':>9}")
    for r in sorted(rows, key=lambda x: -x["ernie"]):
        print(f"{r['name']:<12}{'NEG' if r['llps_neg'] else 'pos':>6}{r['ernie']:>11.4f}{r['rnafm']:>9.4f}")
    print(f"\nAUROC [core-perturbation predicts LLPS-NEGATIVE]  (1.0 = perfectly tracks flank sequestration):")
    print(f"  ERNIE-RNA = {auroc('ernie'):.3f}")
    print(f"  RNA-FM    = {auroc('rnafm'):.3f}   (control: no base-pairing prior)")

    # ── supporting: whole-seq matched-pair embedding separation ──
    PAIRS = [("5pA9","5pC9"),("5pmixed","5pC9"),("3pA9","3pC9"),("3pA9","3pU9"),("3pmixed","3pC9"),
             ("G3A2_4","G2A2_4"),("G3A2_4","G3A5_4")]
    e_emb = {n: e_tokmat[n].mean(0) for n in seq}; f_emb = {n: f_tokmat[n].mean(0) for n in seq}
    print(f"\n=== matched-pair whole-seq embedding DISTANCE (1-cos; higher = more separated) ===")
    print(f"{'pos':<13}{'neg':<10}{'ERNIE-RNA':>11}{'RNA-FM':>9}")
    ed, rd = [], []
    for p, n in PAIRS:
        if p not in e_emb or n not in e_emb: continue
        de = 1 - cos(e_emb[p], e_emb[n]); dr = 1 - cos(f_emb[p], f_emb[n]); ed.append(de); rd.append(dr)
        print(f"{p:<13}{n:<10}{de:>11.4f}{dr:>9.4f}")
    print(f"{'MEAN':<23}{np.mean(ed):>11.4f}{np.mean(rd):>9.4f}")

    out = {"core_perturbation": rows,
           "auroc": {"ernie": auroc("ernie"), "rnafm": auroc("rnafm")},
           "pair_embed_dist": {"ernie_mean": float(np.mean(ed)), "rnafm_mean": float(np.mean(rd))}}
    os.makedirs("outputs", exist_ok=True)
    json.dump(out, open("outputs/ernierna_frozen_probe.json", "w"), indent=2)
    print("\nSaved -> outputs/ernierna_frozen_probe.json")
    print("VERDICT GUIDE: ERNIE AUROC >> RNA-FM AUROC (toward 1.0) = ERNIE-RNA's frozen representation already "
          "encodes the flank-sequestration signal v6 is blind to -> integrate it. Both ~0.5 = data problem, not backbone.")


if __name__ == "__main__":
    main()
