"""
Decision gate before any v5 retrain: can a base-pair-resolution feature separate
the external surgical-KL controls (A vs A_bar, B vs B_bar) that v4 could not reject?

The mechanistic hypothesis: condensation here is driven by KISSING LOOPS — hairpin
loops whose sequences are reverse-complementary so loops on different molecules pair.
Scrambling the 6-nt loop palindrome (A_bar) abolishes loop-loop pairing while leaving
stems/length/composition intact, so GLOBAL self-complementarity barely moves (why v4
failed) but a LOOP-LOOP complementarity score should collapse.

Candidate features (all ViennaRNA-based, computed over a bounded window):
  kl_loop_comp   : MFE fold -> hairpin loops -> best loop-loop revcomp run (the KL signal)
  loop_self_pal  : best palindrome WITHIN any single loop
  ens_paired     : partition-function ensemble paired fraction
  ens_entropy    : mean positional entropy of the ensemble (structural ambiguity)

Pass criterion: a feature must separate BOTH (A vs A_bar / B vs B_bar) AND
(training positive vs its structural-negative shuffle). Otherwise features-alone
cannot close the external gap and we report that instead of retraining blindly.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python validate_kl_feature.py
"""
import numpy as np
import RNA

WIN = 800
_COMP = {"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"}


def _win(s):
    s = s.upper().replace("T", "U")
    return s if len(s) <= WIN else s[(len(s) - WIN) // 2:(len(s) - WIN) // 2 + WIN]


def revcomp(s):
    return "".join(_COMP.get(c, "N") for c in reversed(s))


def max_comp_run(a, b):
    """Longest contiguous antiparallel WC-complementary run between a and b
    (a[i..] pairs b[j..] reading b backward). O(|a||b|)."""
    br = revcomp(b)
    best = 0
    la, lb = len(a), len(br)
    for i in range(la):
        for j in range(lb):
            k = 0
            while i + k < la and j + k < lb and a[i + k] == br[j + k] and a[i + k] != "N":
                k += 1
            best = max(best, k)
    return best


def hairpin_loops(seq):
    struct, mfe = RNA.fold_compound(seq).mfe()
    pt = RNA.ptable(struct)
    n = len(struct)
    loops = []
    for i in range(1, n + 1):
        j = pt[i]
        if j > i and all(pt[k] == 0 for k in range(i + 1, j)):
            loops.append(seq[i:j - 1])   # 1-indexed (i+1..j-1) -> 0-indexed slice
    return loops, struct, mfe


def features(raw):
    seq = _win(raw)
    n = max(len(seq), 1)
    loops, struct, mfe = hairpin_loops(seq)
    # kl_loop_comp: best loop-loop complementarity across all loop pairs (incl. self)
    kl = 0
    for a in range(len(loops)):
        for b in range(a, len(loops)):
            kl = max(kl, max_comp_run(loops[a], loops[b]))
    # loop_self_pal: best internal palindrome within a single loop
    selfpal = max((max_comp_run(L, L) for L in loops), default=0)
    # ensemble features (partition function)
    fc = RNA.fold_compound(seq)
    _, _ = fc.mfe()
    pf_struct, gfe = fc.pf()
    bpp = np.array(fc.bpp())  # (n+1, n+1) upper triangle pair probs
    ens_paired = float(bpp.sum() * 2) / n if bpp.size else 0.0
    # positional entropy
    try:
        ent = np.array([fc.positional_entropy()[i] for i in range(1, n + 1)])
        ens_entropy = float(ent.mean())
    except Exception:
        ens_entropy = float("nan")
    return dict(kl_loop_comp=kl, loop_self_pal=selfpal,
                ens_paired=ens_paired, ens_entropy=ens_entropy,
                n_loops=len(loops), mfe=mfe)


def read_named(path):
    out = {}
    h = None; s = []
    for ln in open(path):
        ln = ln.rstrip()
        if ln.startswith(">"):
            if h is not None: out[h] = "".join(s)
            h = ln[1:]; s = []
        elif ln: s.append(ln)
    if h is not None: out[h] = "".join(s)
    return out


def main():
    ext = read_named("Data/raw/multispecies/external/external_deleaked.fasta")
    def find(name):
        for h, s in ext.items():
            if h.split("|")[2] == name if "|" in h else False:
                return s
        return None
    pairs = [("A", "A_bar"), ("B", "B_bar")]
    print("=== EXTERNAL surgical-KL controls (the v4 failure) ===")
    print(f"{'seq':<8}{'kl_loop':>9}{'self_pal':>9}{'ens_pair':>9}{'ens_ent':>9}{'nloops':>7}{'mfe':>8}")
    for pos, neg in pairs:
        for nm in (pos, neg):
            s = find(nm)
            if s is None:
                print(f"{nm:<8} (not found)"); continue
            f = features(s)
            print(f"{nm:<8}{f['kl_loop_comp']:>9}{f['loop_self_pal']:>9}{f['ens_paired']:>9.3f}"
                  f"{f['ens_entropy']:>9.3f}{f['n_loops']:>7}{f['mfe']:>8.1f}")
        print()

    # training positives vs their structural-neg shuffle (does the feature also separate these?)
    pos = read_named("Data/raw/multispecies/strict_pool_v3_positives.fasta")
    pos_list = list(pos.items())
    sneg = read_named("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
    print("=== TRAINING positive vs its structural-neg shuffle (kl_loop_comp) ===")
    print(f"{'parent':>7}{'pos_kl':>8}{'neg_kl':>8}{'pos_selfpal':>12}{'neg_selfpal':>12}")
    diffs = []
    for h, ns in list(sneg.items())[:14]:
        pidx = int(h.split("parent=")[1].split("|")[0])
        ps = pos_list[pidx][1]
        pf = features(ps); nf = features(ns)
        diffs.append((pf["kl_loop_comp"], nf["kl_loop_comp"]))
        print(f"{pidx:>7}{pf['kl_loop_comp']:>8}{nf['kl_loop_comp']:>8}"
              f"{pf['loop_self_pal']:>12}{nf['loop_self_pal']:>12}")
    d = np.array(diffs)
    print(f"\nmean kl_loop_comp: pos {d[:,0].mean():.1f}  vs  shuffle-neg {d[:,1].mean():.1f}  "
          f"(separates if pos > neg)")


if __name__ == "__main__":
    main()
