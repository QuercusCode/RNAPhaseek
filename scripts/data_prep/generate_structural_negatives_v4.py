"""
Structural hard-negative generator for RNAPhaseek v4.

For every phase-separating POSITIVE P, synthesize a composition-matched twin P-bar
that is identical in mono- AND di-nucleotide composition but has its self-
complementarity (palindromes / kissing-loop stems) destroyed — the property the
frozen v3 model could not read (it scored Fabrini A_bar/B_bar ~0.96, same as the
LLPS positives).

Operator = GLOBAL Altschul-Erickson di-nucleotide-preserving shuffle (Eulerian walk,
last symbol fixed, retry-on-disconnect). This is intentionally a DIFFERENT operator
from the external A_bar/B_bar (surgical KL-loop scramble), so the frozen external test
stays a clean CROSS-OPERATOR generalization probe.

Acceptance gate is sequence-only (no O(n^3) ViennaRNA fold — pool sequences run to
~30 knt): exact mono+di match AND a measured drop in the same self-complementarity
metrics that became biophysical features 33/36/37, computed over the shared
STRUCT_WIN centered window.

Every negative's header encodes parent=<positive index> so downstream splitting can
keep a parent and all its derived negatives in the SAME train/val/test partition
(follow-the-parent — prevents a ~di-identical opposite-label pair crossing the split).

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python generate_structural_negatives_v4.py
"""
import random
import collections
from pathlib import Path
from collections import defaultdict

from Functions.RNA_biophysical.biophysical_features import (
    _rc_selfalign_raw, _rc_kmer_selfpair, _longest_rc_stem, _normalise, STRUCT_WIN,
)

ROOT = Path(__file__).resolve().parents[2]
POS_FASTA = ROOT / "Data/raw/multispecies/strict_pool_v3_positives.fasta"
OUT_FASTA = ROOT / "Data/raw/multispecies/strict_struct_negatives_v4.fasta"
BASE_SEED = 4242
MAX_SEEDS = 25       # di-shuffle attempts (exact di-match — primary, cleanest control)
MONO_SEEDS = 12      # mono-shuffle fallback attempts (Fisher-Yates — 2nd mechanism)


def read_fasta(path):
    """T->U upper-case, mirrors RNAPhaseek_hybrid_data.read_fasta."""
    recs, hdr, parts = [], None, []
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if hdr is not None:
                recs.append((hdr, "".join(parts).upper().replace("T", "U")))
            hdr, parts = line[1:], []
        elif line:
            parts.append(line)
    if hdr is not None:
        recs.append((hdr, "".join(parts).upper().replace("T", "U")))
    return recs


def dinucs(s):
    return [s[i:i + 2] for i in range(len(s) - 1)]


def ae_dishuffle(seq, rng, tries=60):
    """Altschul-Erickson Eulerian walk, last symbol fixed, retry on disconnect.
    Guarantees EXACT mono + di composition. Returns None on repeated failure."""
    if len(seq) < 4:
        return None
    last = seq[-1]
    target_di = collections.Counter(dinucs(seq))
    for _ in range(tries):
        edges = defaultdict(list)
        for i in range(len(seq) - 1):
            edges[seq[i]].append(seq[i + 1])
        for k in edges:
            rng.shuffle(edges[k])
        out, idx, ok = [seq[0]], defaultdict(int), True
        for _ in range(len(seq) - 1):
            h = out[-1]
            if idx[h] >= len(edges[h]):
                ok = False
                break
            out.append(edges[h][idx[h]])
            idx[h] += 1
        cand = "".join(out)
        if (ok and len(cand) == len(seq) and cand[-1] == last
                and collections.Counter(dinucs(cand)) == target_di):
            return cand
    return None


def mono_shuffle(seq, rng):
    """Fisher-Yates shuffle — preserves mono composition only, breaks di-composition.
    Reliably destroys self-complementarity; used as 2nd mechanism when di-shuffle fails."""
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def _window(seq):
    n = len(seq)
    if n > STRUCT_WIN:
        st = (n - STRUCT_WIN) // 2
        return seq[st:st + STRUCT_WIN]
    return seq


def struct_metrics(seq):
    """Self-complementarity over the shared STRUCT_WIN window (sequence-only, fast)."""
    w = _window(seq)
    lw = max(len(w), 1)
    return {
        "align": _rc_selfalign_raw(w) / lw,
        "kmer": _rc_kmer_selfpair(w, 6),
        "stem": _longest_rc_stem(w),
    }


def hamming_frac(a, b):
    return sum(x != y for x, y in zip(a, b)) / max(len(a), 1)


def passes_gate(parent, cand, pm):
    """Structure destroyed enough relative to parent metrics pm (sequence-only)."""
    if cand is None:
        return None
    cm = struct_metrics(cand)
    rel_kmer = (pm["kmer"] - cm["kmer"]) / pm["kmer"] if pm["kmer"] > 0 else 1.0
    stem_drop = pm["stem"] - cm["stem"]
    align_drop = pm["align"] - cm["align"]
    if (rel_kmer >= 0.30 and stem_drop >= 1 and align_drop >= 0.0
            and hamming_frac(cand, parent) >= 0.15):
        return cm
    return None


def main():
    pos = read_fasta(POS_FASTA)
    print(f"loaded {len(pos)} positives from {POS_FASTA.name}")
    out, n_skip_short, n_skip_flat, n_fail = [], 0, 0, 0
    drops = []
    for pidx, (hdr, raw) in enumerate(pos):
        s = _normalise(raw)
        if len(s) < 20:
            n_skip_short += 1
            continue
        pm = struct_metrics(s)
        # parent must have real self-complementarity to destroy, else no informative negative
        if pm["stem"] < 5 and pm["kmer"] < 0.04:
            n_skip_flat += 1
            continue
        gene = hdr.split("|")[1] if "|" in hdr else (hdr.split()[0][:24] if hdr else "NA")
        chosen = None
        # mechanism 1: di-shuffle (exact mono+di match — cleanest control)
        for k in range(MAX_SEEDS):
            rng = random.Random(BASE_SEED + pidx * 100 + k)
            cand = ae_dishuffle(s, rng)
            cm = passes_gate(s, cand, pm)
            if cm is not None:
                chosen = (cand, cm, "dishuf")
                break
        # mechanism 2: mono-shuffle fallback (preserves nucleotide counts; 2nd operator)
        if chosen is None:
            for k in range(MONO_SEEDS):
                rng = random.Random(BASE_SEED + pidx * 100 + 50 + k)
                cand = mono_shuffle(s, rng)
                cm = passes_gate(s, cand, pm)
                if cm is not None:
                    chosen = (cand, cm, "monoshuf")
                    break
        if chosen is None:
            n_fail += 1
            continue
        cand, cm, flavor = chosen
        drops.append((pm, cm, flavor))
        out.append((f"hardneg_struct|parent={pidx}|flavor={flavor}|src={gene}|label=neg", cand))

    with open(OUT_FASTA, "w") as f:
        for h, sq in out:
            f.write(f">{h}\n{sq}\n")

    n_di = sum(1 for d in drops if d[2] == "dishuf")
    n_mono = sum(1 for d in drops if d[2] == "monoshuf")
    print(f"\nwrote {len(out)} structural negatives -> {OUT_FASTA}  ({n_di} di-shuffle, {n_mono} mono-shuffle)")
    print(f"  skipped: {n_skip_short} too-short, {n_skip_flat} unstructured-parent, {n_fail} no-accept")
    if drops:
        import numpy as np
        pk = np.array([d[0]["kmer"] for d in drops]); ck = np.array([d[1]["kmer"] for d in drops])
        ps = np.array([d[0]["stem"] for d in drops]); cs = np.array([d[1]["stem"] for d in drops])
        print(f"  rc_kmer_selfpair  parent {pk.mean():.3f} -> neg {ck.mean():.3f}  (mean drop {(pk-ck).mean():.3f})")
        print(f"  longest_rc_stem   parent {ps.mean():.1f}  -> neg {cs.mean():.1f}   (mean drop {(ps-cs).mean():.1f} bp)")
        # batch audit: di-shuffle must match mono+di exactly; mono-shuffle must match mono
        pos_by_idx = {i: _normalise(r) for i, (_, r) in enumerate(pos)}
        bad_mono = bad_di = 0
        for h, sq in out:
            pidx = int(h.split("parent=")[1].split("|")[0])
            flavor = h.split("flavor=")[1].split("|")[0]
            par = pos_by_idx[pidx]
            if collections.Counter(sq) != collections.Counter(par):
                bad_mono += 1
            if flavor == "dishuf" and collections.Counter(dinucs(sq)) != collections.Counter(dinucs(par)):
                bad_di += 1
        print(f"  audit (n={len(out)}): mono violations={bad_mono}, di-shuffle di violations={bad_di}")


if __name__ == "__main__":
    main()
