#!/usr/bin/env python
"""
RNAPhaseek — command-line tool for RNA-self-LLPS prediction and de novo design.

Wraps the production model (v13: model/strict_eval_v13_production/) and the GA/DEN
generators behind three subcommands:

  score     Predict P(LLPS) for each RNA in a FASTA.
              rnaphaseek score input.fasta -o scores.csv

  design    Generate de novo phase-separating RNA candidates.
              rnaphaseek design --method ga  --n 10 --length 200 -o designs.fasta
              rnaphaseek design --method den --length 200 -o designs.fasta   # diverse library

  validate  Structure-dependence check: score each sequence vs its composition-matched
            scrambles. Delta>0 => the score is structure-driven (trustworthy), not a
            composition artifact.
              rnaphaseek validate designs.fasta -o trust.csv

Run with the project's conda env:
  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python rnaphaseek.py <cmd> ...
"""
import os, sys, csv, json, random, tempfile, argparse, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # project path bootstrap (see paths.py)

DEFAULT_MODEL = "model/strict_eval_v13_production/final_model.pt"
DEFAULT_NORM  = "model/strict_eval_v13_production/norm_stats.npz"
# v6_production, v6_orgbalanced, v7_mil checkpoints are archived to LaCie (only v13 is
# kept locally). --uncertainty and --long-model mil need them; pass --ensemble-from
# <root> or set RNAPHASEEK_ENSEMBLE_ROOT to override the default below.
DEFAULT_ENSEMBLE_ROOT = "/Volumes/LaCie/RNAPhaseek_scripts/model"
MAX_CTX = 1022  # RNA-FM context window; v6 silently truncates sequences longer than this


def _resolve_ensemble_root(a):
    return getattr(a, "ensemble_from", None) or os.environ.get("RNAPHASEEK_ENSEMBLE_ROOT") or DEFAULT_ENSEMBLE_ROOT


def _ensemble_paths(root):
    return [(DEFAULT_MODEL, DEFAULT_NORM),
            (f"{root}/strict_eval_v6_production/final_model.pt",  f"{root}/strict_eval_v6_production/norm_stats.npz"),
            (f"{root}/strict_eval_v6_orgbalanced/final_model.pt", f"{root}/strict_eval_v6_orgbalanced/norm_stats.npz")]


def _mil_paths(root):
    return (f"{root}/strict_eval_v7_mil/final_model.pt", f"{root}/strict_eval_v7_mil/norm_stats.npz")


def _require_files(paths, feature, root):
    miss = [p for p in paths if not os.path.exists(p)]
    if miss:
        sys.exit(
            f"[rnaphaseek] {feature} needs these files but they are missing:\n  " + "\n  ".join(miss) +
            f"\n\nThe v6/v6_orgbalanced/v7_mil checkpoints are archived to {DEFAULT_ENSEMBLE_ROOT}. "
            f"Mount the LaCie drive (or set RNAPHASEEK_ENSEMBLE_ROOT / pass --ensemble-from <root>) "
            f"to point at a directory containing strict_eval_v6_production/, strict_eval_v6_orgbalanced/, "
            f"and strict_eval_v7_mil/.\nCurrent --ensemble-from = {root!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Core scorer (loads the v6 model + normalization once; scores any list of RNAs)
# ─────────────────────────────────────────────────────────────────────────────
class RNAPhaseekScorer:
    def __init__(self, model_path=DEFAULT_MODEL, norm_path=DEFAULT_NORM, quiet=True):
        import torch, multimolecule  # noqa
        from transformers import AutoTokenizer
        from Functions.RNAPhaseek.RNAPhaseek_hybrid        import RNAFMHybridClassifier
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_config import HybridTrainArgs
        from Functions.RNAPhaseek.RNAPhaseek_utils         import setup_device, set_seed
        from Functions.RNA_biophysical                     import RNABiophysicalExtractor
        set_seed(42)
        self.torch = torch
        self.device = setup_device()
        self.args = HybridTrainArgs(bio_dim=38, use_species_embed=False,
                                    unfreeze_last_n=2, freeze_backbone=False)
        self.model = RNAFMHybridClassifier(self.args).to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.tok = AutoTokenizer.from_pretrained(self.args.backbone, trust_remote_code=True)
        nz = np.load(norm_path); self.m = nz["mean"].astype(np.float32); self.sd = nz["std"].astype(np.float32)
        self.ext = RNABiophysicalExtractor(normalize=False)
        if not quiet:
            print(f"[rnaphaseek] loaded {model_path} on {self.device}", file=sys.stderr)

    @staticmethod
    def _norm_seq(s):
        return "".join(c for c in s.upper().replace("T", "U") if not c.isspace())

    def score(self, seqs, batch_size=8):
        """Return P(LLPS) for each sequence (numpy array)."""
        import torch
        from pathlib import Path
        from torch.utils.data import DataLoader
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import HybridRNADataset, make_collate_fn
        from Functions.RNAPhaseek.RNAPhaseek_utils       import list_npz_sorted
        from Functions.precompute_fegs                   import process_fasta
        seqs = [self._norm_seq(s) for s in seqs]
        d = Path(tempfile.mkdtemp(prefix="rnaphaseek_")); fa = d / "in.fasta"
        with open(fa, "w") as f:
            for i, s in enumerate(seqs): f.write(f">s{i}\n{s}\n")
        process_fasta(fa, d, topk=10, seq_len=1024, overwrite=True, workers=4)
        paths = list_npz_sorted(str(d))
        bio = np.stack([self.ext._compute_one(s) for s in seqs]).astype(np.float32)
        bio_n = (bio - self.m) / self.sd
        ds = HybridRNADataset(seqs, paths, np.zeros(len(seqs), int), bio_n, self.args.max_nucleotides)
        ld = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate_fn(self.tok, topk_m=10))
        probs = []
        with torch.no_grad():
            for tk, at, Lh, bi, _ in ld:
                tk = tk.to(self.device); at = at.to(self.device); Lh = Lh.to(self.device)
                bi = bi.to(self.device) if bi is not None else None
                lg, _ = self.model(tk, at, labels=None, Lhat_stack=Lh, bio_features=bi)
                fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
                probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        import shutil; shutil.rmtree(d, ignore_errors=True)
        return np.concatenate(probs)


# ─────────────────────────────────────────────────────────────────────────────
# v7 attention-MIL scorer for LONG (>1022nt) RNAs: tiles into <=1022nt windows
# (stride 512), encodes each with RNA-FM, attention-pools over windows so the
# whole molecule contributes — instead of v6 silently truncating to the first 1022nt.
# (No FEGS bias in the fullseq MIL variant, so this is RNA-FM + biophysics only.)
# ─────────────────────────────────────────────────────────────────────────────
class RNAPhaseekMILScorer:
    def __init__(self, model_path=None, norm_path=None, quiet=True):
        if model_path is None: model_path = f"{DEFAULT_ENSEMBLE_ROOT}/strict_eval_v7_mil/final_model.pt"
        if norm_path  is None: norm_path  = f"{DEFAULT_ENSEMBLE_ROOT}/strict_eval_v7_mil/norm_stats.npz"
        import torch, multimolecule  # noqa
        from transformers import AutoTokenizer
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq        import RNAFMHybridFullSeq
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_config import HybridFullSeqArgs
        from Functions.RNAPhaseek.RNAPhaseek_utils                 import setup_device, set_seed
        from Functions.RNA_biophysical                            import RNABiophysicalExtractor
        set_seed(42)
        self.torch = torch
        self.device = setup_device()
        self.args = HybridFullSeqArgs(bio_dim=38)
        self.model = RNAFMHybridFullSeq(self.args).to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.tok = AutoTokenizer.from_pretrained(self.args.backbone, trust_remote_code=True)
        nz = np.load(norm_path); self.m = nz["mean"].astype(np.float32); self.sd = nz["std"].astype(np.float32)
        self.ext = RNABiophysicalExtractor(normalize=False)
        if not quiet:
            print(f"[rnaphaseek] loaded MIL model {model_path} on {self.device}", file=sys.stderr)

    @staticmethod
    def _norm_seq(s):
        return "".join(c for c in s.upper().replace("T", "U") if not c.isspace())

    def score(self, seqs, batch_size=1):
        torch = self.torch
        from torch.utils.data import DataLoader
        from Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_data import (
            FullSeqRNADataset, make_collate_fn as fs_collate)
        seqs = [self._norm_seq(s) for s in seqs]
        bio = np.stack([self.ext._compute_one(s) for s in seqs]).astype(np.float32)
        bio_n = ((bio - self.m) / self.sd).astype(np.float32)
        coll = fs_collate(self.tok, self.args.window, self.args.stride, self.args.max_windows)
        ld = DataLoader(FullSeqRNADataset(seqs, np.zeros(len(seqs), int), bio_n),
                        batch_size=batch_size, shuffle=False, collate_fn=coll)
        probs = []
        with torch.no_grad():
            for tk, at, wm, bi, _ in ld:
                tk = tk.to(self.device); at = at.to(self.device); wm = wm.to(self.device)
                bi = bi.to(self.device) if bi is not None else None
                lg, _ = self.model(tk, at, wm, labels=None, bio_features=bi)
                fin = torch.isfinite(lg).all(-1, keepdim=True); lg = torch.where(fin, lg, torch.zeros_like(lg))
                probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        return np.concatenate(probs)


def read_fasta(path):
    recs, h, s = [], None, []
    src = sys.stdin if path == "-" else open(path)
    for ln in src:
        ln = ln.rstrip()
        if ln.startswith(">"):
            if h is not None: recs.append((h, "".join(s)))
            h, s = ln[1:], []
        elif ln: s.append(ln)
    if h is not None: recs.append((h, "".join(s)))
    return recs


def _gc(s):
    s = s.upper().replace("T", "U"); n = max(len(s), 1)
    return 100 * sum(c in "GC" for c in s) / n


import contextlib
@contextlib.contextmanager
def _hush():
    """Redirect fd-level stdout to stderr so the underlying libraries' chatter
    (model load, FEGS progress) doesn't pollute the CLI's stdout (CSV/FASTA)."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)          # fd 1 (stdout) -> fd 2 (stderr)
        yield
    finally:
        sys.stdout.flush(); os.dup2(saved, 1); os.close(saved)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommands
# ─────────────────────────────────────────────────────────────────────────────
# 4 independently-trained checkpoints; disagreement = epistemic uncertainty / OOD signal.
# v13 (matched-pair-aware) + v6 + v6_orgbalanced + v7_mil. v6/v6_orgbal/v7_mil are
# resolved at call-time through _resolve_ensemble_root(a) so they can live on LaCie.


def _ensemble_probs(seqs, root, quiet=True):
    """Score with the 4-model ensemble -> [4, n] probabilities (for uncertainty/abstention)."""
    ENSEMBLE = _ensemble_paths(root)
    mil_m, mil_n = _mil_paths(root)
    _require_files([p for pair in ENSEMBLE for p in pair] + [mil_m, mil_n], "--uncertainty mode", root)
    rows = []
    with _hush():
        for mp_, np_ in ENSEMBLE:
            rows.append(RNAPhaseekScorer(mp_, np_, quiet=quiet).score(seqs))
        rows.append(RNAPhaseekMILScorer(mil_m, mil_n, quiet=quiet).score(seqs))
    return np.vstack(rows)


def _cmd_score_uncertainty(a, recs, seqs):
    root = _resolve_ensemble_root(a)
    print(f"[rnaphaseek] uncertainty mode: 4-model ensemble (v13 local + v6/v6_orgbal/v7_mil from {root}) ...", file=sys.stderr)
    ens = _ensemble_probs(seqs, root, quiet=a.quiet)
    mean, std = ens.mean(0), ens.std(0)
    thr = a.abstain_threshold
    rows = [("id", "length", "GC_percent", "P_mean", "ens_std", f"call@{a.threshold}", f"abstain@{thr}")]
    for i, (h, _) in enumerate(recs):
        ab = std[i] > thr
        call = "uncertain" if ab else ("LLPS" if mean[i] >= a.threshold else "no")
        rows.append((h.split()[0], len(seqs[i]), f"{_gc(seqs[i]):.1f}",
                     f"{mean[i]:.4f}", f"{std[i]:.4f}", call, "ABSTAIN" if ab else ""))
    out = sys.stdout if a.out is None else open(a.out, "w", newline="")
    csv.writer(out).writerows(rows)
    n_ab = int((std > thr).sum())
    if a.out: print(f"[rnaphaseek] wrote {len(recs)} ensemble scores -> {a.out}", file=sys.stderr)
    print(f"[rnaphaseek] {n_ab}/{len(recs)} flagged ABSTAIN (ens_std > {thr}: likely out-of-distribution)",
          file=sys.stderr)


def cmd_score(a):
    recs = read_fasta(a.input)
    if not recs: sys.exit("no sequences found in input")
    seqs = [RNAPhaseekScorer._norm_seq(s) for _, s in recs]
    n = len(seqs)
    if getattr(a, "uncertainty", False):
        return _cmd_score_uncertainty(a, recs, seqs)
    short_set = {i for i in range(n) if len(seqs[i]) <= MAX_CTX}
    long_idx = [i for i in range(n) if i not in short_set]
    use_mil = (a.long_model == "mil") and bool(long_idx)
    if long_idx and not use_mil:
        print(f"[rnaphaseek] note: {len(long_idx)} sequence(s) >{MAX_CTX}nt scored on their first "
              f"{MAX_CTX}nt only (held-out tests show no accuracy loss vs full-length). "
              f"Use --long-model mil to read the full length.", file=sys.stderr)
    probs = [0.0] * n; models = [""] * n

    # v6 (production) scores all short RNAs; also long ones if --long-model truncate
    v6_targets = [i for i in range(n) if i in short_set] + ([] if use_mil else long_idx)
    if v6_targets:
        with _hush():
            sc = RNAPhaseekScorer(a.model, a.norm, quiet=a.quiet)
            ps = sc.score([seqs[i] for i in v6_targets])
        for i, p in zip(v6_targets, ps):
            probs[i] = float(p); models[i] = "v13" if i in short_set else "v13_trunc"
    # v7 attention-MIL scores long RNAs full-length (default)
    if use_mil:
        root = _resolve_ensemble_root(a)
        mil_m, mil_n = _mil_paths(root)
        _require_files([mil_m, mil_n], "--long-model mil", root)
        print(f"[rnaphaseek] {len(long_idx)} sequence(s) >{MAX_CTX}nt -> v7 MIL from {root} "
              f"(full-length; use --long-model truncate for the v13 default behavior)", file=sys.stderr)
        with _hush():
            mil = RNAPhaseekMILScorer(mil_m, mil_n, quiet=a.quiet)
            pl = mil.score([seqs[i] for i in long_idx])
        for i, p in zip(long_idx, pl):
            probs[i] = float(p); models[i] = "v7_mil"

    rows = [("id", "length", "GC_percent", "P_LLPS", "model", f"call@{a.threshold}")]
    for i, (h, _) in enumerate(recs):
        rows.append((h.split()[0], len(seqs[i]), f"{_gc(seqs[i]):.1f}", f"{probs[i]:.4f}", models[i],
                     "LLPS" if probs[i] >= a.threshold else "no"))
    out = sys.stdout if a.out is None else open(a.out, "w", newline="")
    csv.writer(out).writerows(rows)
    if a.out: print(f"[rnaphaseek] wrote {n} scores -> {a.out}", file=sys.stderr)
    parr = np.array(probs)
    n_pos = int((parr >= a.threshold).sum())
    note = f" ({len(long_idx)} long via v7 MIL)" if use_mil else ""
    print(f"[rnaphaseek] {n_pos}/{n} predicted LLPS at threshold {a.threshold} "
          f"(mean P={parr.mean():.3f}){note}", file=sys.stderr)


def cmd_design(a):
    if a.method == "den":
        # DEN is gradient-based; delegate to the validated v6 DEN generator.
        print(f"[rnaphaseek] running DEN (diverse library, length={a.length}, v13 model) ...", file=sys.stderr)
        cmd = [sys.executable, "scripts/generation/den_design_v6.py", "--length", str(a.length)]
        if a.out:
            cmd += ["--out", a.out]
        subprocess.run(cmd, check=True)
        dest = a.out or ("outputs/designs/designed_den_v6.fasta" if a.length == 200
                         else f"outputs/designs/designed_den_{a.length}nt.fasta")
        print(f"[rnaphaseek] DEN designs -> {dest}", file=sys.stderr)
        return
    # ── GA: optimize the v6 scorer's P(LLPS) (full-pipeline fitness) ──
    with _hush():
        sc = RNAPhaseekScorer(a.model, a.norm, quiet=a.quiet)
    L, POP, GEN, ELITE, MUT, TOURN = a.length, 64, a.generations, 10, 0.04, 3
    rng = random.Random(a.seed); BASES = "ACGU"
    pop = ["".join(rng.choice(BASES) for _ in range(L)) for _ in range(POP)]
    cache = {}
    for g in range(GEN):
        new = [s for s in pop if s not in cache]
        if new:
            with _hush():
                scored = sc.score(new)
            for s, p in zip(new, scored): cache[s] = float(p)
        fit = [cache[s] for s in pop]
        order = sorted(range(POP), key=lambda i: -fit[i])
        print(f"[rnaphaseek] GA gen {g+1}/{GEN}: best={fit[order[0]]:.4f} mean={np.mean(fit):.4f}", file=sys.stderr)
        elites = [pop[i] for i in order[:ELITE]]; newpop = list(elites)
        def tour():
            c = rng.sample(range(POP), TOURN); return pop[max(c, key=lambda i: fit[i])]
        while len(newpop) < POP:
            x, y = tour(), tour(); i, j = sorted(rng.sample(range(1, L), 2))
            child = x[:i] + y[i:j] + x[j:]
            child = "".join(rng.choice(BASES) if rng.random() < MUT else ch for ch in child)
            newpop.append(child)
        pop = newpop
    top = sorted(set(pop), key=lambda s: -cache.get(s, 0))[:a.n]
    out = sys.stdout if a.out is None else open(a.out, "w")
    for i, s in enumerate(top): out.write(f">rnaphaseek_ga_{i}_P{cache[s]:.3f}\n{s}\n")
    if a.out: print(f"[rnaphaseek] wrote {len(top)} GA designs -> {a.out}", file=sys.stderr)


def cmd_validate(a):
    from generate_structural_negatives_v4 import ae_dishuffle, mono_shuffle
    recs = read_fasta(a.input)
    flat, owner = [], []
    for gi, (h, s) in enumerate(recs):
        s = RNAPhaseekScorer._norm_seq(s)
        flat.append(s); owner.append((gi, True))
        for k in range(a.k):
            rng = random.Random(100 + k)
            flat.append(ae_dishuffle(s, rng) or mono_shuffle(s, rng)); owner.append((gi, False))
    with _hush():
        sc = RNAPhaseekScorer(a.model, a.norm, quiet=a.quiet)
        probs = sc.score(flat)
    rows = [("id", "P_design", "P_scramble_mean", "Delta", "verdict")]
    for gi, (h, s) in enumerate(recs):
        di = [i for i, (g, d) in enumerate(owner) if g == gi and d][0]
        scr = [i for i, (g, d) in enumerate(owner) if g == gi and not d]
        delta = probs[di] - probs[scr].mean()
        rows.append((h.split()[0], f"{probs[di]:.4f}", f"{probs[scr].mean():.4f}", f"{delta:+.4f}",
                     "structure-driven" if delta > 0.1 else ("weak" if delta > 0 else "composition-artifact")))
    out = sys.stdout if a.out is None else open(a.out, "w", newline="")
    csv.writer(out).writerows(rows)
    if a.out: print(f"[rnaphaseek] wrote {len(recs)} validations -> {a.out}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(prog="rnaphaseek", description="RNA-self-LLPS prediction & de novo design (v13 model).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model checkpoint (default: v13 production)")
    p.add_argument("--norm", default=DEFAULT_NORM, help="biophysics norm_stats.npz")
    p.add_argument("--ensemble-from", default=None, dest="ensemble_from",
                   help="root dir containing strict_eval_v6_production / strict_eval_v6_orgbalanced / "
                        f"strict_eval_v7_mil (needed for --uncertainty and --long-model mil; "
                        f"default {DEFAULT_ENSEMBLE_ROOT}, also overridable via RNAPHASEEK_ENSEMBLE_ROOT)")
    p.add_argument("--quiet", action="store_true", help="suppress model-load chatter")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="predict P(LLPS) per RNA in a FASTA")
    s.add_argument("input", help="FASTA file (or '-' for stdin)")
    s.add_argument("-o", "--out", help="output CSV (default: stdout)")
    s.add_argument("-t", "--threshold", type=float, default=0.5)
    s.add_argument("--long-model", choices=["truncate", "mil"], default="truncate",
                   help=f"how to score RNAs >{MAX_CTX}nt: truncate=v6 on the first 1022nt (default; "
                        "held-out tests show no accuracy loss); mil=v7 attention-MIL full-length "
                        "(opt-in; no accuracy gain, for full-length transparency)")
    s.add_argument("--uncertainty", action="store_true",
                   help="4-model ensemble: report ens_std + ABSTAIN to flag out-of-distribution inputs")
    s.add_argument("--abstain-threshold", type=float, default=0.05,
                   help="ens_std above this -> ABSTAIN (default 0.05; in-dist designs ~0.02, OOD kissing-loops ~0.15)")
    s.set_defaults(func=cmd_score)

    d = sub.add_parser("design", help="generate de novo phase-separating RNA")
    d.add_argument("--method", choices=["ga", "den"], default="ga", help="ga=one optimal design+variants; den=diverse library")
    d.add_argument("--n", type=int, default=10, help="number of top designs to return (GA)")
    d.add_argument("--length", type=int, default=200)
    d.add_argument("--generations", type=int, default=40)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("-o", "--out", help="output FASTA (default: stdout)")
    d.set_defaults(func=cmd_design)

    v = sub.add_parser("validate", help="structure-dependence (design vs scramble) trustworthiness check")
    v.add_argument("input", help="FASTA of designs to validate")
    v.add_argument("-k", type=int, default=3, help="scrambles per sequence")
    v.add_argument("-o", "--out", help="output CSV (default: stdout)")
    v.set_defaults(func=cmd_validate)

    a = p.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
