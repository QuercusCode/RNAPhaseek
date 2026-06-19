"""
Unified merge for the multi-species RNAPhaseek training pool.

Combines:
  - Existing single-species human positives + negatives  (from Phase 1)
  - All multispecies positives  (mouse, yeast, celegans, drosophila, virus)
  - All multispecies species-matched negatives

Outputs:
  Data/raw/multispecies/unified_all_positives.fasta
  Data/raw/multispecies/unified_all_negatives.fasta
  Data/raw/multispecies/unified_species_stats.json

Then runs CD-HIT-EST 90% to deduplicate within each side; the dedup is
done within positives and within negatives separately (not cross-class,
to preserve real positive-negative pairs at the same sequence-identity
threshold used for the Phase 1 pool).

Each surviving record carries a species label parseable from the FASTA
header via Functions.RNAPhaseek.species_registry.species_id_for().

Quality + leak guards applied during merge:
  - Drop sequences shorter than SEQ_MIN
  - Drop sequences with N-fraction > 0.20
  - Drop sequences dominated by a single base (> 0.80)
  - Drop exact-sequence duplicates (canonical-hash) within and across files
  - Reject any negative whose canonical-hash collides with a positive
    (label-leak guard)

Run:
    python -m Functions.data_collection_multispecies.unified_merge
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
MS_DIR     = Path("Data/raw/multispecies")
NEG_DIR    = MS_DIR / "negatives"
OUT_POS    = MS_DIR / "unified_all_positives.fasta"
OUT_NEG    = MS_DIR / "unified_all_negatives.fasta"
STATS_OUT  = MS_DIR / "unified_species_stats.json"

# Existing human pool (from Phase 1)
HUMAN_POS  = Path("Data/raw/all_positives_dedup.fasta")
HUMAN_NEG  = Path("Data/raw/negatives_ensembl.fasta")

# Multispecies inputs (positives)
MS_POS_FILES = [
    ("smoops_mouse_positives.fasta",  "mouse"),
    ("yeast_positives.fasta",         "yeast"),
    ("celegans_positives.fasta",      "celegans"),
    ("drosophila_positives.fasta",    "drosophila"),
    ("viral_positives.fasta",         "virus"),
    ("spombe_positives.fasta",        "spombe"),
    ("trypanosoma_positives.fasta",   "trypanosoma"),
    ("arabidopsis_positives.fasta",   "arabidopsis"),
    ("rice_positives.fasta",          "rice"),
]
# Multispecies inputs (negatives)
MS_NEG_FILES = [
    ("mouse_negatives.fasta",         "mouse"),
    ("yeast_negatives.fasta",         "yeast"),
    ("celegans_negatives.fasta",      "celegans"),
    ("drosophila_negatives.fasta",    "drosophila"),
    ("spombe_negatives.fasta",        "spombe"),
    ("trypanosoma_negatives.fasta",   "trypanosoma"),
    ("arabidopsis_negatives.fasta",   "arabidopsis"),
    ("rice_negatives.fasta",          "rice"),
]

SEQ_MIN              = 50
MAX_N_FRAC           = 0.20
MAX_SINGLE_BASE_FRAC = 0.80

# Strict protein-only amino-acid 1-letter codes — never present in RNA.
# Used to reject sequences that are actually proteins masquerading as RNA
# (e.g. legacy `pos_train_seqN` records mixed in from an older Phaseek
#  protein-only training set). Any sequence whose RAW form contains > 5%
# of these characters is dropped BEFORE normalise() strips them down to
# nonsense AUGCN fragments.
_PROTEIN_ONLY_AA = set("MKLRIQFPWYDHSE")
MAX_PROTEIN_AA_FRAC  = 0.05


def looks_like_protein(seq_raw: str) -> bool:
    """Return True if seq_raw contains > MAX_PROTEIN_AA_FRAC of amino-acid
    1-letter codes that are never present in RNA (M/K/L/R/I/Q/F/P/W/Y/D/H/S/E).
    Run on the RAW sequence (before normalise stripping)."""
    if not seq_raw:
        return False
    s = seq_raw.upper()
    n_aa = sum(1 for c in s if c in _PROTEIN_ONLY_AA)
    return n_aa / len(s) > MAX_PROTEIN_AA_FRAC


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalise(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    return re.sub(r"[^AUGCN]", "", seq)


def parse_fasta(path: Path):
    if not path.exists():
        return
    hdr, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(chunks)
                hdr, chunks = line[1:], []
            else:
                chunks.append(line)
        if hdr is not None:
            yield hdr, "".join(chunks)


def ensure_species_in_header(hdr: str, species_hint: str, kind: str = "llps") -> str:
    """Prefix the header with {kind}_{species}| if not already tagged.

    If the header already carries a parseable species (via species_id_for and
    label_for is not 'unknown'), KEEP that species rather than the hint. This
    prevents llps_human| from masking non-human records that already live in
    Data/raw/all_positives_dedup.fasta.
    """
    from Functions.RNAPhaseek.species_registry import species_id_for, label_for
    detected = label_for(species_id_for(hdr))
    chosen   = detected if detected != "unknown" else species_hint
    prefix   = f"{kind}_{chosen}|"
    if hdr.startswith(prefix):
        return hdr
    # If hdr already starts with kind_<other_species>|, leave as-is
    if re.match(rf"^{kind}_(human|mouse|yeast|celegans|drosophila|virus|arabidopsis|unknown|zebrafish|xenopus|rat|rice|spombe|trypanosoma)\|", hdr):
        return hdr
    return f"{prefix}{hdr}"


def passes_quality(seq_n: str) -> tuple[bool, str]:
    """Return (ok, reason). seq_n is already normalised to AUGCN."""
    L = len(seq_n)
    if L < SEQ_MIN:
        return False, "too_short"
    n_count = seq_n.count("N")
    if L > 0 and n_count / L > MAX_N_FRAC:
        return False, "too_many_N"
    # All-N (or empty non-N content)
    if L > 0 and n_count == L:
        return False, "all_N"
    # Single-base dominance
    c = Counter(seq_n)
    if c and L > 0 and max(c.values()) / L > MAX_SINGLE_BASE_FRAC:
        return False, "single_base_dominated"
    return True, ""


def _canon_hash(seq_n: str) -> str:
    return hashlib.sha1(seq_n.encode()).hexdigest()


def _resolve_cdhit_bin(cdhit_bin: str = "cd-hit-est") -> str | None:
    """Locate cd-hit-est.

    Look in this order:
      1. The bin/ directory next to sys.executable (active conda/mamba env)
      2. shutil.which(cdhit_bin) on PATH
      3. ~/.local/bin/cd-hit-est
    Returns the resolved absolute path, or None if not found.
    """
    env_bin = Path(sys.executable).parent / cdhit_bin
    if env_bin.exists() and os.access(env_bin, os.X_OK):
        return str(env_bin)
    which_bin = shutil.which(cdhit_bin)
    if which_bin:
        return which_bin
    local_bin = Path(os.path.expanduser("~/.local/bin")) / cdhit_bin
    if local_bin.exists() and os.access(local_bin, os.X_OK):
        return str(local_bin)
    return None


def run_cdhit(input_fa: Path, output_fa: Path, identity: float = 0.90,
              cdhit_bin: str = "cd-hit-est") -> bool:
    """Run cd-hit-est. Returns True on success, False if cd-hit-est
    was missing or failed (and the input was copied verbatim as a fallback)."""
    resolved = _resolve_cdhit_bin(cdhit_bin)
    if resolved is None:
        print(f"  [warn] {cdhit_bin} not found; skipping dedup, using raw concat.")
        shutil.copyfile(input_fa, output_fa)
        return False
    # cd-hit-est word size: must be 8/9/10 for >=0.90 identity
    word_size = "8" if identity < 0.93 else "10"
    cmd = [
        resolved, "-i", str(input_fa), "-o", str(output_fa),
        "-c", f"{identity:.2f}", "-n", word_size,
        "-T", "0", "-M", "0", "-d", "0",
    ]
    print(f"  running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        stderr_snip = (e.stderr or "")[:500]
        print(f"  [error] cd-hit-est failed: {stderr_snip}")
        shutil.copyfile(input_fa, output_fa)
        return False
    except FileNotFoundError as e:
        print(f"  [error] cd-hit-est not executable: {e}")
        shutil.copyfile(input_fa, output_fa)
        return False


def write_combined(out_path: Path, sources, exclude_hashes: set | None = None):
    """sources: list of (path, species_label, kind).
       exclude_hashes: optional set of canonical seq hashes to skip
       (label-leak guard, e.g. block positives appearing as negatives).

       Also performs intra-write seq-hash dedup and applies passes_quality.

    Returns: (n_written, species_counts, drop_counts, seen_hashes)
    """
    from Functions.RNAPhaseek.species_registry import species_id_for, label_for
    exclude_hashes = exclude_hashes or set()
    seen_hashes: set = set()
    n_written = 0
    species_counts: Counter = Counter()
    drop_counts:    Counter = Counter()
    with open(out_path, "w") as out:
        for src_path, species, kind in sources:
            if not src_path.exists():
                print(f"  [skip] {src_path} not found")
                drop_counts["missing_file"] += 1
                continue
            n_src = 0
            for hdr, seq in parse_fasta(src_path):
                # Protein-contamination guard: reject sequences whose RAW form
                # contains amino-acid codes not present in RNA. Catches the
                # legacy pos_train_seqN / pos_val_seqN / pos_test_seqN records
                # that leaked in from an older protein-only training set.
                if looks_like_protein(seq):
                    drop_counts["protein_contamination"] += 1
                    continue
                seq_n = normalise(seq)
                ok, reason = passes_quality(seq_n)
                if not ok:
                    drop_counts[f"quality_{reason}"] += 1
                    continue
                h = _canon_hash(seq_n)
                if h in exclude_hashes:
                    drop_counts["label_leak"] += 1
                    continue
                if h in seen_hashes:
                    drop_counts["exact_duplicate"] += 1
                    continue
                seen_hashes.add(h)
                new_hdr = ensure_species_in_header(hdr, species, kind)
                # Re-derive the species LABEL we actually wrote (for stats)
                actual_species = label_for(species_id_for(new_hdr))
                out.write(f">{new_hdr}\n{seq_n}\n")
                species_counts[actual_species] += 1
                n_written += 1
                n_src += 1
            print(f"  [{species}/{kind}] {src_path.name}: {n_src} written, "
                  f"{sum(drop_counts.values())} dropped so far")
    return n_written, species_counts, drop_counts, seen_hashes


def parse_species_from_combined(path: Path):
    """Read a combined FASTA and tally species labels."""
    from Functions.RNAPhaseek.species_registry import species_id_for, label_for
    counts = Counter()
    for hdr, _ in parse_fasta(path):
        sp_id = species_id_for(hdr)
        counts[label_for(sp_id)] += 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--identity", type=float, default=0.90)
    p.add_argument("--skip-dedup", action="store_true",
                   help="Skip CD-HIT-EST (faster; for quick iteration)")
    args = p.parse_args()

    MS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build combined positives FIRST ───────────────────────────────────────
    # This also accumulates the canonical hash set we use to guard the
    # negatives against label leakage.
    print("\n=== Combining positives ===")
    pos_sources = [(HUMAN_POS, "human", "llps")]
    for fname, species in MS_POS_FILES:
        pos_sources.append((MS_DIR / fname, species, "llps"))
    raw_pos = MS_DIR / "_raw_combined_positives.fasta"
    n_pos, pos_counts, pos_drops, pos_hashes = write_combined(raw_pos, pos_sources)
    print(f"  total positives concatenated: {n_pos:,}")
    print(f"  positives drop counts: {dict(pos_drops)}")

    # ── Build combined negatives WITH leak guard ─────────────────────────────
    print("\n=== Combining negatives ===")
    neg_sources = [(HUMAN_NEG, "human", "neg")]
    for fname, species in MS_NEG_FILES:
        neg_sources.append((NEG_DIR / fname, species, "neg"))
    raw_neg = MS_DIR / "_raw_combined_negatives.fasta"
    n_neg, neg_counts, neg_drops, _ = write_combined(
        raw_neg, neg_sources, exclude_hashes=pos_hashes
    )
    print(f"  total negatives concatenated: {n_neg:,}")
    print(f"  negatives drop counts: {dict(neg_drops)}")

    # ── Dedup ────────────────────────────────────────────────────────────────
    dedup_requested = not args.skip_dedup
    pos_dedup_ok = False
    neg_dedup_ok = False
    if args.skip_dedup:
        print("\n=== Skipping dedup (--skip-dedup) ===")
        shutil.copyfile(raw_pos, OUT_POS)
        shutil.copyfile(raw_neg, OUT_NEG)
    else:
        print(f"\n=== Dedup positives at {args.identity:.0%} identity ===")
        pos_dedup_ok = run_cdhit(raw_pos, OUT_POS, args.identity)
        print(f"\n=== Dedup negatives at {args.identity:.0%} identity ===")
        neg_dedup_ok = run_cdhit(raw_neg, OUT_NEG, args.identity)

    # ── Verify species parseability post-dedup ───────────────────────────────
    print("\n=== Final species distribution (post-dedup) ===")
    final_pos = parse_species_from_combined(OUT_POS)
    final_neg = parse_species_from_combined(OUT_NEG)
    print(f"  {'Species':<14} {'Positives':>10} {'Negatives':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")
    for sp in sorted(set(final_pos.keys()) | set(final_neg.keys())):
        print(f"  {sp:<14} {final_pos.get(sp, 0):>10} {final_neg.get(sp, 0):>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<14} {sum(final_pos.values()):>10} {sum(final_neg.values()):>10}")

    # ── Stats with drop breakdowns + structured dedup_run ────────────────────
    pos_drop_dict = dict(pos_drops)
    neg_drop_dict = dict(neg_drops)
    quality_dropped_pos = sum(v for k, v in pos_drop_dict.items() if k.startswith("quality_"))
    quality_dropped_neg = sum(v for k, v in neg_drop_dict.items() if k.startswith("quality_"))

    stats = {
        "pre_dedup":   {"positives": dict(pos_counts), "negatives": dict(neg_counts),
                        "totals": {"positives": n_pos, "negatives": n_neg}},
        "post_dedup":  {"positives": dict(final_pos), "negatives": dict(final_neg),
                        "totals": {"positives": sum(final_pos.values()),
                                   "negatives": sum(final_neg.values())}},
        "identity":    args.identity,
        "dedup_run":   {
            "requested":     dedup_requested,
            "positives_ok":  pos_dedup_ok,
            "negatives_ok":  neg_dedup_ok,
        },
        "drop_counts": {
            "positives": pos_drop_dict,
            "negatives": neg_drop_dict,
        },
        "drop_summary": {
            "exact_duplicate_dropped": {
                "positives": pos_drop_dict.get("exact_duplicate", 0),
                "negatives": neg_drop_dict.get("exact_duplicate", 0),
            },
            "leak_dropped": {
                "negatives": neg_drop_dict.get("label_leak", 0),
            },
            "quality_dropped": {
                "positives": quality_dropped_pos,
                "negatives": quality_dropped_neg,
            },
        },
        "outputs":     {"positives": str(OUT_POS), "negatives": str(OUT_NEG)},
    }
    STATS_OUT.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {STATS_OUT}")

    # Cleanup
    for f in [raw_pos, raw_neg,
              OUT_POS.with_suffix(".fasta.clstr"),
              OUT_NEG.with_suffix(".fasta.clstr")]:
        if f.exists() and ".clstr" not in str(f):
            f.unlink()


if __name__ == "__main__":
    main()
