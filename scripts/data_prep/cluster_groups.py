"""
Build a leakage-honest GROUP key for the full v4 pool that controls BOTH:
  (a) positive (and negative) near-duplicate paralogs/fragments — via CD-HIT clusters
  (b) structural-negative -> parent positive — struct neg inherits its parent's cluster

Pool order (must match run_v4_*.py): positives(427) + real negatives(636) + struct negs(184).
Output: Data/splits/cluster_groups_v4.npy  (int group id per pool row) + a summary print.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python cluster_groups.py
"""
import os, sys, subprocess, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, os.getcwd())
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta

CDHIT = "/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/cd-hit-est"
IDENT, WORD = 0.90, 8


def main():
    pos = read_fasta("Data/raw/multispecies/strict_pool_v3_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v3_negatives_all.fasta")
    sneg = read_fasta("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
    npos, nneg, nsn = len(pos), len(neg), len(sneg)
    print(f"pool: {npos} pos / {nneg} real-neg / {nsn} struct-neg")

    # cluster positives + real negatives together (cross-class near-dups also grouped)
    d = Path(tempfile.mkdtemp(prefix="cdhit_grp_"))
    fa = d / "posneg.fasta"
    with open(fa, "w") as f:
        for i, (_, s) in enumerate(pos):
            f.write(f">{i}\n{s}\n")
        for j, (_, s) in enumerate(neg):
            f.write(f">{npos + j}\n{s}\n")
    out = d / "clustered"
    subprocess.run([CDHIT, "-i", str(fa), "-o", str(out), "-c", str(IDENT),
                    "-n", str(WORD), "-M", "0", "-T", "0", "-d", "0"],
                   check=True, capture_output=True)

    # parse .clstr -> cluster id per pool index (0..npos+nneg-1)
    cl = {}
    cur = -1
    for line in open(str(out) + ".clstr"):
        if line.startswith(">Cluster"):
            cur = int(line.split()[1])
        else:
            idx = int(line.split(">")[1].split("...")[0].split("|")[0])
            cl[idx] = cur
    n_real = npos + nneg
    assert len(cl) == n_real, f"clstr parse {len(cl)} != {n_real}"
    n_clusters = max(cl.values()) + 1

    # cross-class clusters (a cluster containing BOTH a positive and a negative) — flag
    members = {}
    for idx, c in cl.items():
        members.setdefault(c, []).append(idx)
    cross = [c for c, m in members.items()
             if any(i < npos for i in m) and any(i >= npos for i in m)]

    # struct negatives -> parent positive's cluster
    groups = np.empty(npos + nneg + nsn, dtype=int)
    for i in range(n_real):
        groups[i] = cl[i]
    n_bad = 0
    for k, (h, _) in enumerate(sneg):
        pidx = int(h.split("parent=")[1].split("|")[0])
        groups[n_real + k] = cl[pidx]   # parent positive's cluster
        if not (0 <= pidx < npos):
            n_bad += 1

    np.save("Data/splits/cluster_groups_v4.npy", groups)
    print(f"\nCD-HIT @{IDENT}: {npos} pos + {nneg} neg -> {n_clusters} clusters")
    print(f"  positives collapse: 427 -> {len(set(cl[i] for i in range(npos)))} pos-bearing clusters")
    print(f"  total groups over full pool: {len(set(groups.tolist()))}")
    print(f"  cross-class clusters (pos+neg same cluster, label-conflict flag): {len(cross)}")
    print(f"  struct-neg parent lookups bad: {n_bad}")
    # quick paralog example
    big = sorted(members.items(), key=lambda kv: -len(kv[1]))[:3]
    for c, m in big:
        tag = "pos" if m[0] < npos else "neg"
        print(f"  largest cluster {c}: {len(m)} members (e.g. idx {m[:5]})")
    print("\nSaved -> Data/splits/cluster_groups_v4.npy")


if __name__ == "__main__":
    main()
