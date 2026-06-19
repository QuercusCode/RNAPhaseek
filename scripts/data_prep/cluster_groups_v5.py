"""
Leakage-honest grouping for the v5 pool, controlling BOTH paralog leakage (CD-HIT
clusters) AND struct-neg->parent, PLUS an organism tag for the yeast-held-out diagnostic.

Pool order (must match run_v5_final.py): positives(1352) + real negatives(641) + struct negs(184).
Outputs:
  Data/splits/cluster_groups_v5.npy   (int group id per pool row)
  Data/splits/is_yeast_v5.npy         (bool per pool row; struct negs inherit parent)

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python cluster_groups_v5.py
"""
import os, sys, subprocess, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, os.getcwd())
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta

CDHIT = "/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/cd-hit-est"
IDENT, WORD = 0.90, 8


def is_yeast(hdr):
    return "saccharomyces" in hdr.lower() or "|vantreeck" in hdr.lower()


def main():
    pos = read_fasta("Data/raw/multispecies/strict_pool_v5_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v5_negatives_all.fasta")
    sneg = read_fasta("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
    npos, nneg, nsn = len(pos), len(neg), len(sneg)
    print(f"v5 pool: {npos} pos / {nneg} real-neg / {nsn} struct-neg")

    d = Path(tempfile.mkdtemp(prefix="cdhit_v5_"))
    fa = d / "posneg.fasta"
    with open(fa, "w") as f:
        for i, (_, s) in enumerate(pos): f.write(f">{i}\n{s}\n")
        for j, (_, s) in enumerate(neg): f.write(f">{npos+j}\n{s}\n")
    out = d / "clustered"
    subprocess.run([CDHIT, "-i", str(fa), "-o", str(out), "-c", str(IDENT),
                    "-n", str(WORD), "-M", "0", "-T", "0", "-d", "0"], check=True, capture_output=True)
    cl = {}; cur = -1
    for line in open(str(out) + ".clstr"):
        if line.startswith(">Cluster"): cur = int(line.split()[1])
        else: cl[int(line.split(">")[1].split("...")[0])] = cur
    n_real = npos + nneg
    assert len(cl) == n_real, f"{len(cl)} != {n_real}"

    groups = np.empty(n_real + nsn, dtype=int)
    for i in range(n_real): groups[i] = cl[i]
    yeast = np.zeros(n_real + nsn, dtype=bool)
    for i, (h, _) in enumerate(pos): yeast[i] = is_yeast(h)
    for j, (h, _) in enumerate(neg): yeast[npos + j] = is_yeast(h)
    for k, (h, _) in enumerate(sneg):
        p = int(h.split("parent=")[1].split("|")[0])
        groups[n_real + k] = cl[p]            # struct neg -> parent cluster
        yeast[n_real + k] = yeast[p]          # inherit parent organism

    np.save("Data/splits/cluster_groups_v5.npy", groups)
    np.save("Data/splits/is_yeast_v5.npy", yeast)
    n_clusters = len(set(groups.tolist()))
    pos_clusters = len(set(cl[i] for i in range(npos)))
    cross = sum(1 for c in set(cl.values())
                if any(i < npos for i in cl if cl[i] == c) and any(i >= npos for i in cl if cl[i] == c))
    print(f"CD-HIT @{IDENT}: {npos} pos + {nneg} neg -> {n_clusters} groups")
    print(f"  positives -> {pos_clusters} pos-bearing clusters")
    print(f"  yeast positives: {int(yeast[:npos].sum())}/{npos}  | non-yeast positives: {npos-int(yeast[:npos].sum())}")
    print("Saved -> cluster_groups_v5.npy, is_yeast_v5.npy")


if __name__ == "__main__":
    main()
