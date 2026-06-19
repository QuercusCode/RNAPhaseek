"""Leakage-honest CD-HIT cluster grouping + is_yeast for the v13 pool (v5 + matched pairs).
Mirrors cluster_groups_v11 (purely additive; struct-neg parents map by sequence to v5 positives).
The new matched pairs (synthetic, non-yeast) cluster among themselves by their shared G4 cores.

  /opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python scripts/data_prep/cluster_groups_v13.py
"""
import os, sys, subprocess, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, os.getcwd())
from Functions.RNAPhaseek.RNAPhaseek_hybrid_data import read_fasta

CDHIT = "/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/cd-hit-est"
IDENT, WORD = 0.90, 8


def is_yeast(h):
    return "saccharomyces" in h.lower() or "|vantreeck" in h.lower()


def nrm(s):
    return "".join(c for c in s.upper().replace("T", "U") if c in "ACGU")


def main():
    pos = read_fasta("Data/raw/multispecies/strict_pool_v13_positives.fasta")
    neg = read_fasta("Data/raw/multispecies/strict_pool_v13_negatives_all.fasta")
    sneg = read_fasta("Data/raw/multispecies/strict_struct_negatives_v4.fasta")
    v5pos = read_fasta("Data/raw/multispecies/strict_pool_v5_positives.fasta")
    npos, nneg, nsn = len(pos), len(neg), len(sneg)
    print(f"v13 pool: {npos} pos / {nneg} real-neg / {nsn} struct-neg")

    d = Path(tempfile.mkdtemp(prefix="cdhit_v13_"))
    fa = d / "posneg.fasta"
    with open(fa, "w") as f:
        for i, (_, s) in enumerate(pos): f.write(f">{i}\n{s}\n")
        for j, (_, s) in enumerate(neg): f.write(f">{npos+j}\n{s}\n")
    out = d / "clustered"
    subprocess.run([CDHIT, "-i", str(fa), "-o", str(out), "-c", str(IDENT), "-n", str(WORD),
                    "-M", "0", "-T", "0", "-d", "0"], check=True, capture_output=True)
    cl = {}; cur = -1
    for line in open(str(out) + ".clstr"):
        if line.startswith(">Cluster"): cur = int(line.split()[1])
        else: cl[int(line.split(">")[1].split("...")[0])] = cur
    n_real = npos + nneg
    dropped = [i for i in range(n_real) if i not in cl]
    nc = (max(cl.values()) + 1) if cl else 0
    for i in dropped: cl[i] = nc; nc += 1
    if dropped: print(f"  {len(dropped)} short seq(s) not indexed by CD-HIT -> own clusters")

    seq2cl = {nrm(s): cl[i] for i, (_, s) in enumerate(pos)}
    next_cl = max(cl.values()) + 1
    groups = np.empty(n_real + nsn, dtype=int); yeast = np.zeros(n_real + nsn, dtype=bool)
    for i in range(n_real): groups[i] = cl[i]
    for i, (h, _) in enumerate(pos): yeast[i] = is_yeast(h)
    for j, (h, _) in enumerate(neg): yeast[npos + j] = is_yeast(h)
    orphan = 0
    for k, (h, _) in enumerate(sneg):
        p_old = int(h.split("parent=")[1].split("|")[0]) if "parent=" in h else -1
        parent_seq = nrm(v5pos[p_old][1]) if 0 <= p_old < len(v5pos) else None
        parent_hdr = v5pos[p_old][0] if 0 <= p_old < len(v5pos) else ""
        if parent_seq in seq2cl:
            groups[n_real + k] = seq2cl[parent_seq]
        else:
            groups[n_real + k] = next_cl; next_cl += 1; orphan += 1
        yeast[n_real + k] = is_yeast(parent_hdr)

    np.save("Data/splits/cluster_groups_v13.npy", groups)
    np.save("Data/splits/is_yeast_v13.npy", yeast)
    print(f"CD-HIT @{IDENT}: {n_real} real -> {len(set(groups.tolist()))} groups  ({orphan} orphaned struct-negs)")
    print(f"yeast pos: {int(yeast[:npos].sum())}/{npos}  | non-yeast pos: {npos-int(yeast[:npos].sum())}")
    print("Saved -> cluster_groups_v13.npy, is_yeast_v13.npy")


if __name__ == "__main__":
    main()
