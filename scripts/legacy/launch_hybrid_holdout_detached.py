"""
Detached launcher for Phase 1 retrain on the 85% train+val subset.

The held-out 15% test set is preserved separately in Data/raw/positives_heldouttest.fasta
and Data/raw/negatives_heldouttest.fasta -- evaluated ONCE after training to get
publishable test numbers.

Outputs (separate from the original Phase 1, which stays intact in model/phase1/):
  model/hybrid_holdout_best.pt
  model/hybrid_holdout_final.pt
  model/hybrid_holdout_best_thresh.npy
  model/hybrid_holdout_train.log

Run:
    python launch_hybrid_holdout_detached.py
"""
import os
import sys

LOG = "model/hybrid_holdout_train.log"
PY  = "/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python"

pid = os.fork()
if pid != 0:
    print(f"detached child PID: {pid}", flush=True)
    sys.exit(0)

os.setsid()

os.makedirs("model", exist_ok=True)
log_fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log_fd, 0)
os.dup2(log_fd, 1)
os.dup2(log_fd, 2)
os.close(log_fd)

# Same hyperparameters as Phase 1 (which hit 0.7707 val AUROC), but pointed at
# the trainval subset only. The trainer's internal 85/15 split applies to the
# trainval -> effective 72.25% train / 12.75% val / 15% truly held out.
os.execv(PY, [
    PY, "-u",
    "-m", "Functions.RNAPhaseek.RNAPhaseek_hybrid_train",
    "--fasta_pos",  "Data/raw/positives_trainval.fasta",
    "--fasta_neg",  "Data/raw/negatives_trainval.fasta",
    "--src_pos",    "Data/processed/fegs_topk_trainval_pos",
    "--src_neg",    "Data/processed/fegs_topk_trainval_neg",
    "--bio_pos",    "Data/splits/biophys_trainval_pos.npy",
    "--bio_neg",    "Data/splits/biophys_trainval_neg.npy",
    "--best_ckpt",  "model/hybrid_holdout_best.pt",
    "--final_ckpt", "model/hybrid_holdout_final.pt",
])
