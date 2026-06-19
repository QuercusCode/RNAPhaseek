"""
Detached launcher for the full-sequence hybrid training.

Same os.setsid() pattern as launch_hybrid_detached.py — survives harness
session reloads.

Run:
    python launch_hybrid_fullseq_detached.py
"""
import os
import sys

LOG = "model/hybrid_fullseq_train.log"
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

os.execv(PY, [
    PY, "-u",
    "-m", "Functions.RNAPhaseek.RNAPhaseek_hybrid_fullseq_train",
    # Initialize from Phase 1 -- the model already scores 1022-nt windows well
    # at AUROC 0.77; attention-pool just extends it to multi-window.
    "--init_from", "model/phase1/hybrid_best.pt",
    # Lower LR since we're fine-tuning a pre-trained model, not from scratch.
    "--lr", "5e-5",
])
