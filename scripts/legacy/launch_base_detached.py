"""
Detached launcher for the base RNAPhaseek model (Step 5B).

Same os.setsid() pattern as launch_hybrid_detached.py — survives harness
session reloads. Logs to model/rna_phaseek_train.log.

Run:
    python launch_base_detached.py
"""
import os
import sys

LOG = "model/rna_phaseek_train.log"
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
    "-m", "Functions.RNAPhaseek.RNAPhaseek_train",
])
