"""
Detached Phase 2 launcher for RNAPhaseek hybrid training.

Phase 2 = unfreeze the last N RNA-FM layers (default 2) and fine-tune them
together with the adapter + head. Backbone params use a much lower LR
(backbone_lr=5e-6) than the adapter (lr=2e-4) so we don't blow up RNA-FM's
pretrained weights.

The Phase 2 run:
  - Loads Phase 1's best weights into the model via --init_from
  - Starts with a FRESH optimizer (Phase 2 has more trainable params)
  - Restarts the LR schedule (warmup from 0 again)
  - Saves to model/hybrid_best.pt (Phase 1 is safe in model/phase1/)

Run:
    python launch_hybrid_phase2.py
"""
import os
import sys

LOG = "model/hybrid_train.log"
PY  = "/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python"
INIT_FROM = "model/phase1/hybrid_best.pt"

if not os.path.exists(INIT_FROM):
    print(f"ERROR: {INIT_FROM} does not exist. Phase 1 weights missing.", flush=True)
    sys.exit(1)

# fork(): parent reports child PID and exits; child becomes detached session leader
pid = os.fork()
if pid != 0:
    print(f"detached child PID: {pid}", flush=True)
    sys.exit(0)

os.setsid()  # decouple from harness session

# Truncate log and redirect all fds to it
os.makedirs("model", exist_ok=True)
log_fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log_fd, 0)
os.dup2(log_fd, 1)
os.dup2(log_fd, 2)
os.close(log_fd)

os.execv(PY, [
    PY, "-u",
    "-m", "Functions.RNAPhaseek.RNAPhaseek_hybrid_train",
    "--unfreeze_last_n", "2",
    "--init_from",       INIT_FROM,
])
