"""
Detached launcher for RNAPhaseek hybrid training.

Forks into a new session (os.setsid) so the training process survives
shell/session lifecycle events that would kill an attached child process.

Run:
    python launch_hybrid_detached.py
"""
import os
import sys

LOG = "model/hybrid_train.log"
PY  = "/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python"

# fork() once — parent prints PID and exits; child continues.
pid = os.fork()
if pid != 0:
    # Parent: report child PID then exit so caller's shell returns immediately
    print(f"detached child PID: {pid}", flush=True)
    sys.exit(0)

# Child: detach from controlling terminal + session group
os.setsid()

# Redirect stdin/stdout/stderr to log file
os.makedirs("model", exist_ok=True)
log_fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log_fd, 0)   # stdin from log (effectively unused)
os.dup2(log_fd, 1)   # stdout to log
os.dup2(log_fd, 2)   # stderr to log
os.close(log_fd)

# Now exec the training as PID 1 of its own session
os.execv(PY, [
    PY, "-u",
    "-m", "Functions.RNAPhaseek.RNAPhaseek_hybrid_train",
])
