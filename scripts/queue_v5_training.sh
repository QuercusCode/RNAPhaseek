#!/usr/bin/env bash
# Auto-start v5 training once (a) features are regenerated and (b) the cluster-CV frees the GPU.
# Sequence: wait feature-regen -> build groups -> wait GPU -> train v5.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/opt/homebrew/Caskroom/mambaforge/base/envs/rnaphaseek/bin/python
FLOG=model/strict_eval_v4/precompute_v5.log
mkdir -p model/strict_eval_v5
exec >> model/strict_eval_v5/queue_driver.log 2>&1
echo "=== [driver] start $(date) ==="

# 1) wait for feature regeneration (CPU, already running concurrently)
echo "[driver] waiting for v5 feature regen..."
while true; do
  grep -q "DONE precompute_v5_features" "$FLOG" 2>/dev/null && { echo "[driver] features DONE"; break; }
  if ! pgrep -f precompute_v5_features >/dev/null; then
    echo "[driver] FATAL: feature regen process gone with no DONE marker — aborting (features incomplete)"; exit 1
  fi
  sleep 30
done

# 2) build cluster + organism grouping (fast, CPU)
echo "[driver] building v5 grouping..."
$PY cluster_groups_v5.py || { echo "[driver] FATAL: grouping failed"; exit 1; }

# 3) wait for the cluster-CV to release the GPU (no two MPS jobs at once)
echo "[driver] waiting for cluster-CV to finish (free GPU)..."
while pgrep -f run_v4_clustercv >/dev/null; do sleep 60; done
echo "[driver] GPU free. cluster-CV summary: $(ls model/strict_eval_v4_clustercv/eval_summary.json 2>/dev/null || echo 'NONE (may have died; resumable from checkpoint)')"

# 4) train v5 (5-fold cluster-grouped CV + final + locked test, checkpointed)
echo "[driver] === starting v5 training $(date) ==="
$PY run_v5_final.py
echo "=== [driver] v5 training finished $(date) ==="
