#!/usr/bin/env bash
# Epoch-count trace (2026-06-10 task): re-train one incremental fine-tuning stage
# keeping ALL epoch checkpoints, then evaluate every epoch on the CUMULATIVE val
# set, so we can plot old-class vs new-class mAP per epoch and read off the
# optimal stopping epoch (hypothesis: old-class mAP peaks early then decays as
# the model over-fits the novel-only data -> fewer than the paper's 12 epochs
# may retain old classes better).
#
#   Usage: bash tools_incremental/run_epoch_trace.sh <3stage|6stage> <stage_idx> [GPU_ID]
#
# Faithful to the real naive run: warm-starts from the SAME init_expanded.pth the
# naive run used (so the per-epoch curve is the curve the real run followed).
set -euo pipefail

SCHED="${1:?usage: run_epoch_trace.sh <3stage|6stage> <stage_idx> [GPU]}"
STG="${2:?stage index, e.g. 2}"
GPU_ID="${3:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/venv/bin/python"

BASE_CFG="$REPO/configs/tr3d_incremental_v4/$SCHED/stage${STG}.py"
INIT="$REPO/work_dirs/incremental_v4_naive/$SCHED/stage${STG}/init_expanded.pth"
[ -f "$BASE_CFG" ] || { echo "no base cfg $BASE_CFG"; exit 1; }
[ -f "$INIT" ]     || { echo "no warm-start $INIT"; exit 1; }

WORK_DIR="$REPO/work_dirs/incremental_v4_epochtrace/$SCHED/stage${STG}"
TRACE_CFG="$REPO/configs/tr3d_incremental_v4/$SCHED/trace_stage${STG}.py"
mkdir -p "$WORK_DIR"

# Trace config = base stage config + overrides (later assignments win in a
# top-to-bottom python config): warm-start from the expanded init, KEEP ALL 12
# epoch checkpoints, point work_dir at the trace dir.
cp "$BASE_CFG" "$TRACE_CFG"
cat >> "$TRACE_CFG" <<EOF

# ---- epoch-trace overrides (auto-appended by run_epoch_trace.sh) ----
load_from = '$INIT'
checkpoint_config = dict(interval=1, max_keep_ckpts=12)
work_dir = '$WORK_DIR'
EOF

export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "=================================================================="
echo "[trace/$SCHED/stage$STG] GPU=$GPU_ID  warm-start=$INIT"
echo "[trace/$SCHED/stage$STG] started: $(date -Is)"
echo "=================================================================="

# Train once, keeping every epoch checkpoint (skip if already trained).
if [ ! -e "$WORK_DIR/epoch_12.pth" ]; then
  "$PY" "$REPO/tools/train.py" "$TRACE_CFG" --work-dir "$WORK_DIR" --seed 0 --deterministic
else
  echo "[skip-train] epoch_12.pth already present"
fi

# Evaluate every epoch checkpoint on the cumulative val set.
for E in $(seq 1 12); do
  CKPT="$WORK_DIR/epoch_${E}.pth"
  LOG="$WORK_DIR/eval_epoch_${E}.log"
  [ -e "$CKPT" ] || { echo "[warn] missing $CKPT"; continue; }
  if [ -e "$LOG" ] && grep -q 'Overall' "$LOG"; then echo "[skip-eval] epoch $E"; continue; fi
  echo "######## [trace/$SCHED/stage$STG] eval epoch $E  $(date -Is) ########"
  "$PY" "$REPO/tools/test.py" "$TRACE_CFG" "$CKPT" --eval mAP 2>&1 | tee "$LOG"
done

echo ""
echo "[trace/$SCHED/stage$STG] COMPLETE: $(date -Is)"
