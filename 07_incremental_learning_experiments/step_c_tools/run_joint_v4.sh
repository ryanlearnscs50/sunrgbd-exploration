#!/usr/bin/env bash
# Fully-supervised (joint) upper bound for the v4 incremental setup.
# Trains ONE TR3D from scratch on the joint full-label train pkl, then evaluates
# on the same final cumulative val pkl the incremental chains use.
#
#   Usage: bash tools_incremental/run_joint_v4.sh <3stage|6stage> [GPU_ID]
set -euo pipefail
ORDERING="${1:?usage: run_joint_v4.sh <3stage|6stage> [GPU_ID]}"
GPU_ID="${2:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/venv/bin/python"
CFG="$REPO/configs/tr3d_incremental_v4/$ORDERING/joint.py"
WORK_DIR="$REPO/work_dirs/incremental_v4_joint/$ORDERING"
[ -f "$CFG" ] || { echo "no config at $CFG"; exit 1; }
mkdir -p "$WORK_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "=================================================================="
echo "[joint/$ORDERING] GPU=$GPU_ID | cfg=$CFG | out=$WORK_DIR"
echo "[joint/$ORDERING] started: $(date -Is)"
echo "=================================================================="

if [ -e "$WORK_DIR/latest.pth" ] && [ -e "$WORK_DIR/eval.log" ]; then
  echo "[skip] already complete (latest.pth + eval.log present)"
  exit 0
fi

echo "[train] $PY tools/train.py $CFG --work-dir $WORK_DIR --seed 0 --deterministic"
"$PY" "$REPO/tools/train.py" "$CFG" --work-dir "$WORK_DIR" --seed 0 --deterministic

[ -e "$WORK_DIR/latest.pth" ] || { echo "ERROR: no checkpoint at $WORK_DIR/latest.pth"; exit 2; }

echo "[eval] cumulative val -> $WORK_DIR/eval.log + results.pkl"
"$PY" "$REPO/tools/test.py" "$CFG" "$WORK_DIR/latest.pth" \
    --out "$WORK_DIR/results.pkl" --eval mAP 2>&1 | tee "$WORK_DIR/eval.log"

echo "[joint/$ORDERING] COMPLETE: $(date -Is)"
