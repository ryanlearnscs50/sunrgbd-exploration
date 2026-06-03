#!/usr/bin/env bash
# Step C — VARIANT 1: naive sequential fine-tuning baseline (v4 3D-box split).
#
#   Usage: bash tools_incremental/run_naive_v4.sh <3stage|6stage> [GPU_ID]
#
# For each stage 1..N:
#   * stage 1      : train from scratch (n_classes = stage-1 cumulative count).
#   * stage t > 1  : EXPAND the previous stage's classifier head to the new
#                    cumulative class count (tools_incremental/expand_head.py),
#                    warm-start from that expanded checkpoint, and fine-tune on
#                    this stage's NOVEL-class-only GT. No forgetting mitigation
#                    (that is what the pseudo-label variant adds) -> we expect
#                    old-class forgetting; this run is the baseline to beat.
#   * after every stage: evaluate on the CUMULATIVE val set and save results.
#
# Everything runs through $REPO/venv/bin/python (the /data3 noexec-safe venv).
# Designed to be launched detached (setsid/nohup) so it survives the session.
set -euo pipefail

ORDERING="${1:?usage: run_naive_v4.sh <3stage|6stage> [GPU_ID]}"
GPU_ID="${2:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/venv/bin/python"
CFG_DIR="$REPO/configs/tr3d_incremental_v4/$ORDERING"
OUT_ROOT="$REPO/work_dirs/incremental_v4_naive/$ORDERING"
[ -d "$CFG_DIR" ] || { echo "no configs at $CFG_DIR"; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU_ID"
mapfile -t CFGS < <(ls "$CFG_DIR"/stage*.py | sort -V)
echo "=================================================================="
echo "[naive/$ORDERING] ${#CFGS[@]} stages | GPU=$GPU_ID | out=$OUT_ROOT"
echo "[naive/$ORDERING] started: $(date -Is)"
echo "=================================================================="

PREV_CKPT=""
for CFG in "${CFGS[@]}"; do
  STAGE="$(basename "$CFG" .py)"
  WORK_DIR="$OUT_ROOT/$STAGE"
  mkdir -p "$WORK_DIR"
  echo ""
  echo "######## [naive/$ORDERING/$STAGE] $(date -Is) ########"

  # Idempotent resume: a stage with both a checkpoint and a completed eval is
  # treated as done; we just carry its checkpoint forward and move on. Lets a
  # crashed run (e.g. stage5 here) be re-launched without retraining 1..4.
  if [ -e "$WORK_DIR/latest.pth" ] && [ -e "$WORK_DIR/eval.log" ]; then
    echo "[skip] $STAGE already complete (latest.pth + eval.log present)"
    PREV_CKPT="$WORK_DIR/latest.pth"
    continue
  fi

  TRAIN_ARGS=("$CFG" --work-dir "$WORK_DIR" --seed 0 --deterministic)

  if [ "$STAGE" != "stage1" ]; then
    # --- head expansion + warm-start (the weight transfer the baseline needs) ---
    INIT="$WORK_DIR/init_expanded.pth"
    echo "[expand] $PREV_CKPT  ->  $INIT  (cfg=$CFG)"
    "$PY" "$REPO/tools_incremental/expand_head.py" --prev "$PREV_CKPT" --cfg "$CFG" --out "$INIT"
    TRAIN_ARGS+=(--cfg-options "load_from=$INIT")
  fi

  echo "[train] $PY tools/train.py ${TRAIN_ARGS[*]}"
  "$PY" "$REPO/tools/train.py" "${TRAIN_ARGS[@]}"

  PREV_CKPT="$WORK_DIR/latest.pth"
  [ -e "$PREV_CKPT" ] || { echo "ERROR: no checkpoint produced at $PREV_CKPT"; exit 2; }

  echo "[eval] cumulative val -> $WORK_DIR/eval.log + results.pkl"
  "$PY" "$REPO/tools/test.py" "$CFG" "$PREV_CKPT" \
      --out "$WORK_DIR/results.pkl" --eval mAP 2>&1 | tee "$WORK_DIR/eval.log"
done

echo ""
echo "[naive/$ORDERING] COMPLETE: $(date -Is)"
