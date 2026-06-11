#!/usr/bin/env bash
# Optimal-epoch study (2026-06-10 task) — NAIVE variant, FULL-CHAIN sweep.
#
#   Usage: bash tools_incremental/run_naive_v4_epochs.sh <3stage|6stage> <EPOCHS> [GPU_ID]
#
# Runs the ENTIRE sequential chain at a single epoch budget E (stage1@E ->
# expand -> stage2@E -> expand -> stage3@E ...), because the epoch count is a
# GLOBAL knob: stage t warm-starts from stage t-1 trained at the SAME E, so you
# cannot pick a stage's epochs independently. Different E values ARE independent
# -> run them on separate GPUs in parallel. E=12 already exists as the original
# naive run (work_dirs/incremental_v4_naive); only sweep {2,4,6,8} fresh.
#
# lr milestones are rescaled per E (step=[8,11] is tuned for 12 ep -> would never
# fire for E<8): m1=floor(2E/3), m2=floor(11E/12), deduped/dropped if >=E.
set -euo pipefail

SCHED="${1:?usage: run_naive_v4_epochs.sh <3stage|6stage> <EPOCHS> [GPU]}"
E="${2:?epoch budget, e.g. 4}"
GPU_ID="${3:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/venv/bin/python"
CFG_DIR="$REPO/configs/tr3d_incremental_v4/$SCHED"
OUT_ROOT="$REPO/work_dirs/incremental_v4_naive_ep${E}/$SCHED"
[ -d "$CFG_DIR" ] || { echo "no configs at $CFG_DIR"; exit 1; }

# rescaled lr milestones
M1=$(( 2*E/3 )); M2=$(( 11*E/12 ))
[ "$M1" -lt 1 ] && M1=1
[ "$M2" -ge "$E" ] && M2=$(( E-1 ))
[ "$M1" -ge "$E" ] && M1=$(( E-1 ))
if [ "$M1" -ge "$M2" ]; then STEP="[$M1]"; else STEP="[$M1, $M2]"; fi
[ "$M1" -lt 1 ] && STEP="[1]"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
mapfile -t CFGS < <(ls "$CFG_DIR"/stage*.py | sort -V)
echo "=================================================================="
echo "[naive-ep$E/$SCHED] ${#CFGS[@]} stages | E=$E | lr step=$STEP | GPU=$GPU_ID"
echo "[naive-ep$E/$SCHED] out=$OUT_ROOT | started: $(date -Is)"
echo "=================================================================="

PREV_CKPT=""
for CFG in "${CFGS[@]}"; do
  STAGE="$(basename "$CFG" .py)"
  WORK_DIR="$OUT_ROOT/$STAGE"
  mkdir -p "$WORK_DIR"
  echo ""
  echo "######## [naive-ep$E/$SCHED/$STAGE] $(date -Is) ########"

  if [ -e "$WORK_DIR/latest.pth" ] && [ -e "$WORK_DIR/eval.log" ]; then
    echo "[skip] $STAGE already complete"; PREV_CKPT="$WORK_DIR/latest.pth"; continue
  fi

  # Epoch-overridden config: base stage cfg + appended overrides (last wins).
  ECFG="$CFG_DIR/ep${E}_$STAGE.py"
  cp "$CFG" "$ECFG"
  cat >> "$ECFG" <<EOF

# ---- epoch-sweep overrides (auto-appended by run_naive_v4_epochs.sh) ----
runner = dict(type='EpochBasedRunner', max_epochs=$E)
lr_config = dict(policy='step', warmup=None, step=$STEP)
work_dir = '$WORK_DIR'
EOF

  TRAIN_ARGS=("$ECFG" --work-dir "$WORK_DIR" --seed 0 --deterministic)
  if [ "$STAGE" != "stage1" ]; then
    INIT="$WORK_DIR/init_expanded.pth"
    echo "[expand] $PREV_CKPT -> $INIT"
    "$PY" "$REPO/tools_incremental/expand_head.py" --prev "$PREV_CKPT" --cfg "$ECFG" --out "$INIT"
    TRAIN_ARGS+=(--cfg-options "load_from=$INIT")
  fi

  echo "[train E=$E] $PY tools/train.py ${TRAIN_ARGS[*]}"
  "$PY" "$REPO/tools/train.py" "${TRAIN_ARGS[@]}"
  PREV_CKPT="$WORK_DIR/latest.pth"
  [ -e "$PREV_CKPT" ] || { echo "ERROR: no checkpoint at $PREV_CKPT"; exit 2; }

  echo "[eval] cumulative val -> $WORK_DIR/eval.log"
  "$PY" "$REPO/tools/test.py" "$ECFG" "$PREV_CKPT" \
      --out "$WORK_DIR/results.pkl" --eval mAP 2>&1 | tee "$WORK_DIR/eval.log"
done

echo ""
echo "[naive-ep$E/$SCHED] COMPLETE: $(date -Is)"
