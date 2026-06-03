#!/usr/bin/env bash
# Step C - VARIANT 3 (optional): PSEUDO-LABEL + static-teacher DISTILLATION
# incremental fine-tuning (v4 3D-box split). Closest port of full SDCoT.
#
#   Usage: bash tools_incremental/run_distill_v4.sh <3stage|6stage> [GPU_ID]
#   Env:   SCORE_THR (default .5)   DISTILL_W (default 1.0, SDCoT w_distill=1)
#
# Identical pipeline to run_pseudo_v4.sh (expand head -> make pseudo old-GT ->
# train on novel+pseudo GT -> eval cumulative), PLUS each stage t>1 trains the
# MinkSingleStage3DDetectorInc detector with a frozen stage_{t-1} teacher that
# adds an old-class logit-distillation loss (mink_single_stage_inc.py). The
# teacher is hidden from the student state_dict, so the saved checkpoint is the
# plain-detector format used by eval (tools/test.py with the stock config) and
# by the next stage's expand_head.
# Out root: work_dirs/incremental_v4_distill/<sched>/stage{t}/. Idempotent.
set -euo pipefail

ORDERING="${1:?usage: run_distill_v4.sh <3stage|6stage> [GPU_ID]}"
GPU_ID="${2:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/venv/bin/python"
CFG_DIR="$REPO/configs/tr3d_incremental_v4/$ORDERING"
DATA_DIR="$REPO/data/sunrgbd_incremental/v4_3dbox/$ORDERING"
CARVE="$DATA_DIR/carve_report_v4.json"
OUT_ROOT="$REPO/work_dirs/incremental_v4_distill/$ORDERING"
SCORE_THR="${SCORE_THR:-0.5}"
DISTILL_W="${DISTILL_W:-1.0}"
[ -d "$CFG_DIR" ] || { echo "no configs at $CFG_DIR"; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU_ID"
mapfile -t CFGS < <(ls "$CFG_DIR"/stage*.py | sort -V)
echo "=================================================================="
echo "[distill/$ORDERING] ${#CFGS[@]} stages | GPU=$GPU_ID | score_thr=$SCORE_THR | w_distill=$DISTILL_W"
echo "[distill/$ORDERING] out=$OUT_ROOT | started: $(date -Is)"
echo "=================================================================="

# Seed stage1 from the naive baseline (stage1 has no old classes -> identical).
NAIVE_S1="$REPO/work_dirs/incremental_v4_naive/$ORDERING/stage1"
DISTILL_S1="$OUT_ROOT/stage1"
if [ -e "$NAIVE_S1/latest.pth" ] && [ ! -e "$DISTILL_S1/latest.pth" ]; then
  mkdir -p "$DISTILL_S1"
  cp "$NAIVE_S1/latest.pth" "$DISTILL_S1/latest.pth"
  [ -e "$NAIVE_S1/eval.log" ] && cp "$NAIVE_S1/eval.log" "$DISTILL_S1/eval.log"
  echo "[seed] reused naive stage1 -> $DISTILL_S1 (identical base, no old classes at stage1)"
fi

# helper: read head.n_classes from a stage config via mmcv (== cumulative classes)
read_n_classes() {
  "$PY" - "$1" <<'PYEOF'
import sys
from mmcv import Config
print(Config.fromfile(sys.argv[1]).model['head']['n_classes'])
PYEOF
}

PREV_CKPT=""
PREV_CFG=""
for CFG in "${CFGS[@]}"; do
  STAGE="$(basename "$CFG" .py)"
  WORK_DIR="$OUT_ROOT/$STAGE"
  mkdir -p "$WORK_DIR"
  echo ""
  echo "######## [distill/$ORDERING/$STAGE] $(date -Is) ########"

  if [ -e "$WORK_DIR/latest.pth" ] && [ -e "$WORK_DIR/eval.log" ]; then
    echo "[skip] $STAGE already complete (latest.pth + eval.log present)"
    PREV_CKPT="$WORK_DIR/latest.pth"; PREV_CFG="$CFG"
    continue
  fi

  TRAIN_ARGS=("$CFG" --work-dir "$WORK_DIR" --seed 0 --deterministic)

  if [ "$STAGE" != "stage1" ]; then
    # (a) grow the classifier head to the new cumulative class count
    INIT="$WORK_DIR/init_expanded.pth"
    echo "[expand] $PREV_CKPT  ->  $INIT  (cfg=$CFG)"
    "$PY" "$REPO/tools_incremental/expand_head.py" --prev "$PREV_CKPT" --cfg "$CFG" --out "$INIT"

    # (b) generate old-class pseudo-GT from the stage_{t-1} model over TRAIN scenes
    TRAIN_PKL="$DATA_DIR/sunrgbd_infos_train_${STAGE}.pkl"
    PSEUDO_PKL="$WORK_DIR/sunrgbd_infos_train_${STAGE}_pseudo.pkl"
    echo "[pseudo] $PREV_CKPT (cfg=$PREV_CFG) over $TRAIN_PKL -> $PSEUDO_PKL  (score_thr=$SCORE_THR)"
    "$PY" "$REPO/tools_incremental/make_pseudo_labels.py" \
        --prev-cfg "$PREV_CFG" --prev-ckpt "$PREV_CKPT" \
        --train-pkl "$TRAIN_PKL" --carve-report "$CARVE" \
        --out-pkl "$PSEUDO_PKL" --raw-dets "$WORK_DIR/raw_dets.pkl" \
        --score-thr "$SCORE_THR" --gpu-id 0

    # (c) k_old = previous stage's cumulative class count (= teacher head channels)
    K_OLD="$(read_n_classes "$PREV_CFG")"
    echo "[distill] teacher=$PREV_CKPT  n_old_classes=$K_OLD  w_distill=$DISTILL_W"

    # (d) train on novel+pseudo GT, warm-started, with the frozen teacher distilling old logits
    TRAIN_ARGS+=(--cfg-options
        "load_from=$INIT"
        "data.train.dataset.ann_file=$PSEUDO_PKL"
        "model.type=MinkSingleStage3DDetectorInc"
        "model.teacher_ckpt=$PREV_CKPT"
        "model.n_old_classes=$K_OLD"
        "model.loss_distill_weight=$DISTILL_W")
  fi

  echo "[train] $PY tools/train.py ${TRAIN_ARGS[*]}"
  "$PY" "$REPO/tools/train.py" "${TRAIN_ARGS[@]}"

  PREV_CKPT="$WORK_DIR/latest.pth"; PREV_CFG="$CFG"
  [ -e "$PREV_CKPT" ] || { echo "ERROR: no checkpoint produced at $PREV_CKPT"; exit 2; }

  # (e) evaluate on the CUMULATIVE val set with the STOCK config (plain detector)
  echo "[eval] cumulative val -> $WORK_DIR/eval.log + results.pkl"
  "$PY" "$REPO/tools/test.py" "$CFG" "$PREV_CKPT" \
      --out "$WORK_DIR/results.pkl" --eval mAP 2>&1 | tee "$WORK_DIR/eval.log"
done

echo ""
echo "[distill/$ORDERING] COMPLETE: $(date -Is)"
