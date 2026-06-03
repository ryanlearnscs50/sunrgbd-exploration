#!/usr/bin/env bash
# v4 incremental harness: run the sequential class-incremental TR3D pipeline on the
# v4 3D-box split (configs/tr3d_incremental_v4, work_dirs/incremental_v4).
#   Usage: bash tools_incremental/run_incremental_v4.sh <3stage|6stage> [GPUS]
#
# Chains the per-stage configs from incremental_learning/gen_configs_v4.py:
# stage1 -> ... -> stageN, each warm-starting from the previous stage checkpoint
# (load_from baked into each config). After each stage it evaluates on that stage's
# CUMULATIVE val set.
#
# As written this is NAIVE sequential fine-tuning (Step C variant 1) — no forgetting
# mitigation. The two Step-C insertion points are marked "STEP C HOOK" below:
#   (1) classifier-head expansion / old-weight transfer before stage t>1
#   (2) offline pseudo-label prepass for the pseudo-labeling variant.
set -euo pipefail

ORDERING="${1:?usage: run_incremental_v4.sh <3stage|6stage> [GPUS]}"
GPUS="${2:-1}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CFG_DIR="$REPO/configs/tr3d_incremental_v4/$ORDERING"
[ -d "$CFG_DIR" ] || { echo "no configs at $CFG_DIR (run gen_configs_v4.py)"; exit 1; }

mapfile -t CFGS < <(ls "$CFG_DIR"/stage*.py | sort -V)
echo "[$ORDERING] ${#CFGS[@]} stages: ${CFGS[*]}"

for CFG in "${CFGS[@]}"; do
  STAGE="$(basename "$CFG" .py)"
  WORK_DIR="$REPO/work_dirs/incremental_v4/$ORDERING/$STAGE"
  echo "=================================================================="
  echo "[$ORDERING/$STAGE] config=$CFG  work_dir=$WORK_DIR  gpus=$GPUS"

  # ----- STEP C HOOK (1): classifier expansion + (2) pseudo-label prepass go here -----
  # if [ "$STAGE" != "stage1" ]; then
  #   python tools_incremental/expand_head.py --prev ... --cfg "$CFG"
  #   python tools_incremental/make_pseudo_labels.py --prev ... --cfg "$CFG"
  # fi
  # ------------------------------------------------------------------------------------

  if [ "$GPUS" -gt 1 ]; then
    bash "$REPO/tools/dist_train.sh" "$CFG" "$GPUS" --work-dir "$WORK_DIR"
  else
    python "$REPO/tools/train.py" "$CFG" --work-dir "$WORK_DIR"
  fi

  echo "[$ORDERING/$STAGE] eval on cumulative val:"
  python "$REPO/tools/test.py" "$CFG" "$WORK_DIR/latest.pth" --eval mAP 2>&1 | tee "$WORK_DIR/eval.log"
done
echo "[$ORDERING] v4 incremental run complete."
