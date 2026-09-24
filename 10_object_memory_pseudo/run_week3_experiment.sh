#!/usr/bin/env bash
# Started by launch_week3_experiments.py under nohup and a new session.
set -uo pipefail
ROOT=/data3/ryan/ldmr_exploration
MODE=${1:?expected experiment mode}
SEED=${2:-201}
[[ "$SEED" =~ ^[0-9]+$ ]] || exit 2
RUN_ID="$MODE"
if [[ "$SEED" != 201 ]]; then RUN_ID="${MODE}_s${SEED}"; fi
case "$MODE" in
  pseudo_cosine|pseudo_cosine_v2) CONFIG=tr3d_dynamic_head_8x5_pseudo_only_cosine_52211.py ;;
  dose25_cosine|dose25_cosine_v2) CONFIG=tr3d_dynamic_head_8x5_object_memory_pseudo_dose25_cosine_52211.py ;;
  pseudo_only) CONFIG=tr3d_dynamic_head_8x5_pseudo_only_matched_52211.py ;;
  object_pseudo) CONFIG=tr3d_dynamic_head_8x5_object_memory_random_pseudo_52211.py ;;
  dose25) CONFIG=tr3d_dynamic_head_8x5_object_memory_pseudo_dose25_52211.py ;;
  dose50) CONFIG=tr3d_dynamic_head_8x5_object_memory_pseudo_dose50_52211.py ;;
  *) exit 2 ;;
esac
STATE="$ROOT/week3_runs/$RUN_ID"
mkdir -p "$STATE"
exec 9>"$STATE/run.lock"
flock -n 9 || exit 73
if [[ -f "$STATE/exit_code" ]]; then
  echo 'This experiment already ended; preserve its artifacts and use a new run identity.'
  exit 73
fi
cd "$ROOT/repo" || exit 1
echo "$$" > "$STATE/supervisor.pid"
date --iso-8601=seconds > "$STATE/started_at"
venv/bin/python tools/train_incremental_scene.py \
  "configs/incremental/sunrgbd/$CONFIG" \
  --work-dir "$ROOT/incremental_logs/week3_${MODE}_s${SEED}" \
  --start-stage 2 --end-stage 5 \
  --checkpoint-path "$ROOT/checkpoints/sunrgbd_5stage/stage_01.pth" \
  --seed "$SEED" &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$STATE/trainer.pid"
wait "$TRAIN_PID"
RESULT=$?
echo "$RESULT" > "$STATE/exit_code.tmp"
mv "$STATE/exit_code.tmp" "$STATE/exit_code"
date --iso-8601=seconds > "$STATE/ended_at"
echo "WEEK3_EXPERIMENT_EXIT mode=$MODE code=$RESULT"
exit "$RESULT"
