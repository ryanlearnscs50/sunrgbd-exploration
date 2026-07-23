#!/bin/bash
# Evaluate every released SUN RGB-D checkpoint (all stages of all 3 protocols)
# to produce the full forgetting curve. Splits 18 jobs across the 2 RTX 3090s.
#
# Usage: nohup bash run_full_sweep.sh > logs/sweep_master.log 2>&1 &
# Done marker: logs/SWEEP.DONE   Summary: logs/SWEEP_REPORT.txt

set -u
ROOT=/data3/ryan/ldmr_exploration
REPO=$ROOT/repo
PY=$REPO/venv/bin/python
LOGS=$ROOT/logs
mkdir -p "$LOGS"

C3=configs/incremental/sunrgbd/tr3d_dynamic_head_20x10x10_pseudo_memory_ld_design2_reviewing_521.py
C5=configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_pseudo_memory_ld_design2_reviewing_52211.py
C10=configs/incremental/sunrgbd/tr3d_dynamic_head_4x10_pseudo_memory_ld_design2_reviewing_6111111111.py

# job spec: "<protocol> <stage> <config>"
JOBS=()
for s in 01 02 03;                               do JOBS+=("3stage $s $C3");  done
for s in 01 02 03 04 05;                         do JOBS+=("5stage $s $C5");  done
for s in 01 02 03 04 05 06 07 08 09 10;          do JOBS+=("10stage $s $C10"); done

run_job() {
  local gpu=$1 proto=$2 stage=$3 cfg=$4
  local name="eval_${proto}_s${stage}"
  local log="$LOGS/${name}.log"
  local ckpt="$ROOT/checkpoints/sunrgbd_${proto}/stage_${stage}.pth"

  # skip if a previous run already produced results
  if [ -f "$log" ] && tr '\r' '\n' < "$log" | grep -q "Overall"; then
    echo "[gpu$gpu] SKIP $name (already has results)"
    return 0
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[gpu$gpu] MISSING $ckpt"
    return 1
  fi
  echo "[gpu$gpu] START $name $(date +%H:%M:%S)"
  cd "$REPO" || return 1
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 \
    "$PY" tools/eval_incremental.py "$cfg" "$ckpt" --eval mAP > "$log" 2>&1
  echo "[gpu$gpu] END   $name rc=$? $(date +%H:%M:%S)"
}

# deal jobs alternately to the two GPUs
worker() {
  local gpu=$1
  local i=$gpu
  while [ $i -lt ${#JOBS[@]} ]; do
    # shellcheck disable=SC2086
    run_job $gpu ${JOBS[$i]}
    i=$((i + 2))
  done
  echo "[gpu$gpu] worker finished"
}

echo "=== sweep start $(date) : ${#JOBS[@]} jobs over 2 GPUs ==="
worker 0 &
W0=$!
worker 1 &
W1=$!
wait $W0 $W1
echo "=== all evals done $(date) ==="

"$PY" "$ROOT/collect_results.py" > "$LOGS/SWEEP_REPORT.txt" 2>&1
echo "=== report written $(date) ==="
touch "$LOGS/SWEEP.DONE"
