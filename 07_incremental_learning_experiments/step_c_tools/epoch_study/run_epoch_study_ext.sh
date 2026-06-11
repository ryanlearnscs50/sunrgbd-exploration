#!/usr/bin/env bash
# Optimal-epoch study EXTENSION (2026-06-11): does PSEUDO keep improving past
# the paper's E=12, or does it plateau / overfit?  The original sweep only went
# up to E=12 and found pseudo monotonically improving — so we must look beyond.
#
# Runs the PSEUDO 3stage full chain at E in {16,24,36} (E<=12 already on disk),
# two GPUs in parallel, then analyzer + vs-epoch graph, then touches a DONE flag.
# Fully detached: needs no mid-run AI decision and survives session close.
#
#   nohup setsid bash tools_incremental/run_epoch_study_ext.sh \
#       > logs/epoch_study_ext_master.log 2>&1 < /dev/null &
set -uo pipefail   # NOT -e: one failed chain must not abort the rest
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/venv/bin/python"
SCHED=3stage
EPOCHS=(16 24 36)

echo "############################################################"
echo "# EPOCH STUDY EXTENSION (pseudo $SCHED, E>12)  $(date -Is)"
echo "# grid: ${EPOCHS[*]}  | GPUs 0,1"
echo "############################################################"

run_one () {  # <E> <GPU>
  local E="$1" GPU="$2"
  echo "[launch] pseudo $SCHED E=$E on GPU$GPU  $(date -Is)"
  bash "$REPO/tools_incremental/run_pseudo_v4_epochs.sh" "$SCHED" "$E" "$GPU" \
      > "$REPO/logs/pseudo_${SCHED}_ep${E}.log" 2>&1
  echo "[done ] pseudo $SCHED E=$E (exit $?)  $(date -Is)"
}

# Wave 1: E=16 -> GPU0, E=24 -> GPU1 (parallel)
run_one 16 0 &
P0=$!
run_one 24 1 &
P1=$!
wait $P0
wait $P1

# Wave 2: E=36 -> GPU0 (the longest chain; run alone)
run_one 36 0

echo ""
echo "==================== ANALYSIS ===================="
"$PY" "$REPO/tools_incremental/analyze_epoch_sweep.py" --sched "$SCHED" 2>&1 \
    | tee "$REPO/logs/EPOCH_STUDY_EXT_REPORT.txt"
echo ""
echo "==================== GRAPH ===================="
"$PY" "$REPO/tools_incremental/plot_epoch_sweep.py" --sched "$SCHED" \
    --epochs 2,4,6,8,12,16,24,36 \
    --out "$REPO/figures/epoch_sweep_${SCHED}.png" 2>&1 \
    | tee -a "$REPO/logs/EPOCH_STUDY_EXT_REPORT.txt"

touch "$REPO/logs/epoch_study_ext.DONE"
echo ""
echo "[ALL DONE] $(date -Is)  -> logs/epoch_study_ext.DONE"
