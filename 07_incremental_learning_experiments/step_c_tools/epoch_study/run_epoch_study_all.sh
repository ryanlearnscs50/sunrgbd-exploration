#!/usr/bin/env bash
# MASTER orchestrator — optimal-epoch study, fully unattended (2026-06-10 task).
# Launch DETACHED so it survives the Claude session closing:
#   cd /data3/ryan/tr3d && setsid bash tools_incremental/run_epoch_study_all.sh \
#       </dev/null >logs/epoch_study_master.log 2>&1 &
#
# Pipeline (3stage):
#   0. wait for the per-stage scout (epochtrace s2,s3) to finish -> frees GPUs.
#   1. NAIVE full-chain sweep E in {2,4,6,8} (E=12 already exists). 2 GPUs, 2 waves.
#   2. PSEUDO full-chain sweep E in {2,4,6,8} (seeds stage1 from the E-epoch naive).
#   3. analyze -> write logs/EPOCH_STUDY_REPORT.txt ; touch logs/epoch_study.DONE
# Not `set -e`: a single failed chain must not abort the whole study (analysis
# tolerates missing eval logs).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
PY="$REPO/venv/bin/python"
SCHED=3stage
REPORT="$REPO/logs/EPOCH_STUDY_REPORT.txt"
say(){ echo "[$(date -Is)] $*"; }

say "===== EPOCH STUDY MASTER START ====="

# 0. wait for the scout to release the GPUs (skip if it never ran).
if ls logs/epochtrace_3stage_s*.log >/dev/null 2>&1; then
  say "waiting for scout (epochtrace s2,s3) to COMPLETE..."
  while ! { grep -q COMPLETE logs/epochtrace_3stage_s2.log 2>/dev/null && \
            grep -q COMPLETE logs/epochtrace_3stage_s3.log 2>/dev/null; }; do sleep 30; done
  say "scout done."
fi

run_wave(){  # $1=runner  $2,$3 = two epoch budgets (each -> its own GPU)
  # plain background (NOT setsid): the master is already detached, so its children
  # run under it and `wait` correctly blocks until each full chain finishes. (setsid
  # here would fork+exit and make `wait` return immediately -> broken sync.)
  local runner="$1" ea="$2" eb="$3"
  say "WAVE: $runner E=$ea (GPU0) + E=$eb (GPU1)"
  bash "tools_incremental/$runner" "$SCHED" "$ea" 0 </dev/null \
      >"logs/${runner%.sh}_${ea}.log" 2>&1 &
  local pa=$!
  bash "tools_incremental/$runner" "$SCHED" "$eb" 1 </dev/null \
      >"logs/${runner%.sh}_${eb}.log" 2>&1 &
  local pb=$!
  wait "$pa"; say "  E=$ea done (rc=$?)"
  wait "$pb"; say "  E=$eb done (rc=$?)"
}

# 1. NAIVE sweep
say "===== NAIVE SWEEP ====="
run_wave run_naive_v4_epochs.sh 2 4
run_wave run_naive_v4_epochs.sh 6 8

# 2. PSEUDO sweep (needs the E-epoch naive stage1 from step 1)
say "===== PSEUDO SWEEP ====="
run_wave run_pseudo_v4_epochs.sh 2 4
run_wave run_pseudo_v4_epochs.sh 6 8

# 3. analysis + report
say "===== ANALYSIS ====="
{
  echo "######################################################################"
  echo "# OPTIMAL-EPOCH STUDY REPORT  ($(date -Is))"
  echo "# Task: find the optimal #epochs (<12?) for incremental fine-tuning so"
  echo "# old classes are retained better. TR3D paper schedule = 12 epochs."
  echo "######################################################################"
  echo
  echo "===================== FINAL-STAGE SWEEP (the answer) ====================="
  "$PY" tools_incremental/analyze_epoch_sweep.py --sched "$SCHED" 2>&1
  echo
  echo "===================== SCOUT: per-stage per-epoch curves ================="
  echo "(stage in isolation, prior fixed at its real 12-epoch value)"
  for S in 2 3; do
    "$PY" tools_incremental/analyze_epoch_trace.py --sched "$SCHED" --stage "$S" --metric 0.25 2>&1
  done
} | tee "$REPORT"

touch "$REPO/logs/epoch_study.DONE"
say "===== EPOCH STUDY MASTER COMPLETE -> $REPORT ====="
