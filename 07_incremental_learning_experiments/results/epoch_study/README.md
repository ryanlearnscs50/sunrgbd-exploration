# Optimal-epoch study (3-stage)

Does incremental fine-tuning train *too long*? The TR3D paper schedule is 12 epochs; the worry was that on
novel-only data the model overfits the new classes and forgets old ones, so a shorter budget might retain
more. This study sweeps epoch budgets E∈{2,4,6,8,12} for both the naive and pseudo variants on the 3-stage
schedule — and, because pseudo old-class mAP was still climbing at E=12, **extends the pseudo sweep to
E∈{16,24}** — plus per-stage per-epoch scout traces.

- **`EPOCH_STUDY_REPORT.txt`** — the full numbers: final-stage old/new/overall mAP@0.25 & @0.50 per budget
  for naive and pseudo (pseudo through E=24), plus the per-stage per-epoch curves.
- **`epoch_sweep_3stage.png`** — old/new/overall mAP@0.25 vs epoch budget for both variants (pseudo to E=24).

**Conclusion: keep E=12 — epoch budget is the wrong lever for forgetting.** Naive old-class retention is
flat-near-zero and noisy across all E. Pseudo old-class mAP *rises monotonically* with epochs and **keeps
rising past 12** (E2 0.032 → E12 0.059 → E24 0.0675, +15% over E12, with no overfit turnover through E=24):
pseudo injects teacher old-class labels into every stage, so each epoch optimizes a joint old+new objective
(rehearsal-by-distillation) rather than novel-only — there is no forgetting pressure to stop early for. The
only cost of longer training is on new classes (new@.25 dips after E16): so E12 stays the sound default, E16
is the best *balanced* budget, and E24 maximizes *old-class* retention — all minor knobs.

> An E=36 chain was launched to locate the eventual old-class turnover but was cut for time — only its stage1
> completed, so it contributes no final-stage point. The report and curve here cover the naive sweep (E≤12)
> and the pseudo sweep through E=24.

Reproduce: `bash ../../step_c_tools/epoch_study/run_epoch_study_all.sh` then
`python ../../step_c_tools/epoch_study/analyze_epoch_sweep.py --sched 3stage`.
