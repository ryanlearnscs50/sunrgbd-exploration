# Optimal-epoch study (3-stage)

Does incremental fine-tuning train *too long*? The TR3D paper schedule is 12 epochs; the worry was that on
novel-only data the model overfits the new classes and forgets old ones, so a shorter budget might retain
more. This study sweeps epoch budgets E∈{2,4,6,8,12} for both the naive and pseudo variants on the 3-stage
schedule, plus per-stage per-epoch scout traces.

- **`EPOCH_STUDY_REPORT.txt`** — the full numbers: final-stage old/new/overall mAP@0.25 & @0.50 per budget
  for naive and pseudo, plus the per-stage per-epoch curves.
- **`epoch_sweep_3stage.png`** — old/new/overall mAP@0.25 vs epoch budget for both variants.

**Conclusion: keep E=12 — epoch budget is the wrong lever for forgetting.** Naive old-class retention is
flat-near-zero and noisy across all E. Pseudo old-class mAP *rises monotonically* with epochs
(E2 0.032 → E12 0.059): pseudo injects teacher old-class labels into every stage, so each epoch optimizes a
joint old+new objective (rehearsal-by-distillation) rather than novel-only — there is no forgetting pressure
to stop early for.

> **Extension in progress.** Because pseudo was still climbing at E=12, a sweep to E∈{16,24,36} is running to
> locate the plateau/overfit point. The report and curve here cover E≤12; the extended results will replace
> them in a later update. (The PNG here is the E≤12 preview.)

Reproduce: `bash ../../step_c_tools/epoch_study/run_epoch_study_all.sh` then
`python ../../step_c_tools/epoch_study/analyze_epoch_sweep.py --sched 3stage`.
