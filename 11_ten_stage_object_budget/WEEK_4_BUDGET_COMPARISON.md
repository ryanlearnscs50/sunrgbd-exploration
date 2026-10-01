# Week 4 — Object-count budget comparison

Updated: 2026-09-30T16:33:10.324868+00:00

Only completed, audited seed pairs are shown. AP is percent; deltas are percentage points relative to the matching B=100 seed.

| Cap/class | Seed | Stage-10 old objects | Final objects | Object reduction | Final @.25 | Delta @.25 | Final @.50 | Delta @.50 | Stage avg @.25 | Stage avg @.50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 200 | 3600 | 3979 | 0.00% | 16.363 | +0.000 | 9.634 | +0.000 | 31.290 | 21.268 |
| 100 | 201 | 3600 | 3979 | 0.00% | 15.000 | +0.000 | 9.032 | +0.000 | 30.494 | 20.637 |
| 50 | 200 | 1800 | 2000 | 49.74% | 14.045 | -2.318 | 8.622 | -1.012 | 29.700 | 19.976 |
| 50 | 201 | 1800 | 2000 | 49.74% | 15.440 | +0.440 | 9.146 | +0.114 | 30.517 | 20.588 |
| 20 | 200 | 720 | 800 | 79.89% | 14.609 | -1.754 | 8.263 | -1.371 | 29.740 | 19.778 |
| 20 | 201 | 720 | 800 | 79.89% | 14.895 | -0.105 | 9.246 | +0.214 | 30.640 | 20.464 |

## Two-seed averages

| Cap/class | Final @.25 mean ± sample SD | Final @.50 mean ± sample SD | Mean paired delta @.25 |
|---|---|---|---|
| 100 | 15.681 ± 0.964 | 9.333 ± 0.426 | +0.000 |
| 50 | 14.742 ± 0.986 | 8.884 ± 0.371 | -0.939 |
| 20 | 14.752 ± 0.202 | 8.755 ± 0.695 | -0.929 |

Stage average is the unweighted mean of the ten seen-class stage mAPs. Final objects include the last cohort added after training; stage-10 old objects describe the bank available during final-stage replay.

Audit checks: successful exit/completion markers, all ten metric scopes and class means, nonempty terminal checkpoints/bank pickles/pseudo-label files, bank caps and seed, actual LR schedule and finite logged losses. Model/bank pickle payloads are not deserialized.

These are full-training seeds under the fixed local object+pseudo protocol. Two seeds provide descriptive variability, not statistical equivalence. Scene-memory entry counts are different units; this report makes no equal-size comparison to original LDMR. The separate stage-2 selector diagnostic is reported in WEEK_4_SELECTION_RECOVERY.md.
