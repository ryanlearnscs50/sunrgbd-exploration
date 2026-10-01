# Week 4 object budget and selection analysis

Updated: 2026-10-01T05:26:11.699313+00:00. 6 completed runs passed the ten-stage artifact audit.

The random-selection budget sweep reduces final storage from 3,979 to 800 objects (79.89%) with a mean final mAP@.25 loss of 0.9295 percentage points. This exceeds the provisional 0.5 pp tolerance. Two seeds show the tradeoff; they do not establish statistical equivalence.

The full ten-stage point-count selection pair failed at stage-1 bank population and is excluded from numerical conclusions. The population-path repair and separate matched stage-2 diagnostics are documented in WEEK_4_SELECTION_RECOVERY.md; they do not replace the missing ten-stage result.

## Final accuracy and object count

AP values are percentages; deltas are percentage points. Each condition has full-training seeds 200 and 201. Sample SD describes those two runs, not a confidence interval.

| Condition | Final objects | Final AP25 mean ± SD | Final AP50 mean ± SD | Mean paired AP25 delta vs B100 | Worst seed AP25 delta |
|---|---|---|---|---|---|
| Random B=100 | 3979 | 15.681 ± 0.964 | 9.333 ± 0.426 | +0.0000 | +0.000 |
| Random B=50 | 2000 | 14.742 ± 0.986 | 8.884 ± 0.371 | -0.9390 | -2.318 |
| Random B=20 | 800 | 14.752 ± 0.202 | 8.755 ± 0.695 | -0.9295 | -1.754 |

## Paired seed details

| Condition | Seed | Final AP25 | Delta vs B100 | Delta vs random B20 | Stage avg AP25 | Stage avg AP50 | Final old AP25 | Final new AP25 | Forgetting AP25 |
|---|---|---|---|---|---|---|---|---|---|
| Random B=100 | 200 | 16.363 | +0.000 | +1.754 | 31.290 | 21.268 | 17.660 | 4.691 | 12.467 |
| Random B=100 | 201 | 15.000 | +0.000 | +0.105 | 30.494 | 20.637 | 16.104 | 5.072 | 13.976 |
| Random B=50 | 200 | 14.045 | -2.318 | -0.564 | 29.700 | 19.976 | 15.164 | 3.977 | 14.418 |
| Random B=50 | 201 | 15.440 | +0.440 | +0.545 | 30.517 | 20.588 | 16.779 | 3.388 | 12.830 |
| Random B=20 | 200 | 14.609 | -1.754 | +0.000 | 29.740 | 19.778 | 15.646 | 5.284 | 13.289 |
| Random B=20 | 201 | 14.895 | -0.105 | +0.000 | 30.640 | 20.464 | 16.060 | 4.405 | 13.354 |

Stage average is the unweighted mean of seen-class mAP over ten stages. Old/new means cover the first 36/final 4 classes. Forgetting is the mean, over the first 36 classes, of maximum AP from introduction through stage 9 minus stage-10 AP. It is signed: an improvement beyond the prior peak contributes negatively. The evaluation set grows with stages, so raw stage-mAP decline alone is not a forgetting measure.

## Object budget and resource context

| Condition | Seed | Replay objects at stage 10 | Final objects | Final points | Final bank MiB | Hours | Peak logged GPU MB |
|---|---|---|---|---|---|---|---|
| Random B=100 | 200 | 3600 | 3979 | 14,319,421 | 328.50 | 6.82 | 3647 |
| Random B=100 | 201 | 3600 | 3979 | 14,271,624 | 327.40 | 6.84 | 3638 |
| Random B=50 | 200 | 1800 | 2000 | 7,299,944 | 167.46 | 6.74 | 3651 |
| Random B=50 | 201 | 1800 | 2000 | 7,244,922 | 166.20 | 6.80 | 3642 |
| Random B=20 | 200 | 720 | 800 | 3,080,035 | 70.65 | 6.67 | 3648 |
| Random B=20 | 201 | 720 | 800 | 3,017,222 | 69.21 | 6.70 | 3636 |

Final banks include the last cohort added after training; the stage-9 bank supplies stage-10 replay. Bank MiB is the final pickle file size. Runtime includes all stages, evaluation, pseudo labels and bank extraction. GPU memory is the maximum recorded training-log value, not a hardware-wide peak. Object count is the primary budget measure; denser crops can increase points and bytes at the same object count. Configured insertion probability/count stays fixed, but accepted paste counts are not measured.

## Sensitivity to the performance tolerance

The following descriptive checks use final AP25 relative to the seed-matched random B100 baseline. A pass means observed loss is within a chosen tolerance; it is not an equivalence test. The 0.5 pp convention was proposed before these runs; 1.0 and 2.0 pp show sensitivity, not a revised acceptance criterion chosen after seeing results.

| Condition | Object reduction | Tolerance pp | Mean loss within tolerance | Both seed losses within tolerance |
|---|---|---|---|---|
| Random B=50 | 49.74% | 0.5 | no | no |
| Random B=50 | 49.74% | 1.0 | yes | no |
| Random B=50 | 49.74% | 2.0 | yes | no |
| Random B=20 | 79.89% | 0.5 | no | no |
| Random B=20 | 79.89% | 1.0 | yes | no |
| Random B=20 | 79.89% | 2.0 | yes | yes |

## Scope and evidence for the research discussion

This is a local ten-stage extension at the reference object budget, not an exact reproduction of a verified Peisheng ten-stage object score. The retained reference audit did not identify such a target. The separate 19.38% ten-stage scene-memory result uses a different protocol; scene entries and object crops are different budget units. See WEEK_4_PROTOCOL_AUDIT.md and WEEK_4_TRAINING_PLAN.md for source commits and remaining implementation differences.

Two independent full-training seeds are available per completed condition. Equal seed numbers support paired comparisons but do not guarantee bitwise deterministic training. A third matched seed and a pre-agreed performance tolerance would strengthen a final retention claim. No B=10/5 runs were started because B=20 already exceeded the provisional loss tolerance.

Audits verify successful exits and completion markers, all metric class scopes/means, nonempty checkpoints/banks/pseudo files, budget/seed/selector settings, finite logged losses and LR schedules. Pickle/model payloads are not deserialized by the audit.

Machine-readable evidence: `week4_runs/analysis.json` and `week4_runs/week4_per_class.csv`. Exportable figures: `figures/week4/week4_final_accuracy.png` and `figures/week4/week4_stage_accuracy.png`, with PDF copies. Failed full selection runs remain excluded. See WEEK_4_SELECTION_RECOVERY.md for the separately scoped diagnostic.
