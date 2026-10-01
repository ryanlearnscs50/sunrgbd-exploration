# Week 4 ten stage object memory study

Experimental phase closed. Six full ten-stage runs passed the completion audit. Four matched stage-2 diagnostics also passed.

Evidence refresh: 2026-10-01T05:26:14.298973+00:00.

Reducing random object memory from 100 to 20 crops per class reduces the final bank from 3,979 to 800 objects (79.89%). Mean final mAP@0.25 falls from 15.6815% to 14.7520%, a loss of 0.9295 percentage points. This does not meet the provisional 0.5-point retention tolerance. Two seeds show a storage–accuracy tradeoff; they do not establish statistical equivalence.

## Question and experimental design

This study extends the earlier five-stage work to ten stages, uses the reference object budget, reduces the number of stored objects, repeats across two seeds, and tests whether another selection criterion makes a smaller bank useful.

The completed sweep uses SUN RGB-D with 40 frequency-ordered classes introduced four at a time. For each budget (100, 50 and 20 objects per class), seeds 200 and 201 train from scratch through all ten stages. Stage 1 uses six epochs and later stages one epoch each, with dataset repeat 15, batch size 16 and AdamW LR 0.001. LR falls to 0.0001 for the sixth initial epoch and remains 0.001 in later stages. Random selection, pseudo supervision, insertion probability 0.7, up to three candidate pastes and all placement settings stay fixed. Only the per-class and total object caps change.

Week 3 reduced the probability of replay while leaving stored capacity fixed. Week 4 changes stored capacity while holding the configured replay dose fixed. Accepted paste counts were not logged, so equal configured dose is not a measurement of equal accepted pastes.

## Accuracy and stored object budget

AP is reported as a percentage; losses are percentage points. SD is the sample standard deviation of two full runs, not a confidence interval.

| Objects per class | Actual final objects | Final AP25 mean ± SD | Final AP50 mean ± SD | Mean AP25 change | Bank MiB mean |
|---|---|---|---|---|---|
| 100 | 3,979 | 15.6815 ± 0.964 | 9.3330 ± 0.426 | +0.0000 | 327.95 |
| 50 | 2,000 | 14.7425 ± 0.986 | 8.8840 ± 0.371 | -0.9390 | 166.83 |
| 20 | 800 | 14.7520 ± 0.202 | 8.7545 ± 0.695 | -0.9295 | 69.93 |

The B=20 paired changes are −1.754 and −0.105 pp. B=50 changes are −2.318 and +0.440 pp. B=50 and B=20 have nearly the same mean, and the seed ordering reverses at B=50. These observations do not support a precise monotonic budget curve.

At the provisional 0.5 pp loss tolerance, neither reduced budget passes even on the mean. At 1 pp, both means pass but neither passes in both seeds. At 2 pp, B=20 passes in both seeds. These are descriptive sensitivity checks, not a revised acceptance criterion or an equivalence test. No budget smaller than B=100 has demonstrated retention within 0.5 pp in this study.

## What storage and accuracy measurements mean

The 3,979-object final baseline falls below its nominal 4,000 cap because mirror, laptop and towel provide fewer eligible crops. The final bank is written after adding the last cohort; stage-10 training instead uses the stage-9 bank: 3,600 old objects for B=100 and 720 for B=20. These are separate quantities.

Random B=20 occupies roughly 69–71 MiB versus 327–329 MiB for B=100, but total run time only changes from about 6.8 to 6.7 hours. The measured benefit is mainly bank storage under the fixed training schedule. Runtime includes evaluation, pseudo-label generation and bank construction; it is not a controlled throughput benchmark.

Final AP25 gives equal weight to each of the 40 classes. recycle_bin loses 19.1705 class AP points on average at B=20, contributing 0.4793 pp to the overall 0.9295 pp loss. Eleven classes lose in both seeds and seven gain in both. This decomposition identifies where the aggregate change occurs; it does not identify its cause.

The stage-average metric averages seen-class mAP across ten different class scopes. A falling stage curve alone is not a forgetting estimate. The separate forgetting measure averages prior-peak-minus-final AP for the first 36 classes; an improvement beyond the prior peak contributes a negative value. See the full analysis for both metrics and final old/new AP.

## Selection criterion and recovery

Both full ten-stage largest-point-count runs failed after completing stage-1 training. Their bank implementation supported the selector, but the SUN RGB-D population wrapper rejected it. The preflight tested the isolated ranking function and missed the actual population path. The repaired path scans all eligible crops and retains a bounded top-20 set with stable ties. Four new population regression cases and 22 prior checks passed. An independent exhaustive ranking confirmed the selected identities and point counts for all 40 classes. Failed run evidence is retained.

In the bounded stage-2 comparison, largest-point-count selection changes AP25 by +0.870 and +1.080 pp for seeds 200 and 201 (mean +0.9750 pp). Each pair shares the exact same saved stage-1 checkpoint and resets the continuation seed. The comparison covers eight seen classes and uses 80 replay objects; it does not provide a final ten-stage selection result. Pseudo-label boxes and scores also differ within each pair, so the gain cannot be attributed solely to crop selection. Full details and old/new AP are in [the recovery report](WEEK_4_SELECTION_RECOVERY.md).

The offline 40-class largest-point-count bank contains 800 crops, 10,990,248 points and 251.70 MiB. It uses 3.60 times the mean bytes of random B=20. Against random B=100 it saves 79.89% of objects but only 23.25% of bytes. This is bank-construction evidence, with no full ten-stage detector score. More points are not by themselves proof of better semantic quality or class representativeness.

## Relation to the reference work

The retained reference audit establishes the nominal 100-object-per-class budget and the ten-stage class split, but does not identify a verified ten-stage object-memory score to reproduce. The separate 19.38% ten-stage scene-memory result has a different protocol and budget unit. Local box-origin/collision fixes, uniform subset selection, object sampling without replacement and the absence of extra crop yaw remain implementation differences. This is a local ten-stage extension and controlled object-budget study, not a confirmed numerical reproduction.

Reference commits: TR3D_OBJ `cd667a180c3cbf4eae29cf779ea907592400266f`; LDMR_backup `1c58607f04b86f1b66233b39a07b631f624f201c`. The local [protocol audit](WEEK_4_PROTOCOL_AUDIT.md) and [training plan](WEEK_4_TRAINING_PLAN.md) retain the source evidence.

## Conclusion and next experiment

Random B=20 is a substantial storage reduction with a measured mean cost of about 0.93 pp AP25. It is a candidate when that loss is acceptable, not a demonstrated same-performance replacement under the provisional 0.5 pp criterion. Agree on an acceptable loss before selecting a deployment budget. A third matched full-training seed and a completed ten-stage selector comparison would strengthen the evidence. Targeting classes with consistent losses is a hypothesis for a later study, not a validated budget-allocation rule. No additional full runs are queued.

## Evidence and reproducibility

- [Full numerical analysis](WEEK_4_ANALYSIS.md) and [budget comparison](WEEK_4_BUDGET_COMPARISON.md).
- [Selection repair and bounded results](WEEK_4_SELECTION_RECOVERY.md).
- [Storage and class diagnostics](WEEK_4_EXPLANATORY_DIAGNOSTICS.md).
- [Research slide PDF](presentation/week4/WEEK_4_RESEARCH_SLIDES.pdf) and [speaker notes](presentation/week4/WEEK_4_SPEAKER_NOTES.md).
- Machine-readable full-run results: `week4_runs/analysis.json`; bounded results: `week4_runs/selection_recovery/audit.json`.
- Full-run completion audits check exits, class scopes and metric means, bank caps/seed/selector, finite training losses, LR, and nonempty checkpoints/pseudo artifacts. Model payloads are not deserialized by that audit.
