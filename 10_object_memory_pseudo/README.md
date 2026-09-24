# 10 — Object replay with pseudo supervision

This series tests why object replay underperformed scene replay, then isolates
old-class pseudo supervision, replay probability, and within-stage learning rate.
SUN RGB-D 40-class detection uses five stages of eight classes; all continuations
start from the same released stage-1 checkpoint.

**Main findings:** pseudo labels raise the matched object result from 8.08% to
22.82% final mAP@0.25. A 25% replay attempt probability beats pseudo-only by
0.975 percentage points on average across three continuation seeds under the
original constant LR. Cosine improves final mAP@0.50 in both tested seeds, but
replay's added mAP@0.25 under cosine is +1.522 points in seed 201 and zero at
reported precision in seed 202. No default is changed on this evidence.

| Report | Contents |
|---|---|
| [Weekly report](WEEK_3_WRAP_UP.md) | Consolidated results, diagnostics and limitations |
| [Initial comparison](WEEK_3_RESULTS.md) | Pseudo-only versus object replay with pseudo labels |
| [Dose replication](WEEK_3_REPLICATION.md) | Three continuation seeds and stage/cohort effects |
| [Cosine replication](WEEK_3_COSINE_REPLICATION.md) | Two seeds, both replay policies, both IoU thresholds |
| [Source audit](OBJECT_MEMORY_PEISHENG_COMPARISON.md) | Supervision and implementation differences |
| [Experiment notes](EXPERIMENT_NOTES.md) | Protocol, runtime and reproduction setup |
| [Final run inventory](WEEK_3_RUN_STATUS.md) | Twelve valid completed runs and two excluded attempts |
| `week3_runs/` | Compact metrics, diagnostic JSON, original manifests and hashes |
| [Source overlay](../09_object_level_memory/ldmr_overlay/) | Updated implementation, configs, tests and analysis tools |

![Initial pseudo-supervision comparison](visualizations/week3_pseudo_comparison.png)

The two original cosine attempts did not apply the scheduler override and are
explicitly marked invalid. Corrected runs include verified logged LR decay in
every trained stage. The best individual run is 25.157% @0.25 / 15.251% @0.50;
this does not establish a matched improvement over full LDMR.

Large checkpoints, raw training logs, datasets and object-bank pickle files are
excluded. The private comparison repository's source is not redistributed.
