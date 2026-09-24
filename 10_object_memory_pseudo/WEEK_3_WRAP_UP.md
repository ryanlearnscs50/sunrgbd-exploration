# Week 3 wrap-up — September 23–24, 2026

Experiment series completed September 24, 2026.
Scope: incremental 3D detection on SUN RGB-D, 40 classes, five stages of eight classes.

## Final operational state

All **12 valid training continuations completed successfully**. Two earlier,
invalid cosine attempts were intentionally stopped and excluded. The last run,
25% replay + cosine, seed 202, finished at **16:06:30 SGT**; its pseudo-only
partner finished at **15:53:16**. At 16:56, a host-level check showed both GPUs
at 0% utilization, no GPU compute jobs, and no remaining project experiment,
queue, or monitor processes. The overnight queue and both cosine monitors exited 0.
Nothing needs stopping and no further experiment is queued.

Closeout rechecked zero exits, completion markers, nonempty final checkpoints,
correctly scoped stage-2–5 metrics, and final banks for every replay run. All four
valid cosine runs have logged LR decay in all four trained stages. Seed-202
cosine replay's entire initial bank also matches its constant-LR control exactly,
including arrays and metadata; evidence is in
`week3_runs/diagnostics/cosine_s202_initial_bank_match.json`.

## What we learned

### 1. Missing old-class supervision explained much of the weak object baseline

Last week's random-object result was only 8.08% final mAP@0.25. The audit of
Peisheng's pinned TR3D_OBJ source showed that its reported object experiments used
old-class pseudo labels, while our completed object experiments did not. Our
pipeline filtered old annotations but retained their points and active loss
channels. At stage 5, this leaves 2,733 old instances across 606 of 644 natural
training scenes without their ground-truth labels. Pasting one old object cannot
replace supervision for all those naturally occurring objects.

The controlled seed-201 comparison strongly supports addressing that mismatch:

| Policy | Final mAP@0.25 (%) |
|---|---:|
| Historical matched object replay, no pseudo labels | 8.08 |
| Object replay + pseudo labels, 100% attempt probability | 22.824 |
| Pseudo labels only | 23.521 |

Adding pseudo supervision to the object policy gains about **14.74 percentage
points**, but full-dose object replay still trails pseudo-only by **0.697 points**
and trails it at every trained stage. This compares complete training policies:
later teachers and augmentation streams diverge. It does not assign the entire
cross-repository gap to one cause.

### 2. Less replay helped final performance under the original schedule

We isolated the probability of attempting one paste per scene, preserving the
20-object/class bank, selection, placement, pseudo thresholds and training budget.
Actual accepted insertion can be lower because of collision rejection.

| Attempt probability, seed 201 | Final mAP@0.25 (%) |
|---|---:|
| 0% / pseudo-only | 23.521 |
| 25% | **24.612** |
| 50% | 23.406 |
| 100% | 22.824 |

The selected 25% policy was then checked against pseudo-only on two more seeds:

| Continuation seed | Pseudo-only (%) | 25% replay + pseudo (%) | Gain (percentage points) |
|---|---:|---:|---:|
| 201, selection seed | 23.521 | 24.612 | +1.091 |
| 202, follow-up | 23.864 | 24.925 | +1.061 |
| 203, follow-up | 23.475 | 24.247 | +0.772 |
| Mean | 23.620 | 24.595 | **+0.975** |

Sample SD of the paired gain is 0.176 points; follow-up seeds alone average
+0.917 points. Final old- and new-class mAP@0.25 both improve in every seed.
However, mean stage-2–4 mAP@0.25 changes are negative, and final mAP@0.50 falls
in seed 203. This is evidence for better final retention/performance under this
schedule, not a uniform improvement throughout learning.

### 3. Cosine helps stricter-IoU performance; the replay interaction did not replicate

The original step milestones [8, 11] never fire in the two-/one-epoch continuation
stages, so their effective LR is constant. We tested within-stage iteration cosine
decay from about .001 toward .0001 at the same training budget.

| Seed | Policy | Constant LR @.25 (%) | Cosine @.25 (%) | Constant LR @.50 (%) | Cosine @.50 (%) |
|---|---|---:|---:|---:|---:|
| 201 | Pseudo-only | 23.521 | 23.635 | 13.286 | 14.394 |
| 201 | 25% replay + pseudo | 24.612 | **25.157** | 14.031 | **15.251** |
| 202 | Pseudo-only | 23.864 | 24.867 | 13.671 | 14.816 |
| 202 | 25% replay + pseudo | 24.925 | 24.867 | 14.403 | 15.182 |

Cosine improves final mAP@0.50 in all four matched comparisons, by 0.779–1.220
points. At mAP@0.25, seed-202 replay slightly declines against constant LR
(-0.058 points). Replay's benefit over pseudo-only under cosine is +1.522 points
in seed 201 but **0.000 at reported precision in seed 202**. Under cosine,
replay still adds +0.857 and +0.366 points at mAP@0.50, respectively.

The best individual run is 25.157% / 15.251% at IoU .25 / .50, but the second
seed does **not** establish that cosine and replay reliably reinforce each other
at IoU .25. It also does not establish a win over full LDMR: the older released
checkpoint evaluation of 25.03% used a different training history and is not a
matched comparator.

## Diagnostics and implementation work

- Pseudo labels are useful but incomplete: object-branch old-class recall at
  IoU .25 is 67.97% in stage 2 and 38.86% in stage 5, with precision 93.17%
  and 84.82%. These are different class/scene cohorts, not longitudinal recall
  on a fixed population. Stage-5 pseudo-only recall is 37.07%.
- In an offline 256-scene placement audit, 239 pastes were accepted (93.36%).
  None had grossly inadequate sampled point/voxel support under the audit's
  thresholds. However, 117 accepted pastes overlap hidden-old-object AABB
  envelopes; that is a conservative geometry proxy, not proven collision.
- Added controlled pseudo/dose/cosine configs, detached launch and queue tooling,
  persistent status, artifact checks, coverage audits, and result summaries.
- Found and fixed a real scheduler handoff defect: explicit incremental
  `lr_config` was dropped when preparing each stage. The initial cosine pair
  therefore stayed at constant LR. Both attempts were stopped (exit 143), marked
  invalid, preserved, and restarted from the shared stage-1 checkpoint after
  the fix. Regression tests exercised actual stage-config propagation; recorded
  focused checks passed. Actual training logs now independently verify decay.
- Corrected the historical Design-2 comparison description: its matched random
  control shared the stage-1 checkpoint, but only 2/160 initial object identities.
  It compares whole selection policies, not just subsequent bank updates.

## Conclusions and open questions

Pseudo supervision is the strongest supported correction. A 25% replay attempt
probability is a promising candidate for final-stage performance with the original
schedule. Cosine provides consistent final mAP@0.50 gains in the tested pairs,
but its mAP@0.25 effects and interaction with replay remain mixed. No default was
changed.

All seeds share the released stage-1 checkpoint: these are continuation/bank
replications, not independent full-training seeds. Three dose seeds and two cosine
seeds are limited evidence, with candidate selection on seed 201. Stochastic
inference, evolving teachers/banks and altered worker RNG also limit attribution.
Cosine changes average LR as well as decay shape.

The most direct unresolved check is
the missing matched cosine seed-203 pair, completing the schedule comparison on
the existing continuation seeds. Pseudo-coverage improvements and placement/context
changes should remain separate later interventions. Design-2/reviewing have not
been revalidated under the improved supervision.

The source overlay, configs, tests and analysis tools accompany this report.
Large checkpoints, banks and raw training logs remain local. Compact result and
provenance files are included under `week3_runs/`.

## Evidence index

- `OBJECT_MEMORY_PEISHENG_COMPARISON.md`: pinned source audit and objective mismatch.
- `WEEK_3_RESULTS.md`: initial pseudo comparison and trajectory figure.
- `WEEK_3_REPLICATION.md`: dose replication and stage/cohort tradeoffs.
- `WEEK_3_COSINE_RESULTS.md`: seed-201 schedule analysis (historical).
- `WEEK_3_COSINE_REPLICATION.md`: verified results for both cosine seeds.
- `WEEK_3_RUN_STATUS.md`, `week3_runs/status.json`: refreshed final run inventory.
- `EXPERIMENT_NOTES.md`: protocol, runtime requirements and reproducibility notes.
- `week3_runs/diagnostics/`: bank matching, pseudo coverage and placement evidence.
