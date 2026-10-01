# Week 4 selection recovery

Updated: 2026-10-01T05:26:10.560796+00:00.

The full ten-stage largest-point-count runs failed at stage-1 bank population. The bank accepted the selector but the SUN RGB-D population wrapper rejected it. The earlier preflight tested ranking and missed this integration path. The failed runs and their exit-1 evidence are preserved.

The repair scans the complete eligible crop pool and retains the top 20 per class with stable ties, using a bounded heap. Four new population regression cases and 22 prior focused checks passed. The random branch retains its seeded candidate order.

## Bounded comparison

Both selectors use the same saved stage-1 terminal checkpoint within each seed (200 or 201), reset the continuation seed, and train stage 2 for one epoch. Both have B=20 and identical resolved configuration except selection. The head has 8 seen classes; replay contains 80 old objects from stage 1. The 160-object bank written after stage 2 includes the four new classes and is not the bank used during this diagnostic. These runs do not estimate final ten-stage accuracy.

All four stage-2 runs passed the completion audit.

| Seed | Selector | Status |
|---|---|---|
| 200 | random | complete |
| 200 | largest_point_count | complete |
| 201 | random | complete |
| 201 | largest_point_count | complete |

## Stage 2 results

AP is in percent; differences below are percentage points.

| Seed | Selector | AP25 | AP50 | Old AP25 | New AP25 | Replay points | Replay MiB | Minutes |
|---|---|---|---|---|---|---|---|---|
| 200 | random | 48.788 | 34.933 | 50.839 | 46.737 | 294,981 | 6.77 | 27.44 |
| 200 | largest_point_count | 49.658 | 35.137 | 52.950 | 46.367 | 1,786,103 | 40.90 | 28.36 |
| 201 | random | 49.194 | 35.119 | 51.421 | 46.968 | 262,197 | 6.02 | 27.77 |
| 201 | largest_point_count | 50.274 | 36.063 | 53.758 | 46.791 | 1,786,103 | 40.90 | 28.53 |

| Seed | Largest minus random AP25 | AP50 | Old AP25 | New AP25 | Identical pseudo files |
|---|---|---|---|---|---|
| 200 | +0.870 | +0.204 | +2.110 | -0.370 | False |
| 201 | +1.080 | +0.944 | +2.337 | -0.176 | False |

Mean stage-2 AP25 change is +0.9750 pp. This is an early-stage selection diagnostic. It cannot establish ten-stage retention or repair the missing full selection comparison. Equal seeds do not guarantee bitwise deterministic GPU training.

A separate content audit found different saved pseudo-label boxes and scores in all 1,549 seed-200 scenes and all 1,525 seed-201 scenes; one scene per seed also differs in detection count. The hashes differ for more than metadata alone. These are comparisons in saved detection order, without rematching. The observed AP gain therefore cannot be attributed solely to the selector. A subsequent selector comparison should reuse one cached pseudo-label artifact per paired checkpoint. See `week4_runs/selection_recovery/pseudo_content_audit.json`.

## Completion audit

For each accepted run, checks cover exit 0, completion marker, no traceback, checkpoint lineage and hashes, exactly eight metric classes and consistent means, finite losses and LR .001, stage-1/2 bank capacities and selector/seed, and nonempty pseudo labels and terminal checkpoint. Random stage-1 bank identities also match the corresponding original B=20 bank. Pairing checks identical checkpoint hashes. Pseudo file identity is reported separately. Full-source hashes and the accumulated diff were saved before launch.

Sources: `week4_runs/selection_recovery/audit.json` and per-run manifests/logs. All four trainers finished naturally on October 1, 2026 at approximately 11:18 SGT. The experimental phase is closed.
