# Week 3 replay replication

Same released stage-1 checkpoint; seeds vary continuation training and random bank selection.
Values are mAP@0.25 fractions. Only completed jobs contribute.

| Seed | Pseudo only | 25% object replay + pseudo | Paired difference |
|---|---:|---:|---:|
| 201 | 0.2352 | 0.2461 | 0.0109 |
| 202 | 0.2386 | 0.2492 | 0.0106 |
| 203 | 0.2347 | 0.2425 | 0.0077 |

Mean paired difference: +0.0097; sample SD across 3 seeds: 0.0018.

Seed 201 selected the 25% candidate; seeds 202 and 203 are follow-up checks.
These runs share a pretrained checkpoint and do not measure variability from training stage 1.
A few seeds provide limited evidence; no default is changed automatically.

Queue details: `week3_runs/overnight_20260924/status.json`.

Follow-up seeds only: mean difference +0.0092 (2 pairs).

## Stage and cohort checks

Mean paired differences (25% replay minus pseudo-only); old/new classes are defined at each stage.

| Stage | Pairs | mAP@.25 | Old mAP@.25 | New mAP@.25 | mAP@.50 |
|---|---:|---:|---:|---:|---:|
| 2 | 3 | -0.0029 | -0.0070 | +0.0012 | +0.0014 |
| 3 | 3 | -0.0040 | -0.0042 | -0.0038 | -0.0028 |
| 4 | 3 | -0.0016 | -0.0016 | -0.0016 | -0.0024 |
| 5 | 3 | +0.0097 | +0.0108 | +0.0057 | +0.0044 |

Final old- and new-class mAP@.25 improve in all three seeds. Earlier-stage effects are mixed.
Final mAP@.50 improves in seeds 201 and 202, but decreases in seed 203.
The two follow-up seeds support the selected dose; they do not establish broad statistical significance.
