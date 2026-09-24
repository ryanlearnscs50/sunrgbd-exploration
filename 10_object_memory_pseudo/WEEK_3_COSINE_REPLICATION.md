# Week 3 cosine replication

Same released stage-1 checkpoint; seed 202 is a follow-up to seed 201.
Only verified complete cosine runs enter the comparison. Values are mAP fractions.

| Seed | Policy | Stage | mAP@.25 | Old @.25 | New @.25 | mAP@.50 | Delta @.25 vs original LR | Delta @.50 vs original LR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 201 | 25% replay | 2 | 0.40751 | 0.53098 | 0.28404 | 0.28074 | +0.00904 | +0.00587 |
| 201 | 25% replay | 3 | 0.34606 | 0.38770 | 0.26278 | 0.22471 | +0.01488 | +0.01735 |
| 201 | 25% replay | 4 | 0.31100 | 0.32516 | 0.26852 | 0.19763 | +0.01339 | +0.01162 |
| 201 | 25% replay | 5 | 0.25157 | 0.28598 | 0.11391 | 0.15251 | +0.00545 | +0.01220 |
| 202 | 25% replay | 2 | 0.40756 | 0.52683 | 0.28829 | 0.28367 | +0.00986 | +0.00839 |
| 202 | 25% replay | 3 | 0.34727 | 0.38600 | 0.26981 | 0.22669 | +0.01592 | +0.01675 |
| 202 | 25% replay | 4 | 0.30550 | 0.31775 | 0.26874 | 0.19538 | -0.00538 | +0.00794 |
| 202 | 25% replay | 5 | 0.24867 | 0.28525 | 0.10235 | 0.15182 | -0.00058 | +0.00779 |
| 201 | Pseudo only | 2 | 0.40409 | 0.53167 | 0.27651 | 0.28195 | +0.00792 | +0.01191 |
| 201 | Pseudo only | 3 | 0.33904 | 0.38229 | 0.25253 | 0.22026 | +0.00649 | +0.00892 |
| 201 | Pseudo only | 4 | 0.29967 | 0.31334 | 0.25866 | 0.18961 | -0.00468 | +0.00228 |
| 201 | Pseudo only | 5 | 0.23635 | 0.26949 | 0.10380 | 0.14394 | +0.00114 | +0.01108 |
| 202 | Pseudo only | 2 | 0.41021 | 0.53339 | 0.28704 | 0.28436 | +0.00871 | +0.01017 |
| 202 | Pseudo only | 3 | 0.34362 | 0.38567 | 0.25951 | 0.22482 | +0.00666 | +0.01255 |
| 202 | Pseudo only | 4 | 0.31107 | 0.32223 | 0.27758 | 0.19186 | +0.00636 | -0.00098 |
| 202 | Pseudo only | 5 | 0.24867 | 0.28524 | 0.10240 | 0.14816 | +0.01003 | +0.01145 |

## Run checks

- dose25_cosine_v2: exit=0; verified=True; LR stages observed=4; error=none.
- dose25_cosine_v2_s202: exit=0; verified=True; LR stages observed=4; error=none.
- pseudo_cosine_v2: exit=0; verified=True; LR stages observed=4; error=none.
- pseudo_cosine_v2_s202: exit=0; verified=True; LR stages observed=4; error=none.

Experiment series completed September 24. All runs finished; no further launches queued. See WEEK_3_WRAP_UP.md for interpretation.
A second continuation seed does not measure stage-1 variability or establish statistical significance.
