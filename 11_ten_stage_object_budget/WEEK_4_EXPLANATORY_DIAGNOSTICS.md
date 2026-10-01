# Week 4 explanatory diagnostics

These diagnostics explain the measured storage tradeoff and where accuracy changes occur. They do not establish a causal mechanism or statistical equivalence.

## Equal crop counts and unequal storage

The exhaustive production-path audit selects 800 dense crops containing 10,990,248 points and occupying 251.70 MiB. This is 3.60 times the mean random B=20 bank size. Relative to random B=100, object count falls by 79.89% but bytes fall by only 23.25%. This bank was built offline from training GT; no full ten-stage detector score is attached to it.

The selection audit independently enumerated every eligible crop and compared stable top-20 identities and point counts for all 40 classes against the production output. Dense support is the selection criterion; it is not a measured guarantee of representativeness or semantic quality.

## Final accuracy by introduction cohort

Each row averages four class APs at stage 10, then averages the two full-training seeds. All values are percentages, except the delta in percentage points.

| Cohort | Classes | B100 AP25 | B50 AP25 | B20 AP25 | B20 minus B100 |
|---|---|---|---|---|---|
| 1 | chair, table, pillow, sofa_chair | 39.263 | 37.536 | 37.833 | -1.430 |
| 2 | desk, bed, sofa, computer | 37.508 | 35.984 | 34.866 | -2.642 |
| 3 | lamp, box, garbage_bin, cabinet | 11.039 | 10.348 | 11.453 | +0.415 |
| 4 | shelf, drawer, night_stand, endtable | 4.327 | 3.096 | 3.683 | -0.644 |
| 5 | sink, picture, stool, coffee_table | 7.120 | 8.242 | 8.005 | +0.885 |
| 6 | bookshelf, painting, keyboard, dresser | 4.652 | 5.473 | 5.601 | +0.948 |
| 7 | tv, whiteboard, cpu, toilet | 21.206 | 20.633 | 20.637 | -0.570 |
| 8 | paper, ottoman, bench, recycle_bin | 17.259 | 12.925 | 13.049 | -4.211 |
| 9 | monitor, printer, plant, door | 9.560 | 9.507 | 7.551 | -2.008 |
| 10 | book, mirror, laptop, towel | 4.881 | 3.682 | 4.845 | -0.037 |

B=20 loses AP25 in both seeds for 11/40 classes and gains in both for 7/40. Other classes have opposing effects or a zero change. These counts are descriptive and use no significance threshold. Classwise effects can vary widely even when aggregate mAP changes by less than one percentage point.

The largest mean class loss is recycle_bin: −19.1705 AP25 points (seed changes −19.482 and −18.859). With equal class weighting over 40 classes, this contributes −0.4792625 pp to the total −0.9295 pp mAP change. This is an arithmetic decomposition, not evidence of the mechanism causing the loss.

Artifacts: `figures/week4/week4_bank_costs.png`, `figures/week4/week4_cohort_and_seed_effects.png`, PDF copies, class/cohort CSV files and `week4_explanatory_summary.json`. Underlying accuracy comes only from the six audited full runs.
