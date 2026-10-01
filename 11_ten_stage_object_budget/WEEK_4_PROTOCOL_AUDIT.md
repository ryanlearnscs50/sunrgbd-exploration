# Week 4 ten stage protocol and budget audit

Reference audit conducted September 30, 2026; consolidated after the October 1 closeout.

## Finding and target status

The ten-stage study is feasible with the existing SUN RGB-D metadata
and class mapping. However, **the supplied repository does not establish a
ten-stage object-bank result to reproduce** in the evidence inspected.

The retained GitHub `main` snapshot is `cd667a180c3cbf4eae29cf779ea907592400266f`, the Week 3
reference revision. The branch listing returned only `main`; the recursive tree
was complete (8,777 entries, `truncated=false`). Its six `tr3d_objbank_*` configs
and object-bank launcher cover three and five stages. The ten-stage SUN RGB-D
configs cover finetuning and scene-memory methods. A `20x10x10` filename means
three stages with 20, 10, and 10 classes, not ten stages.

The [object-bank report](https://github.com/qianpeisheng/TR3D_OBJ/blob/cd667a180c3cbf4eae29cf779ea907592400266f/OBJBANK_PLAN.md)
reports these final mAP@0.25 percentages:

| Protocol | Pseudo only | Object + pseudo | Object + pseudo + masking |
|---|---:|---:|---:|
| Three stages, 20+10+10 | 28.19 | 25.37 | 25.07 |
| Five stages, 8×5 | 25.03 | 18.33 | 19.10 |

These are repository-reported results, not independently reproduced scores. The
report's named `work_dirs/objbank_feasibility/analysis/feasibility_report.md`
is absent from the recursive tree. No ten-stage object score should be inferred
from these numbers. Historical scene-memory reports also exist in the tree;
this audit does not claim the repository lacks ten-stage experiments altogether.

Our downloaded ten-stage checkpoint manifest reports **19.38%**, but its source
run is LDMR with scene memory, pseudo labels and reviewing. This is a separate
method, not a 100-object/class baseline. All ten local checkpoint SHA-256 hashes
match that manifest. No verified ten-stage object-memory numerical target was
identified in the retained reference evidence.

## Verified protocol and implementation differences

The local and reference class-mapping Python ASTs match after excluding the
module docstring. The supervised detector config also matches, ignoring trailing
newlines. Local metadata has 5,285 training and 5,050 validation scenes; every
annotation's numeric label matches its name in the 40-class mapping.

| Setting | Week 3 local object policy | Reference object policy / ten-stage evidence |
|---|---|---|
| Class stages | 5 × 8 | Object reports: 5 × 8 or 20+10+10; ten-stage scene config: 10 × 4 |
| Epochs | 5,2,2,1,1 | Object S5: same; ten-stage scene config: 6,1,1,1,1,1,1,1,1,1 |
| Stored object cap | 20/class; total cap 800 | 100/class; nominal final cap 4,000 |
| Crop support | Minimum 5 points | Minimum 20 points |
| Replay dose | 1 attempted crop; probability 1, .5 or .25 depending on run | 3 attempted crops with probability .7 |
| Collision threshold | 0 | .3 AABB IoU |
| Placement retries | 20 | 10 |
| Placement height | 1st-percentile floor + preserved source height | Scene minimum Z, floor placement |
| Extra crop yaw / XY jitter | No extra yaw / 0 | Random yaw / ±1 metre |
| Object selection | Uniform subset without replacement within class | Unconditional random replacement after cap is full; not uniform reservoir sampling |
| Stage-1 initialization | Shared released S5 checkpoint | Object trainer initializes from scratch |
| Seeds | 201–203 continuation seeds | Launcher uses 200 |
| LR actually used | Constant in short continuation stages; separate cosine ablations | Object trainer overrides step to `max(1, epochs−1)` per stage |
| Pseudo supervision | Enabled in Week 3 corrected runs | Enabled in reported object conditions |
| Pseudo thresholds | Confidence .5, NMS .3, pseudo/GT IoU .25, cap 100 | Same nominal settings |
| Batch / repeats / optimizer | 16 / 15 / AdamW .001, weight decay .0001 | Same in resolved configs |

The inherited LR milestones [8,11] printed by config resolution are **not** the
reference object trainer's runtime milestones. For a ten-stage adaptation with
six initial epochs, its rule gives step [5] in stage 1 and [1] in later stages;
one-epoch stages finish before a decayed training epoch begins. Actual
runtime LR was verified in all six completed runs; config parsing alone was insufficient.

The earlier static source findings about reference pasted-box Z origin and
old-class loss masking remain relevant. A local corrected implementation and
the literal reference implementation are different experimental conditions.
The completed sweep uses the unmasked object+pseudo condition and holds the local
geometry, selection, replay dose and LR fixed across budgets.

## What matching the number of objects means

The reference object budget is **100 per class**, not 100 total. At stage t of
the 4×10 protocol, nominal replay capacity is `4 × (t−1) × B`, and the bank saved
after that stage has capacity `4 × t × B`. Thus stage 10 trains with at most
3,600 old-class crops at B=100; the 4,000-cap final bank includes objects added
after the last training stage.

Actual occupancy can be smaller. Training metadata contains only 98 mirrors,
93 laptops and 91 towels, limiting the final distinct-object count to at most
3,982 even before the minimum-point requirement. The report's 3,979-object
debug bank illustrates this distinction but is not a measured ten-stage bank.
Log actual per-class occupancy, unique object identities, point counts and bytes.
The reference's approximate 400 MB figure is not our measured storage cost.

Reducing replay probability from 1 to .25 leaves the stored bank unchanged.
Week 3's dose experiments therefore do not answer the new storage-budget question.
Keep insertion settings fixed while varying B.

| Proposed B per class | Stage-10 old-object capacity | Final bank capacity | Nominal cap reduction vs B=100 |
|---|---:|---:|---:|
| 100 | 3,600 | 4,000 | 0% |
| 50 | 1,800 | 2,000 | 50% |
| 20 | 720 | 800 | 80% |
| 10 | 360 | 400 | 90% |
| 5 | 180 | 200 | 95% |

These are proposed settings and upper bounds, not measured savings or performance.
The ten-stage scene config uses a 528-scene budget. Comparing it to an object
bank requires counting retained annotations/crops and bytes; 528 scenes cannot
be equated to 528 stored objects.

## Ten-stage data check

| Stage | New classes | Natural training scenes before pseudo filtering |
|---|---|---:|
| 1 | chair, table, pillow, sofa_chair | 3,454 |
| 2 | desk, bed, sofa, computer | 1,857 |
| 3 | lamp, box, garbage_bin, cabinet | 1,246 |
| 4 | shelf, drawer, night_stand, endtable | 899 |
| 5 | sink, picture, stool, coffee_table | 602 |
| 6 | bookshelf, painting, keyboard, dresser | 568 |
| 7 | tv, whiteboard, cpu, toilet | 621 |
| 8 | paper, ottoman, bench, recycle_bin | 455 |
| 9 | monitor, printer, plant, door | 358 |
| 10 | book, mirror, laptop, towel | 309 |

Scenes can recur across stages. These counts use current-class GT eligibility,
not the final pseudo-filtered training loader. At stage 10, 291 of 309 scenes
contain 1,409 old GT instances. Preserve pseudo supervision when extending the
corrected object baseline; one or three pasted objects do not label the existing
old objects in a scene.

## Completed study scope

The final sweep contains B=100, B=50 and B=20 at full-training seeds 200 and
201. All six runs trained stage 1 independently and completed all ten stages.
A separate selector comparison contains four stage-2 checkpoint continuations.
The full ten-stage selector attempts failed at bank population and contribute
no final accuracy scores. B=0, B=10, B=5, Design-2 and a third full seed were
not evaluated this week.

The provisional retention convention was a paired mean final AP25 loss no
greater than 0.5 percentage points relative to B=100. Neither reduced budget
meets it. The final analysis also shows 1.0 and 2.0 pp sensitivity; these do not
replace the original convention. Two seeds describe variability and cannot
establish statistical equivalence.

Metrics include final and per-stage AP25/AP50, final old/new AP, signed
forgetting, actual bank objects, points, bytes, runtime and logged GPU memory.
Accepted paste counts and unique replayed identities were not measured.
See [the weekly report](WEEK_4_WRAP_UP.md) for the completed findings.

## Evidence and validation

The exported `week4_audit/protocol_evidence.json` records resolved settings,
metadata hashes and scene counts, stage cohorts, snapshot checks and released
checkpoint hashes. The source revisions are pinned above. Private comparison
source files remain outside this publication. The host audit used the local
reference snapshots, metadata and checkpoints; those large or private inputs
are not required to inspect the exported findings.
