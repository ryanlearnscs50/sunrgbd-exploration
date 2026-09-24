# Object memory comparison with TR3D_OBJ

> Pre-experiment audit, September 23. The proposed comparisons are now complete;
> see `WEEK_3_WRAP_UP.md` for measured results. The private reference source is
> not redistributed in this repository.

Date: 2026-09-23. Scope: SUN RGB-D 40 classes, five-stage 8×5 protocol.
Reference: https://github.com/qianpeisheng/TR3D_OBJ/tree/cd667a180c3cbf4eae29cf779ea907592400266f
Selected source snapshot: `reference/TR3D_OBJ/` (not executed).

## Finding

The strongest explanatory difference is old-class supervision: **every completed
local object-bank experiment disabled pseudo labels; both of Peisheng's reported
object-bank conditions enabled them.** Our pasted crops provide old-class positives,
but naturally occurring old objects are unlabeled and can be trained as background.
Peisheng's pseudo labels supply supervision for those natural old objects.

This is a confirmed difference in training objectives and a strong causal hypothesis,
not a measured attribution of the entire cross-repository performance gap. There
are additional differences in seed, initialization, learning rate, budget and placement.
No new training was launched during this audit.

## Results and what they actually compare

All values are final five-stage mAP@0.25 percentages.

| Source / condition | Result | Old-class pseudo labels |
|---|---:|---|
| Local no-memory fine-tuning | 3.14 | No |
| Local random object, height-aware, released stage-1 checkpoint | 8.08 | No |
| Local random object, exact Design-2 stage-1 checkpoint, random initial bank | 9.31 | No |
| Local Design-2 object selection | 6.67 | No |
| Local Design-2 plus source-scene reviewing | 6.32 | No |
| Local random scene control, released stage-1 checkpoint | 19.29 | No |
| Peisheng pseudo-only | 25.03 | Yes |
| Peisheng object bank + pseudo | 18.33 | Yes |
| Peisheng object bank + pseudo + masking condition | 19.10 | Yes |

Local source: `OBJECT_MEMORY_FINDINGS.md` and the September 10 stopping-point
entry in `MEMORY.md`. Remote source: `reference/TR3D_OBJ/OBJBANK_PLAN.md`.
The remote recursive tree contains neither the named feasibility report nor the
experiment summary artifacts, so Peisheng's numbers are repository-reported,
not independently reproduced here. His pseudo-only 25.03% is a different experiment
from our numerically identical released full-LDMR checkpoint evaluation.

His own object-bank conditions trail his pseudo-only baseline by 6.70 and 5.93
percentage points. The repository does not establish that copy-paste itself was a
successful improvement. Its claim that distribution shift explains the decline is
an interpretation; the reported ablation establishes the decline, not its cause.

## Confirmed local supervision problem

1. `repo/mmdet3d/datasets/incremental_sunrgbd.py:170` uses only the current
   stage's classes in training; `_apply_class_filter()` at line 429 removes other
   annotations. Loading points does not remove those old objects.
2. `repo/tools/train_incremental_scene.py:782` enables all seen classes in the
   training mask, which is necessary for replay but also leaves old-class losses active.
3. `repo/mmdet3d/models/dense_heads/tr3d_head.py:313` constructs background targets
   for unmatched proposals; the focal loss includes the old-class channels. Naturally
   occurring old objects consequently receive no correct positive target and may
   generate false-negative supervision (or be assigned to another retained box).
4. The local object config explicitly sets `use_pseudo_labels=False`. Our completed
   Design-2/reviewing runs preserve that setting. The pseudo-enabled config exists,
   but the September 10 notes explicitly say its experiment never ran.

A read-only audit of the correct local HF training metadata counted all scenes with
at least one current-stage object, matching natural training-scene eligibility:

| Stage | Natural scenes | Scenes containing old objects | Old instances whose GT is filtered | Old instances/scene |
|---|---:|---:|---:|---:|
| 2 | 1,760 | 1,327 | 4,880 | 2.77 |
| 3 | 1,120 | 903 | 3,977 | 3.55 |
| 4 | 1,045 | 854 | 3,792 | 3.63 |
| 5 | 644 | 606 | 2,733 | 4.24 |

These are annotation counts, not a measured gradient balance or counts of visible
detector proposals. They nevertheless show why one attempted pasted crop does not
solve the missing-label problem. Pseudo labels only recover teacher-detected old
objects; they are not equivalent to restoring all old GT.

## Other verified differences

| Setting | Local preferred random object run | Peisheng object+pseudo |
|---|---|---|
| Bank cap per class | 20 | 100 |
| Minimum crop points | 5 | 20 |
| Attempted paste dose | 1 with probability 1 | 3 with probability 0.7 |
| Collision IoU threshold | 0 | 0.3 |
| Height | Source-floor offset retained | All crops placed on scene minimum Z |
| Extra per-object yaw | None | Uniform random yaw |
| Per-object XY jitter | 0 | ±1 m |
| Selection/review | Random, or source-scene Design-2/review ablations | Random replacement; no LD review |
| Seed | 201 | 200 |
| Stage-1 initialization | Released checkpoint or separate local scratch run | Scratch |
| Nominal stage epochs / repeats | 5,2,2,1,1 / 15 | 5,2,2,1,1 / 15 |
| LR schedule | Inherited milestones [8,11] | Per-stage milestone max(1, epochs−1) |

The local matched-random log confirms LR remains 0.001 in epoch 2 of stages 2/3.
Peisheng's trainer configures a step after the first epoch of those two-epoch stages
(normally 0.0001 in epoch 2). Same epoch counts therefore do not imply the same
optimization schedule. The larger bank and stricter minimum support are plausible
contributors, but have no isolated effect estimate. Nominal attempts also do not
measure successful insertion or points surviving PointSample/voxelization.

Local height-aware replay already improved 6.82% to 8.03%; Peisheng still uses floor
placement. There is no evidence that superior height handling explains his advantage.
Local scene replay's 19.29% versus object's 8.08% mostly reflects retention: novel
cohort AP is 10.72% versus 10.37%. The budgets are not matched by bytes or old-object
supervision, so this does not isolate representation alone.

Our matched-checkpoint comparison found 9.31% for random selection versus 6.67% for
Design-2. The initial banks also differ: the September 23 artifact audit found only
2 of 160 initial object identities shared. This is a selection-policy comparison
including stage-1 bank choice, not an isolation of updates after stage 1 alone.
Its scores come from source scene/class dynamics rather than direct isolated-object
quality; reviewing similarly transfers source-scene weights. Treating this proxy as
validated object selection was too optimistic. Adding reviewing gave 6.32%; that
small additional difference is not a multi-seed causal estimate.

## Cautions before copying the reference implementation

Peisheng's source is useful evidence, not a bug-free replacement:

- `bank.py` calls its policy reservoir sampling, but after filling a class it
  unconditionally replaces an existing entry for every new object. This is biased
  toward later input records, unlike uniform reservoir sampling.
- `insertion.py` builds pasted box Z at the gravity center, then passes it into the
  default box constructor without a gravity-center origin. For the Depth box
  convention used here this shifts the box relative to its points by half its height.
  It also treats existing bottom-center boxes as gravity-centered in collision checks.
- The loss masking branch identifies old classes only from crops pasted in that
  sample. It is not a blanket ignore mask for every previous class. For those classes
  it overwrites logits outside pasted matches, including potential pseudo positives;
  this should be redesigned with explicit loss weights before adoption.

These are static source findings at the pinned revision. We have not established
whether the reported experiments ran that exact revision or quantified their impact.

## Recommended next comparison

First hold seed, stage-1 checkpoint, bank, insertion and LR fixed and compare:

1. Pseudo labels only, without replay.
2. Random object replay plus the same pseudo labels.
3. The matching no-pseudo random object control (reuse only if all settings match).

Do this before reintroducing Design-2 or reviewing. Our existing full pseudo config
bundles those unsuccessful selection/review changes, so it is not the cleanest test
of the main finding. Subsequently isolate the per-stage LR schedule, 100/class budget,
minimum crop support and actual successful replay dose. Measure old/new cohort AP,
pseudo coverage, accepted insertions and retained crop points. Do not assume larger
budgets or more pasted objects will help.
