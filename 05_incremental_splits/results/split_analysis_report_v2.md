# Incremental Split Analysis Report — V2 (Data-Driven)

**Dataset:** SUN RGB-D · 10,295 scenes · Top-24 classes (wall/floor/ceiling excluded; `sign` dropped for arithmetic divisibility)

---

## What Changed From V1

V1 defined stages by **manually assigning object classes to room types**, motivated post-hoc by Jaccard co-occurrence values. The mentor's correction: the dataset already contains ground-truth room-type labels in `scene.txt` files — work from those labels first, then derive class assignments from the data.

**V2 approach:**
1. Read the ground-truth `scene.txt` label for every scene (45 unique scene types across 10,295 scenes)
2. Compute **lift** for every (class, scene type) pair: `lift = freq(class | scene_type) / freq(class overall)`. Lift > 1 means the class appears more in this scene type than its dataset-wide baseline.
3. Assign each class to the scene type where its lift is highest, using a balanced greedy algorithm to ensure 8 classes per stage (3-stage) or 4 classes per stage (6-stage).
4. **Scene membership** changes from "does the scene contain a Stage-N class?" → "does the scene's label equal the Stage-N scene type?" Stage-N training data genuinely consists of Stage-N scene types.
5. Strict filter condition (b) — no future-stage classes — is unchanged.

---

## 1. Ground-Truth Scene Type Distribution

Scene type counts in the working dataset (scenes with parsed annotations):

| Scene type | Scenes | Used in |
|---|---|---|
| office | 1,040 | 3-stage, 6-stage |
| classroom | 1,021 | 3-stage, 6-stage |
| bedroom | 1,082 | 3-stage, 6-stage |
| living_room | 524 | 6-stage |
| dining_area | 395 | 6-stage |
| library | 376 | 6-stage |
| furniture_store | ~895 | excluded (retail) |
| rest_space | ~690 | excluded (ambiguous) |
| idk | ~167 | excluded (unknown) |

---

## 2. Lift Table — 3-Stage Split

Lift = freq(class | scene_type) / freq(class overall). Values > 1.5 highlighted as strongly characteristic.

| Class | Bedroom | Office | Classroom | Assigned to |
|---|---|---|---|---|
| bed | **5.385** | 0.071 | 0.009 | bedroom |
| pillow | **3.334** | 0.132 | 0.032 | bedroom |
| curtain | **2.922** | 0.280 | 0.807 | bedroom |
| lamp | **2.641** | 0.549 | 0.112 | bedroom |
| picture | **1.969** | 0.726 | 0.370 | bedroom |
| bag | **2.308** | 1.428 | 0.288 | bedroom |
| mirror | 1.209 | 0.189 | 0.096 | bedroom |
| bottle | **1.672** | 1.331 | 0.333 | bedroom |
| keyboard | 0.195 | **4.978** | 0.693 | office |
| monitor | 0.326 | **4.605** | 0.736 | office |
| paper | 1.157 | **2.772** | 0.664 | office |
| cup | 0.769 | **2.244** | 0.177 | office |
| cabinet | 0.505 | **1.570** | 0.613 | office |
| box | 1.709 | **2.097** | 0.487 | office |
| shelf | 1.110 | **1.448** | 0.412 | office |
| light | 0.550 | **1.208** | 0.939 | office |
| chair | 0.635 | 1.227 | **1.552** | classroom |
| desk | 1.028 | 2.462 | **2.366** | classroom |
| table | 0.546 | 1.084 | **1.175** | classroom |
| book | 2.052 | 2.093 | 0.282 | classroom* |
| door | 0.952 | 1.002 | **1.017** | classroom |
| window | 0.849 | 1.167 | **0.917** | classroom |
| sofa | 0.539 | 0.409 | 0.054 | classroom* |
| counter | 0.065 | 0.379 | **0.248** | classroom |

*Assigned by capacity balancing (lift is not highest in this stage, but it is the next available slot after higher-lift stages are full).

**Key observations:**
- `bed` (lift 5.4) and `pillow` (lift 3.3) are unmistakably bedroom-specific — the data confirms the intuition.
- `keyboard` (lift 5.0) and `monitor` (lift 4.6) are unmistakably office-specific.
- `chair`, `table`, `window`, `door` have low lifts everywhere (<1.6 max) — they are **generic** classes with no strong scene-type allegiance.
- `desk` (lift 2.46 in office, 2.37 in classroom) is genuinely ambiguous — barely higher in office than classroom.
- The data-driven assignment matches what a human would intuit for the clearly characteristic classes, and reveals genuine ambiguity for the generic ones.

---

## 3. Stage Definitions — 3-Stage Split

| Stage | Scene type | Classes assigned by lift |
|---|---|---|
| Stage 1 – Bedroom | bedroom (1,082 scenes) | bed, pillow, curtain, lamp, picture, mirror, bag, bottle |
| Stage 2 – Office | office (1,040 scenes) | keyboard, monitor, paper, cup, cabinet, box, shelf, light |
| Stage 3 – Classroom | classroom (1,021 scenes) | chair, window, counter, sofa, desk, table, book, door |

---

## 4. Stage Definitions — 6-Stage Split

| Stage | Scene type | Classes assigned by lift |
|---|---|---|
| Stage 1 – Bedroom | bedroom (1,082 scenes) | bed, curtain, mirror, lamp |
| Stage 2 – Living Room | living_room (524 scenes) | sofa, cabinet, box, picture |
| Stage 3 – Office | office (1,040 scenes) | keyboard, monitor, paper, cup |
| Stage 4 – Classroom | classroom (1,021 scenes) | bottle, light, door, bag |
| Stage 5 – Library | library (376 scenes) | pillow, shelf, desk, book |
| Stage 6 – Dining Area | dining_area (395 scenes) | table, counter, window, chair |

---

## 5. Survival Results

### 3-Stage Split

| Stage | Scene type | Total scenes | Surviving | % Lost |
|---|---|---|---|---|
| Stage 1 – Bedroom | bedroom | 1,082 | 116 | 89.3% |
| Stage 2 – Office | office | 1,040 | 25 *** | 97.6% |
| Stage 3 – Classroom | classroom | 1,021 | 1,021 | 0.0% |

*** UNVIABLE — fewer than 50 scenes.

### 6-Stage Split

| Stage | Scene type | Total scenes | Surviving | % Lost |
|---|---|---|---|---|
| Stage 1 – Bedroom | bedroom | 1,082 | 41 *** | 96.2% |
| Stage 2 – Living Room | living_room | 524 | 14 *** | 97.3% |
| Stage 3 – Office | office | 1,040 | 16 *** | 98.5% |
| Stage 4 – Classroom | classroom | 1,021 | 31 *** | 97.0% |
| Stage 5 – Library | library | 376 | 29 *** | 92.3% |
| Stage 6 – Dining Area | dining_area | 395 | 395 | 0.0% |

*** UNVIABLE — fewer than 50 scenes.

### Comparison with V1

| Stage | V1 hand-coded survive | V2 data-driven survive |
|---|---|---|
| Stage 1 – Bedroom | 442 | 116 |
| Stage 2 – Office | 2,727 | 25 |
| Stage 3 – Kitchen (v1) / Classroom (v2) | 6,808 | 1,021 |

---

## 6. Why Survival Is Still Low — And What This Reveals

The data-driven class assignment is more honest than v1, but survival is actually worse. This is not a flaw in the approach — it exposes the root cause that was previously hidden.

**The root cause is the strict filter, not the class assignments.**

The strict filter says: "Stage-N training data must contain zero classes from any later stage." Because `chair` (57% of all scenes) and `table` (46%) are assigned to Stage 3 (classroom, highest lift) or Stage 6 (dining area), they are *forbidden* in every earlier stage. Any office scene that contains a chair or table — which is nearly every office scene — is eliminated from Stage-2 training data.

In V1, this problem was partially hidden because the class assignments were **manually tuned to maximise survival** (chair was placed in Stage 2 so it was never a future contaminant for Stage 2 itself). The data-driven approach places chair where it actually belongs (classroom/dining area), which is semantically correct but catastrophic for survival under the strict filter.

**The strict filter and data-driven class assignment are in tension because:**
- Generic, ubiquitous classes (chair, table, window, door) naturally belong to whichever scene type uses them most (classroom, dining area)
- But once assigned to a later stage, they contaminate and eliminate scenes from all earlier stages
- V1's "fix" was to hand-code those classes into early stages — not because they belong there semantically, but because it maximised survival

---

## 7. Options Going Forward

Three directions, in order of invasiveness:

**Option A — Relax the strict filter**
Replace "zero future classes" with "majority of present classes are from this stage." This is a softer curriculum boundary. More scenes survive; less strict separation between stages.

**Option B — Anchor generic classes to Stage 1**
Assign `chair`, `table`, `window`, `door` to Stage 1 regardless of their lift (lift is too close to uniform across scene types to be informative). They are never future-forbidden. This is a small, principled deviation from pure lift-based assignment.

**Option C — Accept low survival and merge small stages**
Merge bedroom + living room into one stage, accept that Stage 2 (office) is small, and treat this as a hard upper bound on what the strict filter allows. The splits are correct; the dataset simply does not support strict separation at this granularity.

The choice depends on how strictly SDCoT's filter must be replicated.
