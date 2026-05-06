# Best-Ordering Split Analysis Report — V2 (Ordering-Optimised)

**Dataset:** SUN RGB-D · 10,295 scenes · Top-24 classes (wall/floor/ceiling excluded)

**Optimisation criterion:** Maximise the minimum surviving stage count.
We do not optimise for total scenes. A split where every stage has a reasonable
number of scenes is far more useful than one where a few stages have thousands
and others have single-digit counts.

---

## What "Best Ordering" Means

The **class-to-scene-type assignments** (determined by lift, section 6 of the script)
are fixed. Only the **stage number** — which scene type appears first, second, etc.
— changes across the ordering sweep.

The strict filter says: Stage-N training data must contain **zero classes from any later
stage**. This makes stage position critical: a class assigned to a late stage becomes
forbidden in *all* earlier stages. Common classes like `chair` (57% of scenes) and
`table` (46%) are catastrophic when placed in a late stage — they contaminate and
eliminate most scenes from every earlier stage.

The ordering sweep tests all n! permutations and picks the one where no single stage
collapses.

---

## 1. Best Ordering — 3-Stage Split

**Default ordering:** bedroom → office → classroom
**Best ordering:**    classroom → office → bedroom
**Ordering changed:** YES

### Lift Table (best ordering column sequence)

| Class | Classroom | Office | Bedroom | Assigned to |
---||---||---||---||---|
| chair | **1.552** | 1.227 | 0.635 | classroom |
| table | **1.175** | 1.084 | 0.546 | classroom |
| window | **0.917** | 1.167 | 0.849 | classroom |
| door | **1.017** | 1.002 | 0.952 | classroom |
| cabinet | 0.613 | **1.570** | 0.505 | office |
| shelf | 0.412 | **1.448** | 1.110 | office |
| desk | **2.366** | 2.462 | 1.028 | classroom |
| box | 0.487 | **2.097** | 1.709 | office |
| picture | 0.370 | 0.726 | **1.969** | bedroom |
| book | **0.281** | 2.093 | 2.052 | classroom |
| paper | 0.664 | **2.772** | 1.157 | office |
| lamp | 0.112 | 0.549 | **2.641** | bedroom |
| sofa | **0.054** | 0.409 | 0.539 | classroom |
| pillow | 0.032 | 0.132 | **3.333** | bedroom |
| bed | 0.009 | 0.071 | **5.385** | bedroom |
| monitor | 0.736 | **4.605** | 0.326 | office |
| curtain | 0.807 | 0.280 | **2.922** | bedroom |
| bottle | 0.333 | 1.331 | **1.672** | bedroom |
| bag | 0.288 | 1.428 | **2.308** | bedroom |
| mirror | 0.096 | 0.189 | **1.209** | bedroom |
| counter | **0.248** | 0.379 | 0.065 | classroom |
| keyboard | 0.693 | **4.979** | 0.195 | office |
| light | 0.939 | **1.208** | 0.550 | office |
| cup | 0.177 | **2.244** | 0.769 | office |

Bold = assigned stage (highest lift subject to capacity).

### Stage Definitions

| Stage | Scene type | Classes assigned by lift |
|---|---|---|
| Stage 1 – Classroom | classroom (1,021 scenes) | chair, window, counter, sofa, desk, table, book, door |
| Stage 2 – Office | office (1,040 scenes) | keyboard, monitor, paper, cup, cabinet, box, shelf, light |
| Stage 3 – Bedroom | bedroom (1,082 scenes) | bed, pillow, curtain, lamp, picture, mirror, bag, bottle |

### Survival Results

| Stage | Scene type | Total scenes | Surviving | % Lost |
|---|---|---|---|---|
| Stage 1 – Classroom | classroom | 1,021 | 622 | 39.1% |
| Stage 2 – Office | office | 1,040 | 680 | 34.6% |
| Stage 3 – Bedroom | bedroom | 1,082 | 1,082 | 0.0% |

*** UNVIABLE — fewer than 50 scenes.

**Minimum stage count: 622**
**Total surviving: 2,384**

### Comparison with Default Ordering

| Scene type | Default survived | Best ordering survived | Change |
|---|---|---|---|
| bedroom | 116 | 622 | +506 |
| office | 25 | 680 | +655 |
| classroom | 1,021 | 1,082 | +61 |

---

## 2. Best Ordering — 6-Stage Split

**Default ordering:** bedroom → living_room → office → classroom → library → dining_area
**Best ordering:**    dining_area → classroom → library → bedroom → living_room → office
**Ordering changed:** YES

### Lift Table (best ordering column sequence)

| Class | Dining Area | Classroom | Library | Bedroom | Living Room | Office | Assigned to |
---||---||---||---||---||---||---||---|
| chair | **1.576** | 1.552 | 1.474 | 0.635 | 0.847 | 1.227 | dining_area |
| table | **1.945** | 1.175 | 1.298 | 0.546 | 1.299 | 1.084 | dining_area |
| window | **1.204** | 0.917 | 1.044 | 0.849 | 0.990 | 1.167 | dining_area |
| door | 0.690 | **1.017** | 0.448 | 0.952 | 1.194 | 1.002 | classroom |
| cabinet | 0.255 | 0.613 | 0.338 | 0.505 | **1.032** | 1.570 | living_room |
| shelf | 0.170 | 0.412 | **1.556** | 1.110 | 1.152 | 1.448 | library |
| desk | 0.122 | 2.366 | **1.570** | 1.028 | 0.301 | 2.462 | library |
| box | 0.280 | 0.487 | 0.239 | 1.709 | **1.093** | 2.097 | living_room |
| picture | 0.253 | 0.370 | 0.133 | 1.969 | **2.352** | 0.726 | living_room |
| book | 0.055 | 0.281 | **1.700** | 2.052 | 2.180 | 2.093 | library |
| paper | 0.264 | 0.664 | 0.515 | 1.157 | 1.208 | **2.772** | office |
| lamp | 0.077 | 0.112 | 0.628 | **2.641** | 2.123 | 0.549 | bedroom |
| sofa | 0.359 | 0.054 | 0.629 | 0.539 | **4.408** | 0.409 | living_room |
| pillow | 0.041 | 0.032 | **0.043** | 3.333 | 3.071 | 0.132 | library |
| bed | 0.023 | 0.009 | 0.000 | **5.385** | 0.335 | 0.071 | bedroom |
| monitor | 0.030 | 0.736 | 0.968 | 0.326 | 0.381 | **4.605** | office |
| curtain | 0.368 | 0.807 | 0.161 | **2.922** | 2.265 | 0.280 | bedroom |
| bottle | 0.184 | **0.333** | 0.129 | 1.672 | 0.811 | 1.331 | classroom |
| bag | 0.097 | **0.288** | 0.204 | 2.308 | 2.150 | 1.428 | classroom |
| mirror | 0.178 | 0.096 | 0.000 | **1.209** | 0.617 | 0.189 | bedroom |
| counter | **0.784** | 0.248 | 0.262 | 0.065 | 0.349 | 0.379 | dining_area |
| keyboard | 0.000 | 0.693 | 1.321 | 0.195 | 0.201 | **4.979** | office |
| light | 0.418 | **0.939** | 0.571 | 0.550 | 0.631 | 1.208 | classroom |
| cup | 0.641 | 0.177 | 0.096 | 0.769 | 1.381 | **2.244** | office |

Bold = assigned stage.

### Stage Definitions

| Stage | Scene type | Classes assigned by lift |
|---|---|---|
| Stage 1 – Dining Area | dining_area (395 scenes) | table, counter, window, chair |
| Stage 2 – Classroom | classroom (1,021 scenes) | bottle, light, door, bag |
| Stage 3 – Library | library (376 scenes) | pillow, shelf, desk, book |
| Stage 4 – Bedroom | bedroom (1,082 scenes) | bed, curtain, mirror, lamp |
| Stage 5 – Living Room | living_room (524 scenes) | sofa, cabinet, box, picture |
| Stage 6 – Office | office (1,040 scenes) | keyboard, monitor, paper, cup |

### Survival Results

| Stage | Scene type | Total scenes | Surviving | % Lost |
|---|---|---|---|---|
| Stage 1 – Dining Area | dining_area | 395 | 223 | 43.5% |
| Stage 2 – Classroom | classroom | 1,021 | 434 | 57.5% |
| Stage 3 – Library | library | 376 | 230 | 38.8% |
| Stage 4 – Bedroom | bedroom | 1,082 | 461 | 57.4% |
| Stage 5 – Living Room | living_room | 524 | 402 | 23.3% |
| Stage 6 – Office | office | 1,040 | 1,040 | 0.0% |

*** UNVIABLE — fewer than 50 scenes.

**Minimum stage count: 223**
**Total surviving: 2,790**

### Comparison with Default Ordering

| Scene type | Default survived | Best ordering survived | Change |
|---|---|---|---|
| bedroom | 41 | 223 | +182 |
| living_room | 14 | 434 | +420 |
| office | 16 | 230 | +214 |
| classroom | 31 | 461 | +430 |
| library | 29 | 402 | +373 |
| dining_area | 395 | 1,040 | +645 |

---

## 3. Why Common Classes Drive the Ordering

The dominant effect is always the stage position of **generic, ubiquitous classes**:

| Class | Overall frequency | Assigned scene type |
|---|---|---|
| chair | 57.0% of all scenes | classroom |
| table | 45.7% of all scenes | classroom |
| window | 27.8% of all scenes | classroom |
| door | 24.9% of all scenes | classroom |

When the scene type containing `chair` and `table` is placed in Stage 1, those classes
are never future-forbidden for any subsequent stage. Every scene that contains a chair
or table is eligible for later stages. Placing that scene type last is the single biggest
cause of early-stage collapse in the default ordering.

---

## 4. Remaining Limitations

The ordering sweep is the maximum improvement available **within the current
strict-filter framework**. If survival counts are still too low after reordering,
the root cause is the strict filter itself — see Options A–C in
`split_analysis_report_v2.md`.
