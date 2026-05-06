# Incremental Split Analysis Report
**Dataset:** SUN RGB-D · 10,295 scenes · Top-24 classes (wall/floor/ceiling excluded; `sign` dropped for arithmetic divisibility)

---

## 1. Split Definitions

### Sem-A — Semantic, 3 stages × 8 classes

| Stage | Classes |
|-------|---------|
| Stage 1 – Bedroom/Lounge | bed, pillow, lamp, curtain, mirror, picture, sofa, light |
| Stage 2 – Office/Study | desk, monitor, keyboard, chair, table, shelf, book, paper |
| Stage 3 – Kitchen/Utility | counter, cabinet, bottle, cup, box, bag, window, door |

### Sem-B — Semantic, 6 stages × 4 classes

| Stage | Classes |
|-------|---------|
| Stage 1 – Bedroom core | bed, pillow, lamp, curtain |
| Stage 2 – Bedroom accessories | mirror, picture, sofa, light |
| Stage 3 – Office core | monitor, keyboard, desk, chair |
| Stage 4 – Study/library | table, shelf, book, paper |
| Stage 5 – Kitchen | counter, cabinet, bottle, cup |
| Stage 6 – Storage/circulation | box, bag, window, door |

### Surv-A — Survival (frequency-descending), 3 stages × 8 classes

| Stage | Classes |
|-------|---------|
| Stage 1 – High-freq (structural/furniture) | chair, table, window, door, cabinet, shelf, desk, box |
| Stage 2 – Mid-freq (decor/study objects) | picture, book, paper, lamp, sofa, pillow, bed, monitor |
| Stage 3 – Low-freq (kitchen/accessories) | curtain, bottle, bag, mirror, counter, keyboard, light, cup |

### Surv-B — Survival (frequency-descending), 6 stages × 4 classes

| Stage | Classes |
|-------|---------|
| Stage 1 – Top-freq 1–4 | chair, table, window, door |
| Stage 2 – Top-freq 5–8 | cabinet, shelf, desk, box |
| Stage 3 – Mid-freq 9–12 | picture, book, paper, lamp |
| Stage 4 – Mid-freq 13–16 | sofa, pillow, bed, monitor |
| Stage 5 – Low-freq 17–20 | curtain, bottle, bag, mirror |
| Stage 6 – Low-freq 21–24 | counter, keyboard, light, cup |

---

## 2. Survival Table

A scene **survives** Stage N if and only if (a) it contains ≥1 Stage N class, **and** (b) it contains zero classes from any later stage. The "before filter" column counts scenes satisfying condition (a) alone, computed from the full 10,295-scene annotation corpus. "Surviving" counts come from `incremental_splits_all.csv`.

### Sem-A

| Stage | Before Filter | Surviving | % Lost |
|-------|--------------|-----------|--------|
| Stage 1 – Bedroom/Lounge | 4,581 | 442 | 90.4% |
| Stage 2 – Office/Study | 8,144 | 2,727 | 66.5% |
| Stage 3 – Kitchen/Utility | 6,808 | 6,808 | 0.0% |

### Sem-B

| Stage | Before Filter | Surviving | % Lost |
|-------|--------------|-----------|--------|
| Stage 1 – Bedroom core | 2,774 | 151 | 94.6% |
| Stage 2 – Bedroom accessories | 3,281 | 291 | 91.1% |
| Stage 3 – Office core | 6,353 | 706 | 88.9% |
| Stage 4 – Study/library | 6,456 | 2,021 | 68.7% |
| Stage 5 – Kitchen | 2,866 | 1,001 | 65.1% |
| Stage 6 – Storage/circulation | 5,807 | 5,807 | 0.0% |

### Surv-A

| Stage | Before Filter | Surviving | % Lost |
|-------|--------------|-----------|--------|
| Stage 1 – High-freq (structural/furniture) | 9,287 | 3,254 | 64.9% |
| Stage 2 – Mid-freq (decor/study objects) | 5,209 | 2,554 | 51.0% |
| Stage 3 – Low-freq (kitchen/accessories) | 4,169 | 4,169 | 0.0% |

### Surv-B

| Stage | Before Filter | Surviving | % Lost |
|-------|--------------|-----------|--------|
| Stage 1 – Top-freq 1–4 | 8,392 | 2,037 | 75.7% |
| Stage 2 – Top-freq 5–8 | 4,811 | 1,217 | 74.7% |
| Stage 3 – Mid-freq 9–12 | 3,811 | 982 | 74.2% |
| Stage 4 – Mid-freq 13–16 | 3,331 | 1,572 | 52.8% |
| Stage 5 – Low-freq 17–20 | 2,680 | 1,905 | 28.9% |
| Stage 6 – Low-freq 21–24 | 2,264 | 2,264 | 0.0% |

---

## 3. Key Findings

**Why Sem-A and Sem-B lose so many scenes in early stages.** The strict filter forbids a scene from Stage N if it contains any class belonging to a later stage. In the semantic splits, the grouping logic prioritises room-type coherence: bedroom objects form Stage 1, office objects Stage 2, and kitchen/circulation objects Stage 3. This means that chair (present in 57% of all scenes) and table (present in 46%) are assigned to Stage 2 — and therefore act as forbidden contaminants for every Stage 1 scene. A scene cannot be used for Stage 1 training if it contains a single chair or table, yet those two classes appear together in the majority of the dataset. The result is that Sem-A Stage 1 loses 90.4% of eligible scenes (4,581 before the filter → 442 surviving), and Sem-B Stage 1 loses 94.6% (2,774 → 151). The pattern compounds at finer granularity: Sem-B Stages 1–3 all lose more than 88% of their eligible scenes, because the high-frequency classes window (28%) and door (25%) are likewise placed last, making them forbidden for five of the six stages.

**Why Surv-A and Surv-B survive much better.** The survival-oriented splits reverse the logic: classes are assigned to stages in strictly descending order of scene frequency. Chair, table, window, and door therefore land in Stage 1 — the earliest stage — and are never a forbidden future class for any later stage. A scene in Stage 2 or Stage 3 is free to contain any of these high-frequency objects without penalty. Losses still occur because the low-frequency classes in Stage 3 (Surv-A) or Stage 6 (Surv-B) become forbidden for earlier stages, but those classes appear in at most 8% of scenes each, so the contamination is modest. The worst loss in Surv-A is 64.9% at Stage 1, compared to 90.4% for Sem-A Stage 1 — a meaningful improvement driven entirely by placing the most common classes first.

---

## 4. Recommendation

**Surv-A is the most viable split for model training.** With 3,254 / 2,554 / 4,169 surviving scenes across its three stages, every stage clears 2,500 scenes — a threshold large enough to train a detection head without severe data starvation. The frequency-descending ordering keeps losses bounded and predictable: Stage 2 loses roughly half its eligible scenes (51.0%) and Stage 1 loses about two-thirds (64.9%), both far more tolerable than the 90%+ losses seen in the semantic splits. That said, Sem-A should be retained as a semantically meaningful baseline: its stage groupings are grounded in Jaccard co-occurrence analysis and produce interpretable room-type curricula (bedroom → office → kitchen). Stage 1 of Sem-A has only 442 scenes and is clearly too small to train from scratch, but it serves as a controlled test of whether a model pre-trained on later stages can be fine-tuned or evaluated on bedroom-only data — a scientifically interesting question even if the scene count is low.
