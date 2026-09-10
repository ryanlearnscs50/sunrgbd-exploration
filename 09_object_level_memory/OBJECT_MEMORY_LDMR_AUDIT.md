# Object memory versus LDMR: implementation audit

**Date:** 2026-09-09
**Scope:** SUN RGB-D five-stage 8x5 protocol only.

## Bottom line

The current object bank is mechanically sound but algorithmically a minimal
random-replay baseline. It reuses LDMR's stage trainer, dynamic-head expansion,
class masks, evaluation, and forgetting reports. It does **not** yet implement
the mechanisms that make LDMR more than a memory buffer: Design-2
learning-dynamics selection, intra-stage review, or old-class pseudo labels.

Consequently, comparing object replay at .0808 mAP@.25 directly with the
released full LDMR checkpoint at .2503 is not a fair object-vs-scene test.

## Mechanism comparison

| Mechanism | Full LDMR scene path | Current object path | Consequence |
|---|---|---|---|
| Storage unit | 528 referenced full scenes (10% of 5,285) | 20 self-contained crops/class, 800 final | Different budget semantics; scenes retain context and often multiple objects |
| Candidate selection | Design-2 recall trajectories at IoU .50 | Deterministic random per class | No preference for learnable/forgettable objects |
| Class allocation | Class-need and supply weighting, minimum quotas | Exact fixed 20/class | Balanced, but not difficulty-adaptive |
| Redundancy | Dynamics-space redundancy penalty | None beyond random sampling | Similar objects can occupy multiple seats |
| Cross-stage evolution | Stage-ratio constrained swaps | Old classes remain fixed; each new class gets 20 seats | No learned eviction/replacement policy |
| Replay | Whole stored scenes appended to the training set | One crop pasted into each current natural scene | Object replay loses original support, co-occurrence, and background context |
| Intra-stage review | Four checkpoints; forgotten seats are oversampled | Disabled | Replay distribution cannot react while forgetting occurs |
| Old-class pseudo labels | Enabled on natural/replay scenes | Disabled | Naturally present old objects may remain unlabeled during later stages |
| Metrics | Per-seat TP/FP/FN trajectories, class need, replay priority, forgetting | Final validation/cohort metrics only | Metrics report outcomes but do not drive object selection or replay |

The final object bank has good source diversity: 800 objects come from 701
source scenes, with 17--20 distinct source scenes within each class. The main
issue is therefore not obvious duplication.

## Why object replay can still be weaker even in a matched experiment

An axis-aligned crop from a ground-truth 3D box is not an instance mask. It can
carry background, support-surface, or occluder points that happen to lie in the
box. Moving those points with the object creates label noise. Placement also
samples XY uniformly inside the scene envelope and checks overlap with labeled
boxes, but it does not model support surfaces, walls, free space, or semantic
compatibility. Preserving source-relative height fixed one real error, but not
these contextual mismatches.

Scene replay has none of those synthetic-placement errors and preserves object
co-occurrence and room geometry. Object replay's potential advantage is a
compact, class-balanced, self-contained memory with much finer control over
what is rehearsed. Realizing that advantage requires object-aware selection
and placement rather than random crops alone.

## Reusable LDMR machinery

The most direct extension is to keep LDMR's existing Design-2 definitions and
change the seat from a scene to an object:

1. Evaluate natural source scenes at LDMR's review checkpoints and retain
   per-scene, per-class TP/FP/FN trajectories.
2. Attach the source scene/class term to each candidate object. For an object
   seat, supply is one; class need still comes from the global recall gap.
3. Rank candidates by the Design-2 unary term and enforce the existing
   per-class object quota. Apply the same gain/drop dynamics-space redundancy
   penalty, with deterministic tie-breaking and preferably one object per
   source scene before allowing repeats.
4. During a stage, track object-seat forgetting and sample replay objects with
   LDMR's `ld_drop` weighting rather than uniformly.
5. Separately enable old-class pseudo labels so unlabeled old objects in current
   natural scenes are not trained as background. Keep this as a separate
   ablation because it is not a memory-selection change.

Source scene/class inheritance is the safest first version: it reuses the
tested LDMR evaluator and scoring formulas without pretending a cropped object
can be evaluated like a complete detector input. A later version can add
object-specific features or carrier-scene evaluation if source-level ties are
too coarse.

## Required experiments, in order

1. **Matched granularity control:** same seed/checkpoint, random scene memory,
   no pseudo labels, no reviewing. Compare it with random object memory. This
   isolates scene versus object replay and has not yet been run locally.
2. **Object Design-2:** replace random object selection while keeping pseudo and
   reviewing off. This isolates selection quality.
3. **Object Design-2 plus reviewing:** add adaptive replay weights.
4. **Object Design-2 plus reviewing plus pseudo labels:** compare the complete
   extension with full LDMR.

The released stage-1 checkpoint does not include the stage-1 learning-dynamics
JSON needed for faithful Design-2 selection. We must either reproduce the
stage-1 trajectory run or explicitly test a static-checkpoint approximation;
the latter should not be labeled faithful LDMR Design-2.
