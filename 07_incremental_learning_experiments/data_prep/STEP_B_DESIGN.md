# Step B Design Doc — Class-Incremental TR3D on SUN RGB-D (SDCoT → TR3D)

Status: **Step B complete** (data + scaffolding + this design). Training methods = Step C.
Author context: mentor's 3-part task ([A] copy+relabel data, [B] this, [C] fine-tune & pseudo-label variants).

---

## 1. Goal & setup

Build **multi-step sequential class-incremental** 3D object detection with TR3D, using SDCoT as the
structural reference. At step *t* the model already knows the classes of steps 1..t-1 and must learn
step *t*'s **new** classes without forgetting the old ones.

- Dataset: SUN RGB-D, **19 box-available classes** from ryan's scene-type split.
- Two class-introduction orderings (the "try more than one ordering" requirement):
  - **3-stage** (Classroom → Office → Bedroom): 6 → +7 → +6 = 19 classes.
  - **6-stage** (Dining → Classroom → Library → Bedroom → Living → Office): 2 → 4 → 8 → 12 → 15 → 19.
- This differs from SDCoT, which is a **single** base→novel step. We generalize: each step's "static
  teacher" is the **previous stage's** model; "old classes" = all classes from earlier stages.

---

## 2. What Step B produced (recap, with paths)

Foundation (Step A): `data/sunrgbd_incremental/` — standalone copy, 10335 `points/*.bin` shared by all stages.

**Data (chunks 1-3):**
- `stage_manifest/` — every split-CSV scene → mmdet3d index → train/val membership (0 unmapped).
- `<ordering>/sunrgbd_infos_train_stage{t}.pkl` — stage-t train, GT filtered to that stage's **novel**
  classes only; `class` field = **global gid** in introduction order; `index` re-enumerated.
- `<ordering>/sunrgbd_infos_val_stage{t}.pkl` — **cumulative** val (all classes seen through stage t).
- Builders: `incremental_learning/{build_stage_manifest,carve_stage_pkls,build_val_pkls}.py`.

**Scaffold (chunk 4):**
- `tr3d/configs/tr3d_incremental/<ordering>/stage{t}.py` — self-contained TR3D configs; per stage:
  `n_classes = len(class_names) = len(label2level) =` cumulative count; train ann = novel pkl, val ann =
  cumulative pkl; `load_from` = previous stage checkpoint.
- `tr3d/tools_incremental/run_incremental.sh` — chains stages; **as-is = naive fine-tuning baseline**;
  marked `STEP C HOOK`s for head-expansion and pseudo-label prepass.
- Generator: `incremental_learning/gen_incremental_configs.py`.

**Known data caveat (sparsity):** the scene-type split was built on 2D/scene-level class *presence* +
stage-size balancing, not on 3D boxes — so some assigned classes are near-empty in 3D (e.g. 3-stage
stage-1 `sofa`=0/`book`=0/`door`=2 boxes; 6-stage stage-2 = 5 boxes total). Mentor's call: keep the
split as designed. Expect thin tail classes in results.

---

## 3. SDCoT mechanism (reference, from `/data3/projects/SDCoT`)

VoteNet/PointNet++ based, single base→novel step. Three coordinated networks + a 3-term loss:

- **Static teacher** — frozen base-class model (`sdcot_trainer.py:54-58`). Runs on the (un-augmented)
  cloud, emits base-class logits used as the **distillation** target.
- **Dynamic teacher** — EMA copy of the student (`:48-51`, update `:137-141`, `ema_decay=0.999`). Runs on
  a differently-augmented cloud; provides the **consistency** target.
- **Pseudo-labels** — the base detector is run per scene; detections filtered (obj_conf>0.95,
  cls_conf>0.9, 3D-NMS IoU 0.25, drop boxes with <5 pts) and **merged as pseudo-GT for base classes**,
  while real GT is kept **only for novel classes** (`sunrgbd_inc.py:60-77`, `ap_helper.py:45-181`).
- **Classifier expansion** — base classifier weights copied & preserved, novel slots Kaiming-init,
  backbone NOT frozen (`sdcot_trainer.py:40-74`).
- **Total loss** (`sdcot_trainer.py:110-113`):
  `L = L_sup + w_cons·L_cons + w_distill·L_distill`, with `w_cons=10`, `w_distill=1`, both sigmoid-ramped
  over 30 epochs. `L_distill` = sqrt-MSE on **mean-centered base-class logits** (`loss_helper.py:366-378`).
  `L_cons` = center(NN) + class(KL) + size(MSE) between student and EMA, after inverse-augmentation.

---

## 4. Mapping SDCoT → TR3D

TR3D is **anchor-free, sparse-voxel** (MinkResNet → TR3DNeck → TR3DHead), not vote-based. Component map:

| SDCoT (VoteNet) | TR3D equivalent | Notes |
|---|---|---|
| proposals / votes | per-voxel predictions from `TR3DHead` | `cls_conv`, `bbox_conv` (k=1 Mink convs), `tr3d_head.py:40-43` |
| base-class logits for distill | old-class channels of `cls_preds` | `cls_preds` shape (N_vox, n_classes); take `[:, :k_old]` |
| proposal-center matching for teacher↔student | **voxel-coordinate matching** | teacher & student voxelize the *same* points deterministically → align by sparse coords. **Simpler than SDCoT** (no Hungarian/NN proposal match needed) for the static-teacher distillation path |
| classifier head | `cls_conv` only | `bbox_conv` is class-agnostic (`n_reg_outs=8`) → **does not change across stages**; expand `cls_conv` output channels only |
| vote loss + detection loss | TR3D loss (`_loss_single`, `tr3d_head.py:134-170`) | center-based assignment via `TR3DAssigner`; bbox = `RotatedIoU3DLoss(diou)`, cls = focal-style w/ background index `= n_classes` |
| labels from `instance_bboxes[:,-1]` | `annos['class']` gid (read directly, `sunrgbd_dataset.py:142`) | our pkls already store global gids |

**Head expansion (TR3D-specific).** To go stage t-1 → t, grow `cls_conv` from `k_old` to `k_new`
output channels:
```
new = MinkowskiConvolution(in_channels, k_new, kernel_size=1, bias=True, dimension=3)
new.kernel[..., :k_old] = old.kernel            # copy preserved old-class weights
new.bias[:k_old]        = old.bias
kaiming_init(new.kernel[..., k_old:]); zero/neg-bias(new.bias[k_old:])   # new classes
```
Everything else (`bbox_conv`, backbone, neck) loads unchanged from the previous checkpoint. Backbone
**not frozen** (matches SDCoT). Implement in `tools_incremental/expand_head.py`, emitting the stage-t
init checkpoint that `load_from` points to (replaces the size-mismatch reinit the skeleton currently does).

**Distillation (TR3D-specific).** Add a frozen `teacher = stage_{t-1} model` (k_old classes). For each
forward pass on the student's (un-augmented) cloud, run the teacher, align by voxel coordinate, and:
`L_distill = sqrt(MSE(center(student_cls[:, :k_old]), center(teacher_cls)))` — directly porting
`get_distillation_loss`. Optionally also distill `bbox_conv` outputs at old-class-positive voxels.

**Consistency / dynamic teacher (optional, port last).** TR3D has no proposals, so SDCoT's
center/class/size consistency needs voxel-level matching between two *augmented* views — feasible but
more involved (different augmentations → different voxel grids → need NN matching in world coords after
inverse-aug). Recommend: get fine-tuning + pseudo-labels + static-teacher distillation working first;
add the EMA consistency term only if forgetting is still high.

---

## 5. Multi-step generalization

At step *t* (t = 1..N):
- **old classes** O_t = classes of stages 1..t-1 (gids `0 .. k_{t-1}-1`); **new** = stage t (next gids).
- **static teacher** = stage t-1 checkpoint (covers O_t). Distillation preserves O_t logits.
- **pseudo-labels** = stage t-1 model run over stage-t train scenes → confident O_t boxes (covers *all*
  old classes at once, not just the immediately-previous stage).
- **GT** = real boxes for stage-t novel classes only (already what our `train_stage{t}.pkl` holds).
- step 1 = plain base training (no teacher, no pseudo).
- **eval** at step t on `val_stage{t}.pkl` (cumulative) → measure both new-class learning and old-class
  retention.

---

## 6. Step C variants (to implement next)

**Variant 1 — naive fine-tuning (baseline).** Head-expand, then train stage t on novel GT only. No
teacher, no pseudo. This is `run_incremental.sh` as written (plus the head-expand hook). Expected:
strong forgetting — the control to beat.

**Variant 2 — pseudo-labeling (mentor's priority; prefers offline).**
1. `tools_incremental/make_pseudo_labels.py`: load stage t-1 checkpoint, run inference over stage-t
   **train** scenes (`tools/test.py` plumbing / `_get_bboxes`), filter by obj/cls confidence + 3D NMS
   (port SDCoT thresholds: conf≈0.9-0.95, NMS IoU 0.25, drop <5-pt boxes), keep only **old-class** boxes.
2. Merge pseudo old-class boxes with the stage-t novel GT → write
   `<ordering>/sunrgbd_infos_train_stage{t}_pseudo.pkl` (same schema; `class`=old gids for pseudo,
   `index` re-enumerated, cap like SDCoT's MAX_NUM_OBJ if needed).
3. Train stage t on the pseudo-augmented pkl (head-expanded). Offline = simple + fast + reproducible
   (no base detector in the data loader), exactly as mentor prefers.
4. **Optional add-on:** also enable static-teacher distillation (Section 4) for a "pseudo + distill"
   ablation — closest to full SDCoT.

Both variants reuse the same per-stage configs; only the train ann_file (real vs `_pseudo`) and the
presence of the distillation/expand hooks differ.

---

## 7. Concrete Step C work-list (files to add under `tr3d/`)

- `tools_incremental/expand_head.py` — checkpoint surgery for `cls_conv` (Section 4). **Required for all variants.**
- `tools_incremental/make_pseudo_labels.py` — offline pseudo-GT generation + pkl merge (Variant 2).
- `mmdet3d/models/detectors/mink_single_stage_inc.py` *(optional, for distillation)* — subclass of
  `MinkSingleStage3DDetector` that holds a frozen teacher and adds `L_distill` in `forward_train`;
  register in `detectors/__init__.py`; select via `model.type` in the stage configs.
- Wire the two `STEP C HOOK` lines already stubbed in `run_incremental.sh`.
- Hyperparameters to port: `ema_decay=0.999`, `w_distill=1` / `w_cons=10` (sigmoid ramp 30 ep — rescale
  to TR3D's 12-epoch schedule, e.g. ramp ≈ 6-8 ep), pseudo conf/NMS thresholds above.

---

## 8. Evaluation protocol

- Per stage t, eval on cumulative `val_stage{t}.pkl` → per-class AP@0.25 (SUN RGB-D standard).
- Report, at the final stage: overall mAP, **old-class mAP vs new-class mAP**, and a **forgetting**
  measure (drop in a class's AP from the step it was learned to the final step).
- Baselines for context: joint training on all 19 (upper bound) and naive fine-tuning (lower bound).
- Compare 3-stage vs 6-stage to see how chain length affects forgetting.

---

## 9. Open questions / risks

- **Sparsity (Section 2):** thin classes (sofa/book/door in 3-stage S1; whole 6-stage S2) may yield
  near-zero AP regardless of method. May warrant revisiting class→stage assignment with mentor, or
  reporting these as known-degenerate.
- **label2level** is a size heuristic (`max(mean dx,dy)>1.0`); may need per-ordering tuning.
- **EMA consistency** in a dense voxel head is the least-certain port — keep optional.
- **Pseudo-label quality** depends on stage t-1 detector strength; sparse early stages → weak teachers →
  noisy pseudo-labels. Confidence thresholds may need raising for sparse stages.
- **Schedule:** TR3D trains 12 epochs/stage vs SDCoT's longer schedule; ramp lengths must be rescaled.
