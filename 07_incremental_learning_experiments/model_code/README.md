# model_code — modified TR3D / mmdet3d source

These three files are **drop-in replacements** for files in the TR3D `mmdet3d` tree. They are included here
for reference (to show exactly what changed); to actually run, copy them over the corresponding paths in a
TR3D checkout.

| file here                  | replaces (in TR3D)                                  |
|----------------------------|-----------------------------------------------------|
| `tr3d_head.py`             | `mmdet3d/models/dense_heads/tr3d_head.py`           |
| `mink_single_stage_inc.py` | `mmdet3d/models/detectors/mink_single_stage_inc.py` (new file) |
| `detectors__init__.py`     | `mmdet3d/models/detectors/__init__.py`              |

---

## 1. `tr3d_head.py` — empty-batch crash fix

**Why:** the thin incremental stages (e.g. 6-stage "kitchen" = one new class, fridge, in ~40/232 scenes,
trained novel-only) occasionally yield a training batch with **zero** ground-truth objects. The stock loss
did `torch.cat(bbox_losses)` on an empty list → `RuntimeError: expected a non-empty list of Tensors`, killing
the run.

**Change (in `_loss`):**
- When `bbox_losses` is empty, set `bbox_loss` to a **graph-connected zero**
  (`sum(t.sum() for level in bbox_preds for t in level) * 0.0`) instead of `torch.cat([])`. Note `bbox_preds`
  is a list-over-levels of lists-over-scenes, so it must be double-flattened.
- Clamp the classification denominator (`torch.sum(torch.cat(pos_masks)).clamp(min=1)`) to avoid divide-by-zero
  on an all-negative batch.

**Safety:** this is a **no-op for any batch with ≥1 positive**, so it does not change TR3D's canonical results
(verified the stock 66.6/49.6 SUN RGB-D reproduction is unaffected). It also protects the pseudo/distill
variants, not just naive.

---

## 2. `mink_single_stage_inc.py` — distillation detector (`MinkSingleStage3DDetectorInc`)

A subclass of `MinkSingleStage3DDetector` that adds **static-teacher logit distillation** — the closest port
of full SDCoT onto TR3D.

- Builds a **frozen** stage *t−1* teacher (copies of the backbone/neck/head cfgs with `head.n_classes=k_old`,
  loaded from `teacher_ckpt`, set to `eval()` with `requires_grad=False`).
- `forward_train` adds `distill_loss = w * sqrt(MSE(center(student_cls[:, :k_old]), center(teacher_cls)))`,
  where `center` subtracts the per-voxel mean over the class dimension (ported from SDCoT's
  `get_distillation_loss`).
- **Voxel alignment is free:** teacher and student voxelize identical points, so MinkowskiEngine's sparse
  coordinate maps are weight-independent and the classification rows match 1-to-1 per (level, scene). A guard
  skips the term on any rare shape mismatch.
- The teacher is stored **list-wrapped** (`self._teacher = [model]`) so it is **not** registered as an
  `nn.Module` child — it is therefore excluded from `state_dict()`, keeping saved checkpoints in plain
  detector format (so eval with the stock config and next-stage head expansion both still work). It is moved
  to the right device lazily in `forward_train`.
- Weight is a constant `w_distill = 1.0` (no epoch ramp — kept simple/robust; tunable via the config).

Used only by `step_c_tools/run_distill_v4.sh`, which sets `model.type`, `model.teacher_ckpt`,
`model.n_old_classes`, and `model.loss_distill_weight` via `--cfg-options`.

## 3. `detectors__init__.py`

Registers `MinkSingleStage3DDetectorInc` so the config `model.type` can resolve it.
