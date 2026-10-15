# LDMR and object-extension implementation contract

Use the [four-path guide](../../docs/reference_methods.md) for training commands.
Original LDMR defines scene memory; the teammate TR3D_OBJ `objbank_pseudo` branch
defines the named object recipe. UniDet3D supplies its own detector and optimizer.
The supported full-method host protocol is ScanNet-35 S5. The shared mapping also
contains S3/S10, but that does not make their pseudo/LD/reviewing methods supported.

## Method components

| Component | Scene: `--recipe ldmr` | Object: `--recipe tr3d_obj` |
|---|---|---|
| Training annotations | Current-class natural GT plus saved scene visibility | Current-class natural GT plus pasted old-class boxes |
| Teacher | Frozen completed previous-stage detector; old-class predictions | Same teacher lifecycle |
| Pseudo confidence, stages 2–5 | .45/.40/.35/.30 | .50/.50/.50/.50 |
| Pseudo filtering | Class NMS .30, cap 100, visible-GT conflict IoU .25 | Same filters |
| Pseudo on saved scenes | Enabled | No saved-scene replay |
| Bank | 120 scene entries; original Design-2 and stage-ratio quotas | 100 crops/class, max 3500; unconditional random-slot replacement |
| Replay | Original unique-scene merge and saved visibility | Three class-uniform draws with replacement, probability .7, floor placement |
| Reviewing | Original LD-drop weights and coverage-preserving scene sampling | Disabled |
| Training exposure | 12 nominal epochs ×15 passes | Same ScanNet transfer exposure |
| Host optimizer | Native AdamW lr .0002, weight decay .05, polynomial schedule | Same |

Original TR3D scene training realizes 180 passes as five resumed segments of 36
repeats. Its `[8, 11]` LR steps are unreachable in five runner epochs. The host
keeps continuous optimizer state but its native polynomial LR remains an explicit
optimization difference; matching passes does not reproduce that LR trajectory.

Object Design-2 and source-scene reviewing remain available under `--recipe custom`.
They are research extensions, not replacements for the teammate object method.
Their weighted draws without replacement and source-height placement are distinct
from `tr3d_obj`. The legacy SUN profile retains 5/2/2/1/1 nominal epochs and random
yaw. The UniDet3D adapter has no SUN dataset implementation.

## Source-to-port map

| Reference responsibility | Maintained implementation |
|---|---|
| LDMR class order, raw IDs, stages | `protocol.py` loads `configs/_base_/class_mappings/scannet_dynamic_head_mappings.py` |
| LDMR scene visibility/deduplication | `data.stage_records`, `scene.replay_records`, `scene.merge_replay_records` |
| LDMR learning-dynamics quality and scores | `learning_dynamics.py` loads maintained LDMR scorer/matcher functions; `trajectories.py` prepares their inputs |
| LDMR Design-2 quotas and selection | `selection.py` resolves original scene config; `export_memory.py` calls the existing `SceneMemoryBank` |
| LDMR reviewing | `reviewing.py` reuses original LD-drop weights and scene sampling; `reviewing_host.py` applies them in the host runner |
| TR3D_OBJ replacement bank | Shared `ObjectMemoryBank(selection_strategy='tr3d_obj_replace')`; bounded streaming population |
| TR3D_OBJ insertion | `LDMRInsertObjects` applies class-uniform replay, floor placement and collision policy while retaining host superpoints |
| TR3D_OBJ pseudo policy | `pseudo.scannet_policy(..., policy_name='tr3d_obj')`, generation and merge |
| UniDet3D architecture/loss | Upstream model/encoder/criterion; `host.py` adapts dataset identity, incremental postprocessing and matcher edge cases |
| Incremental classifier growth | `checkpoints.expand_checkpoint` copies all detector tensors and classifier rows by class name, including `no_obj` |

Pinned source revisions and explicit deviations are in
[REFERENCE_DECISIONS.md](REFERENCE_DECISIONS.md). The legacy trainer uses the same
maintained banks; there is no second independent host selector.

## Host target and checkpoint requirements

Both named host recipes use `bbox_by_mask=False` and `target_by_distance=True`.
This preserves explicit GT/pseudo/crop box geometry at the loss. The native
ScanNet mask route would replace supplied boxes with instance-mask geometry;
it cannot represent box-only pseudo labels unchanged. Elastic deformation is
removed from these box-target pipelines. Genuine superpoint rows stay aligned
through crop insertion and augmentation. Hidden old/future GT masks must not
supply pseudo targets. The custom mask-target baseline remains available.

Interchange boxes are aligned ScanNet gravity-center AABBs. Host box storage
uses bottom centers; conversion happens once. Scene points and boxes undergo
compatible geometric augmentation. SUN legacy crops use local box axes and
rotate once by source yaw plus sampled yaw.

Stage transitions preserve sparse module version metadata and copy classifier
rows by class name, including the no-object row. The loader checks the actual
initialized model before training.

## Validation status

See the [current validation report](../../docs/unidet3d_validation.md).
Reference-source decisions and real-data supervision have been checked.
Full-schedule accuracy, exact interrupted-stage resume, distributed execution,
and fresh-machine installation remain unverified.
