# Reference decisions for the UniDet3D port

Current named recipes and their reference boundaries are in
[reference_methods.md](../../docs/reference_methods.md). This table describes
the current named methods; custom weighted/height-preserving replay remains a
separate research policy.

The maintained implementation is LDMR, extended with selectable scene/object
memory. The pinned TR3D_OBJ source snapshot is a reference, not a
second maintained bank or a claim of superior performance.

The [method contract](METHOD_PARITY.md) maps reference responsibilities to code
and explains the explicit-box target route used by both named host methods.

| Reference behavior | Decision | Reason / implementation |
|---|---|---|
| LDMR ScanNet-35 frequency order and 3/5/10-stage splits | Preserve | `protocol.py` reads the existing mapping file; no native ScanNet-18 substitution |
| LDMR natural current-class and evaluation seen-class scene eligibility | Preserve | `data.stage_records` filters scenes before host construction |
| LDMR scene `merge_labels` with unique scene IDs | Preserve | `scene.merge_replay_records` merges current annotations and saved snapshot visibility, leaving intervening/future classes hidden |
| LDMR random/largest crop selection and bounded bank | Reuse | `objects.selected_payloads` joins richer masks to exactly the retained LDMR objects; it does not select another bank |
| TR3D_OBJ per-class budget, explicit source identity and crop-based replay | Follow structure | Rich payload retains scene/object/instance/stage identity and selected geometry |
| TR3D_OBJ unconditional replacement after capacity, described as reservoir sampling | Preserve in named object recipe | Random-slot replacement favors later candidates. Streaming implementation keeps bounded crops; per-class RNG differs from the reference's shared RNG |
| TR3D_OBJ box-contained context points | Preserve points and explicit box supervision | Rich masks retain source-instance membership for custom mask targets; named box targets use geometry/distance support |
| TR3D_OBJ 1 cm absolute crop-boundary tolerance | Preserve in both named object profiles | Bank and rich export share float64 crop membership and `crop_tolerance=.01`; old tolerance is checked on resume |
| Real SUN zero-height GT annotations | Preserve scene evaluation counts; reject object crops | Matches original LDMR FN accounting without creating zero-volume pasted targets |
| TR3D_OBJ floor-only placement and gravity/bottom-center ambiguity | Preserve floor placement, make origins explicit | Named recipe puts the box bottom on the destination minimum Z; convert gravity-center math to host bottom-center storage once. Custom mode can preserve source height |
| TR3D_OBJ random yaw | Not in this ScanNet adapter | First port uses ScanNet axis-aligned boxes; yaw-aware datasets need a separately verified adapter |
| TR3D_OBJ class-uniform draws with replacement | Preserve in named object recipe | Three draws, including repeated crops, with scene probability .7. NumPy worker RNG means cross-host sampled sequences need not match |
| TR3D_OBJ loss-logit masking | Do not copy | The reference branch can suppress pseudo positives and identifies old classes from pasted samples only. An explicit, tested host loss-weighting design is still required |
| TR3D_OBJ pseudo-label controls | Preserve comparison requirement | Named scene recipe includes pseudo/Design-2/reviewing; named object recipe uses reference .5 confidence pseudo supervision without object reviewing. Full-data accuracy of the aligned recipes remains unmeasured |
| UniDet3D semantic/instance/superpoint alignment | Required extension | Rich crops retain aligned masks; insertion allocates separate IDs; stage targets regenerate after augmentation |
| UniDet3D internal alphabetical classifier rows | Remap by name | Copy all learned class rows and `no_obj`, retaining initialization for new rows; reject unknown protocol checkpoints |
| UniDet3D dataset inferred from path name | Make explicit | Single-dataset model and metric adapters identify ScanNet without renaming shared data |

Sources inspected: LDMR checkout upstream HEAD
`ab67f3d7a3e6e98c6f66182832bb37e8656cceaa` plus maintained local changes;
TR3D_OBJ selected source snapshot at
`cd667a180c3cbf4eae29cf779ea907592400266f`; UniDet3D at
`940a730a09711b0bf266fd972504da29a83b91f6`.
