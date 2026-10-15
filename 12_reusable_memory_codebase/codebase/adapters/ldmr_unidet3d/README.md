# LDMR scene and object memory for UniDet3D

The adapter supports the original ScanNet-35 five-stage frequency-order protocol.
It retains the native UniDet3D detector and optimizer while integrating LDMR
memory selection, pseudo supervision, and reviewing. Object replay uses aligned
point, semantic, instance and superpoint crops.

Use [reference methods](../../docs/reference_methods.md) for complete commands:
`--memory-mode scene --recipe ldmr` selects the scene method;
`--memory-mode object --recipe tr3d_obj` selects the reference object extension.
`--recipe custom` exposes separate research variants, including object Design-2
and reviewing. The [method contract](METHOD_PARITY.md) and
[reference decisions](REFERENCE_DECISIONS.md) document each boundary.

## Environment and data

Install the [separate host environment](../environment/README.md). Supply the
pinned upstream UniDet3D checkout, a legacy Python interpreter, and prepared
ScanNet metadata with genuine superpoints. The host must not import the legacy
repository root through PYTHONPATH.

The original 35 class names, NYU40 IDs and stage order are preserved. A native
18-class metadata file is not a substitute. `prepare_stage_infos.py` validates
metadata and point/mask lengths; `prepare_superpoints.py` can derive groups from
matching mesh geometry. `import_superpoints.py` joins verified prepared groups
only when XYZRGB rows match exactly, as described below.

## Entry points

| Tool | Purpose |
|---|---|
| `run_sequence.py` | Manage complete stages, pseudo labels, training, LD collection and bank export |
| `train_stage.py` | Train a single stage in a new work directory |
| `generate_pseudo.py` | Generate old-class boxes from the frozen previous-stage teacher |
| `collect_ld.py` | Collect class/scene quality measurements for selection and reviewing |
| `export_memory.py` | Export scene or rich object banks through the legacy bank implementation |
| `evaluate_stage.py` | Validate and evaluate a saved stage checkpoint |
| `audit_checkpoints.py` | Check source manifests, stage artifacts and loaded-model receipts |
| `audit_metrics.py` | Check validation coverage and finite per-class metrics |

All tools are under `adapters/tools/`; each provides `--help`.
Named recipes use explicit box/distance targets, preserving pseudo and pasted
box geometry through the native loss. Training inputs hide future-class targets.
Classifier expansion copies learned tensors and class rows, including no-object.

## Continuing after a completed stage


`run_sequence.py --continue-from /path/to/previous/run` reuses a contiguous prefix
of completed stages from a **failed or complete** sequence, then trains the
remaining stages in a new `--work-dir`. Repeat the original command's runtime,
input paths, mode, budgets, seed, batch size, workers and epochs; change only the
output directory and, when extending a bounded run, `--last-stage`. For example,
add these arguments to the otherwise identical command after a stage-2 run:

```bash
--continue-from /path/to/stage2_run --work-dir /path/to/new_continuation --last-stage 5
```

Continuation requires matching source paths and hashes, identical input manifests
and every referenced point/mask file, matching settings, and unchanged completed
checkpoints/configs/bank artifacts. It rejects a run still marked running, an empty
or incomplete prefix, and older runs without the new integrity manifests. A
hard-killed supervisor can remain marked running; this is not an automatic orphan
process recovery mechanism. Do not edit its status to bypass the guard.

For new POSIX launches, send SIGTERM to the `run_sequence.py` PID or use Ctrl-C
for a foreground sequence. The supervisor owns each command in a separate
process group and records its PID in `status.json`. It stops that group,
including training workers and review collectors, before recording failure;
after five seconds it kills remaining group members. Repeated interruption
does not bypass cleanup. Other jobs are unaffected. SIGKILL and machine failure
cannot run cleanup and remain outside this guarantee. Previously frozen source
snapshots retain their original shutdown behavior.

Completed stages stay at their original paths and are referenced through
`status.json`'s `inherited_stages`; they are not copied or overwritten. Preserve
those directories with the continuation. New stages use the normal classifier
expansion and fresh optimizer/schedule. A failed partial stage is rerun from its
start. This does not restore a partial stage's optimizer, sampler or RNG state,
and does not promise bitwise reproducibility across CUDA environments.

Initial preflight hashes all referenced inputs using bounded read buffers, adding
one sequential read of the data. This is an intentional integrity cost. Keep
data read-only during training and retain the pinned runtime: content hashes
cover source and data, not modifications to installed dependencies or drivers.
Source checks run both before and after each child. A stage is marked completed
only after its required artifacts have been hashed. The checkpoint auditor
resolves inherited paths and verifies these hashes and recorded input bytes.

Detailed verification scope, reproduction commands and benchmark limitations:
[validation report](../../docs/unidet3d_validation.md).


## Importing existing genuine superpoints


`adapters/tools/import_superpoints.py` supports prepared reference data with
`points/<scene>.bin` and `super_points/<scene>.bin`. It requires exact equality
of every float32 XYZRGB row against the legacy point file before attaching any
group IDs. It preserves the legacy boxes, semantic masks, instance masks and
original ScanNet-35 class mapping. Reference detection labels are never used.
Every source scene must pass; missing scenes and altered row order abort the
whole import. The output is published as a complete new directory.

```bash
/path/to/python adapters/tools/import_superpoints.py \
  --input /path/to/legacy/scannet_infos_train.pkl \
  --data-root /path/to/legacy \
  --reference-root /path/to/verified/prepared/scannet \
  --reference-provenance /path/to/archive_identity.json \
  --box-label-space nyu40_zero_based \
  --output-dir /path/to/new/host_train
```

Use the declared box-label numbering of your source metadata. Supply a JSON
provenance record identifying the reference revision and verified archive hash;
the importer records it together with per-file hashes and exact-match evidence.
The tool verifies alignment, not the origin of arbitrary supplied group IDs.
Use a documented genuine-superpoint source, such as the archive linked by the
[UniDet3D authors](https://github.com/filapro/unidet3d#data-preprocessing).


## Evaluating a saved host checkpoint


Use the isolated host interpreter and the `resolved_stage.py` saved by training:

```bash
/path/to/host/bin/python adapters/tools/evaluate_stage.py \
  --upstream-root /path/to/pinned/UniDet3D \
  --config /path/to/stage_5/training/resolved_stage.py \
  --checkpoint /path/to/stage_5/training/epoch_12.pth \
  --work-dir /path/to/new/evaluation --seed 200
```

The command checks checkpoint protocol/stage, exact tensor keys/shapes/dtypes,
finite weights and validation-manifest visibility before running inference.
It uses the trained model's resolved settings and all seen classes. No training
bank or training pipeline is constructed. `--val-info` can select a replacement
validation manifest with the same stage/protocol; the report hashes that input.
`metrics.json` preserves the upstream dataset-nested metrics, class denominator,
scene count, seed and checkpoint/config/annotation hashes. Undefined metrics are
JSON `null` with their paths listed separately; they must not be treated as zero.
An existing evaluation work directory is rejected.
