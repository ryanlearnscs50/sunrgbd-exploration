# LDMR scene and object memory

This repository contains one LDMR implementation with selectable scene-level
and object-level replay. TR3D is the detector already included in the codebase.
The public entry point is `tools/train_incremental.py`; the historical
`tools/train_incremental_scene.py` entry point remains supported.
For the requested LDMR and TR3D_OBJ-style methods on both detectors, start with
the [four-path reference guide](reference_methods.md). The random-mode examples
below remain useful controls, but omit the full scene method's pseudo/LD/reviewing.

## Installation and working directory

Follow the [repository installation instructions](../README.md#installation).
The existing tested stack is Python 3.9, PyTorch 1.12.1 + CUDA 11.3,
MinkowskiEngine 0.5.4, mmcv-full 1.6.0, and mmdet 2.24.1. This change does not
upgrade those dependencies. A separate legacy installation was validated on this
machine; the [pinned install guide](legacy_install.md) records that environment.
A fresh-machine installation remains unverified.
Activate your installed environment and run commands
from the repository root: inherited configs resolve some paths relative to it.

Prepare the dataset using [SUN RGB-D](../data/sunrgbd/README.md) or
[ScanNet](../data/scannet/README.md) instructions and check the dataset paths in
the inherited configs. Configuration preflight does not verify dataset files.
Use your own writable work directory; shared source data should remain read-only.

## Select a memory mode

The examples below retain the existing five-stage random-memory settings, with
pseudo labels and reviewing disabled. They are baseline examples, not a claim
that random replay is full LDMR or the best-performing configuration.

```bash
# Preflight only: no training, dataset construction, or output run directory.
python tools/train_incremental.py configs/memory/sunrgbd_s5_random.py \
  --memory-mode scene --validate-memory-config
python tools/train_incremental.py configs/memory/sunrgbd_s5_random.py \
  --memory-mode object --validate-memory-config

# Train from scratch with the selected replay method.
python tools/train_incremental.py configs/memory/sunrgbd_s5_random.py \
  --memory-mode scene --seed 200 --work-dir work_dirs/s5_scene
python tools/train_incremental.py configs/memory/sunrgbd_s5_random.py \
  --memory-mode object --seed 200 --work-dir work_dirs/s5_object
```

Use `configs/memory/scannet_s5_random.py` for the corresponding ScanNet-35
example. `--memory-mode none` disables both memory modes; it does not change
pseudo-label, reviewing, or optimization settings. The two examples disable
reviewing explicitly so this no-memory baseline is well-defined.

Set `memory_mode = 'scene'`, `'object'`, or `'none'` in a config, or override it
with `--memory-mode`. Precedence is:

1. `--memory-mode`.
2. Top-level `memory_mode` after applying `--cfg-options`.
3. Legacy `use_scene_memory` / `use_object_memory` inference.

Explicit modes set the two legacy booleans consistently. They retain both bank
configurations and require a mapping for the selected bank. The switch never
invents another mode's budget or selector: an older object-only config with
`scene_memory_config=None` cannot be switched to scenes until scene settings
are supplied. Legacy configs without `memory_mode` retain their previous mode
defaults and still reject simultaneous scene/object replay.

`--validate-memory-config` checks mode resolution, selected bank constructors,
object insertion options, learning-dynamics schema, and the known unsupported
ScanNet object Design-2 combination. It prints a JSON summary and exits before
distributed setup, dataset/model construction, checkpoint loading, or run-directory
creation. A precomputed scene selector may read its configured score files in
the constructor. The check does **not** establish dataset availability, point/box
alignment, stage-specific quota correctness, resume equivalence, full feature
compatibility, or detection accuracy. It imports the normal runtime dependencies.

## What each mode stores and replays

| Property | Scene | Object |
|---|---|---|
| Stored unit | Scene metadata and stage-filtered annotation snapshot | Local-coordinate object point crop and source/box metadata |
| Replay operation | Add/merge whole scene examples | Paste old-class crops into natural scenes before sampling/augmentation |
| Source points needed during replay | Yes | No, for ordinary crop replay |
| Source data needed for LDMR reviewing | Yes | Yes: current object reviewing evaluates original source scenes |
| Budget units | Scene/stage entries, subject to configured quota/deduplication policy | Crops per class plus total crop cap |
| Saved artifacts | Inspection JSON plus versioned global-budget bank checkpoint | Complete crop pickle and metadata JSON |

Equal numbers of scene and object entries do not imply equal bytes, point count,
replay exposure, or compute. Changing mode preserves the shared example protocol,
but it necessarily changes the replay operation and uses separately specified
budgets. Do not describe these examples as storage-matched comparisons.

The SUN RGB-D example retains a nominal 10% scene budget (528 entries) or
20 crops/class with an 800-crop global cap. ScanNet uses a nominal 10% scene
budget (120 entries) or 20 crops/class with a 700-crop cap. Actual occupancy may
be lower when eligible examples are scarce. Insertion attempt counts are not
measurements of successful pastes or points surviving downstream sampling.

## Learning dynamics and capability limits

Existing method configurations remain in `configs/incremental/`. For example,
SUN RGB-D object Design-2 plus reviewing is configured by
`configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_object_memory_ld_design2_reviewing_52211.py`.
It can be passed to the new entry point without adding `--memory-mode`.
Selection policy, reviewing, pseudo supervision, and memory mode are separate
choices. Do not enable or disable one implicitly when comparing the others.

| Path | Current implementation status |
|---|---|
| SUN RGB-D scene random and LD Design-1/2 | Existing integration and focused regression tests |
| SUN RGB-D object random / largest-point-count / Design-2 | Existing population paths; crop, selection, and wiring tests |
| SUN RGB-D object source-scene reviewing | Existing implementation; requires original scene metadata |
| ScanNet scene random and LD paths | Existing integration and selected regression tests |
| ScanNet object random | Existing population/insertion path; broader end-to-end validation remains |
| ScanNet object largest-point-count | Bank selector exists; dataset-path parity is not established here |
| ScanNet object Design-2 | Unsupported; rejected before starting training |
| UniDet3D ScanNet-35 S5 | Scene/object adapters implemented; pseudo, Design-2, reviewing and native loss/backward checked |

This table describes implementation and test coverage, not a guarantee of every
configuration combination. The [current validation report](unidet3d_validation.md)
records additional requirements.

## State, resume, and current limitations

Object state is saved with the completed stage. Existing resume uses
`--checkpoint-path`, `--start-stage`, and `--object-memory-path`. The object
bank must belong to the immediately preceding stage and have the expected
processed-class coverage; stage-3+ object resume requires its bank file. A class
with no eligible retained crops is distinct from an unprocessed class in new
states. Switching
mode starts a different method; a saved scene bank is not interchangeable with
an object bank. The mode flag does not convert bank artifacts.

Global-budget scene banks now write a `.pkl` checkpoint alongside the existing
inspection JSON whenever the trainer saves its bank. Resume with
`--scene-memory-path /path/to/scene_memory_bank_stage_N.pkl`,
`--checkpoint-path /path/to/stage_N_model.pth`, and `--start-stage N+1`.
Scene resume at stage 3+ requires this checkpoint; stage 2 can still reconstruct
stage 1 using the established LDMR path. Legacy per-class scene mode has no new
checkpoint support. JSON reports are not accepted as checkpoints.

Scene checkpoints check version, stage, protocol context, semantic constructor
settings, snapshot consistency, and capacity. They preserve selected annotation
arrays and scores; unselected candidate snapshots are excluded. Tests verify
identical replay and next-stage random/Design-2 selection after restoration.
Diagnostic counters are rebuilt or reset rather than promised as historical logs.

New object states use format **4**, retaining validated format **3** reads.
The authoritative pickle contains constructor configuration, completed stage,
processed classes, and full crops. Loading validates crop shape, finite values,
coordinate frame, class/count metadata, source stages, and budgets. The adjacent
JSON omits point arrays and remains an inspection sidecar. Each file is replaced
atomically; the pair is not a single transaction, so load the pickle for replay.
Load only trusted local pickle artifacts. Older/unknown formats require an
explicit migration; do not relabel a file's version to bypass validation.

The trainer also rejects changed object budget/selector/crop semantics on resume.
The saved bank's selection seed remains authoritative, while the run's training
seed may differ (for continuations sharing one initial bank). Source-cache size
can change independently because it does not alter selection results.

These are **stage-boundary bank checkpoints**, not full training snapshots.
Exact optimizer/RNG/worker/pseudo-label/review-history continuation has not been
established. Keep prior model checkpoints and learning-dynamics/metric artifacts
available when the configured method needs them. The protocol context does not
hash the underlying dataset files. Source-free object reviewing is not implemented.

## Efficiency and measurement

Object population uses a temporary LRU source-point cache controlled by
`object_memory_config.source_cache_max_bytes` (default 64 MiB; 0 disables it).
It reuses source clouds and their floor estimate within one population pass,
without changing candidate order, random draws, or selected crop contents.
SUN RGB-D avoids repeated per-object file reads; ScanNet replaces its previous
unbounded stage-wide cloud cache. A smaller cache may reread evicted files.
`last_population_io` reports loads, cache hits, and peak cached point bytes.
This bound excludes transient oversized source arrays, crop candidates, Python
metadata, and the retained bank; it is not a process-RAM limit.

Pruning retains removal metadata without point arrays. `active_point_bytes` and
`retained_removed_point_bytes` expose point payload accounting; `storage_mb`
remains an alias for active point MiB, not disk size or process RSS. Unmeasured
cache hit rate is `None`. The legacy overflow handler reports actual balanced
bank pruning instead of hypothetical reductions.

See [method and performance notes](memory_methods_and_results.md) for current
detection evidence and the bounded component checks. No new accuracy claim is
implied by these engineering changes.

## Validation and extension points

Run the repository regression suite from the root:

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests
```

The mode tests check legacy defaults, explicit overrides, invalid combinations,
baseline config equivalence, and both CLI entry points. They exercise preflight
in subprocesses and verify that it does not create a training work directory.

| Responsibility | Location |
|---|---|
| Public mode selection | `tools/incremental_scene/memory_config.py` |
| Stage orchestration / wiring | `tools/train_incremental_scene.py` |
| Scene storage and selection | `mmdet3d/datasets/scene_memory_bank.py` |
| Object storage and selection | `mmdet3d/datasets/object_memory_bank.py` |
| Dataset population and label filtering | `incremental_sunrgbd.py`, `incremental_scannet.py` in that directory |
| Crop placement | `mmdet3d/datasets/pipelines/exemplar_insertion.py` |

Keep future host adapters separate from selection/storage logic. Preserve class
mapping, coordinate frame, stage visibility, budget, and source provenance when
adding methods. Test through dataset population and pipeline wiring as well as
isolated selectors. Performance documentation will distinguish upstream-reported
results, local measurements, and configurations without empirical validation.


### Object Design-2 selection cost

When crop validity is already known, the bank now computes only the required
prefix of the greedy Design-2 ranking. Previously it ranked every candidate and
then discarded the suffix after the per-class quota, making full-data population
unnecessarily expensive. Full-pool validation, normalization, seeded tie values,
source-diversity preference and selected-object metadata are unchanged. The
public ranker defaults to a complete ranking; its optional `limit` is an exact
prefix, not a smaller normalization pool. Legacy SUN population uses the lazy
ranking iterator and stops only after enough valid crops have been accepted.
Rejected crops remain part of the ranking history, exactly as in the original
complete-ranking-then-filtering procedure.

Independent comparisons with the archived original ranker preserve selected IDs,
metadata and crops exactly. The final full-data stage-1 export produced all 140
crops in 160 seconds with its 64 MiB source cache bounded. All five exported
files, including bank state, crop payloads, selection and provenance, are
byte-identical to the original full-data export. Evidence is archived in
`week5_audit/object_prefix_20261014/full_data_parity.json` at workspace level. This removes unused ranking work and does not reduce the
transient memory needed to validate candidate crops or establish a training
throughput claim. The accuracy runs retain their original frozen implementation.
