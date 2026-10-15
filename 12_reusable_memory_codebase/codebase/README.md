# LDMR: Breaking the Model Forgetting Cycle in Long-Incremental 3D Object Detection

Research extension of the implementation accompanying

> **Breaking the Model Forgetting Cycle in Long-Incremental 3D Object Detection**
> Peisheng Qian, Jie Xu, Xulei Yang, Na Zhao
> *European Conference on Computer Vision (ECCV), 2026*

[Paper](https://arxiv.org/abs/2607.14560) &nbsp;·&nbsp; [Checkpoints](https://huggingface.co/Peisheng/LDMR) &nbsp;·&nbsp; [Dataset metadata](https://huggingface.co/datasets/Peisheng/LDMR-data)

---

## Scene and object memory extension

This checkout supports selectable scene-level and object-level replay in one
LDMR codebase. See the [memory-bank guide](docs/memory_banks.md) for setup,
configuration, support limits, and validation. Existing scene-named entry points
and legacy configurations remain supported.

For the teammate-aligned methods, start with the [four-path reference guide](docs/reference_methods.md):
TR3D and UniDet3D each support LDMR scene memory and a TR3D_OBJ-style object
method. It gives the named configs/recipes, full training exposure and explicit
host differences. Object Design-2/reviewing is a separate experimental variant.

For team reuse, start with the [researcher handoff](docs/researcher_handoff.md).
It links setup, examples, tests, performance evidence and the supported
UniDet3D port, including S5 pseudo supervision, Design-2 selection and online
reviewing, with explicit validation limits. Full-data accuracy of the current named reference recipes remains unmeasured.

```bash
python tools/train_incremental.py configs/memory/sunrgbd_s5_ldmr.py \
    --seed 200 --work-dir work_dirs/sun_scene
python tools/train_incremental.py configs/memory/sunrgbd_s5_tr3d_obj.py \
    --seed 200 --work-dir work_dirs/sun_object
```

These commands train the scene LDMR and object TR3D_OBJ-style methods.
The original LDMR results below describe scene memory. The four-path guide
provides the UniDet3D commands and identifies measured versus unmeasured recipes.

Incremental 3D object detection must learn new object classes while remembering
old ones. Prior methods rely on pseudo-labeling and hold up over one or two
increments, but collapse over **long** sequences: novel-class distribution shift
degrades the model on old classes, which corrupts the pseudo labels, which
degrades the model further — a self-reinforcing forgetting cycle.

**LDMR** (*Learning-Dynamics-driven Memory and Review*) breaks that cycle by
monitoring per-class detection quality at periodic training checkpoints and
turning those *learning dynamics* into two mechanisms:

- **Human-like intra-stage review** (Sec. 4.2) — each incremental stage is split
  into sub-stages; after every sub-stage the model is evaluated on the memory
  bank, and the next sub-stage over-samples whatever was most forgotten.
- **Scene-aware cross-stage memory evolution** (Sec. 4.3) — the memory bank is
  evolved between stages by jointly scoring scenes for **learnability** (which
  scenes the model can still improve on) and **diversity** (which scenes add
  information the bank lacks).

## Upstream LDMR reference results

LDMR with the TR3D backbone on SUN RGB-D (40 classes) and ScanNetV2 (35 classes),
under the 3-, 5- and 10-stage protocols. Final-stage mAP@0.25 over all seen
classes, as produced by the configs listed below.

| Dataset | Protocol | mAP@0.25 | Seed | Checkpoints |
|---|---|---|---|---|
| SUN RGB-D (40 cls) | 3-stage (20+10+10) | **29.12** | 200 | [🤗](https://huggingface.co/Peisheng/LDMR/tree/main/sunrgbd_3stage) |
| SUN RGB-D | 5-stage (8×5) | **25.10** | 200 | [🤗](https://huggingface.co/Peisheng/LDMR/tree/main/sunrgbd_5stage) |
| SUN RGB-D | 10-stage (4×10) | **19.38** | 200 | [🤗](https://huggingface.co/Peisheng/LDMR/tree/main/sunrgbd_10stage) |
| ScanNetV2 (35 cls) | 3-stage (15+10+10) | **37.81** | 201 | [🤗](https://huggingface.co/Peisheng/LDMR/tree/main/scannet_3stage) |
| ScanNetV2 | 5-stage (7×5) | **27.46** | 201 | [🤗](https://huggingface.co/Peisheng/LDMR/tree/main/scannet_5stage) |
| ScanNetV2 | 10-stage (3–4 cls/stage) | **17.64** | 200 | [🤗](https://huggingface.co/Peisheng/LDMR/tree/main/scannet_10stage) |

The upstream authors release the checkpoint of **every stage**, not just the final one, so each
point of the forgetting curve can be inspected.

## Installation

The legacy detector uses Python 3.9, PyTorch 1.12.1 + CUDA 11.3,
MinkowskiEngine 0.5.4, mmcv-full 1.6.0 and mmdet 2.24.1.
Follow the [legacy installation guide](docs/legacy_install.md) for the pinned
profile, native source build and CUDA verification. The October 14 fresh-install
trial passed native CUDA checks, dependency checks and 389 legacy tests. Use a new environment
and retain any working research environment until validation completes.

Native wheels are not included; the guide supplies the pinned source-build route. UniDet3D uses a
[separate environment](adapters/environment/README.md).

Run training and evaluation from the repository root, since inherited configs
resolve some paths relative to it. `activate_server_env.sh` contains defaults for
the original author's machine; inspect and override those paths before using it.

## Dataset preparation

Download **ScanNetV2** and **SUN RGB-D** from their original sources and accept
their respective terms of use.

- ScanNetV2: <http://www.scan-net.org/> (requires signing the Terms of Use)
- SUN RGB-D: <https://rgbd.cs.princeton.edu/>

Preparation has three steps: extract the raw scans into per-scene arrays, build
the annotation indices, then fix the ScanNet class ids.
[`data/scannet/README.md`](data/scannet/README.md) and
[`data/sunrgbd/README.md`](data/sunrgbd/README.md) cover the first step in full.

**ScanNetV2** — link the downloaded `scans/` (and `scans_test/`) into
`data/scannet/`, then:

```bash
cd data/scannet
python batch_load_scannet_data_40class.py     # -> scannet_instance_data_40class/
cd ../..
python tools/create_data.py scannet --root-path ./data/scannet \
    --out-dir ./data/scannet --extra-tag scannet --use-40-classes
```

`create_data.py` writes `annos['class']` as 0-based indices, whereas the class
mappings and `valid_cat_ids` throughout this codebase treat a class id as the
NYU40 id itself, which is 1-based. Shift them:

```bash
python tools/data_converter/scannet_correct_class_ids.py --data-root ./data/scannet
python tools/validate_scannet_alignment_contract.py    # optional sanity check
```

**SUN RGB-D** — unpack the official archives into `data/sunrgbd/OFFICIAL_SUNRGBD/`,
then run the three MATLAB extraction scripts and build the indices:

```bash
cd data/sunrgbd/matlab
matlab -nosplash -nodesktop -r 'extract_split;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v2;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v1;quit;'
cd ../../..
python tools/create_data.py sunrgbd --root-path ./data/sunrgbd \
    --out-dir ./data/sunrgbd --extra-tag sunrgbd --use-40-classes
```

The configs then expect:

```
data/scannet/scannet_infos_{train,val,test}_40class_corrected.pkl
data/sunrgbd/sunrgbd_infos_{train,val}_40class.pkl
```

If you would rather skip the index-building steps, the exact `.pkl` files we used
(~37 MB) are on the
[dataset repo](https://huggingface.co/datasets/Peisheng/LDMR-data). You still
need the extracted point clouds, so the per-scene extraction above is required
either way.

## Training

Each command runs the full incremental sequence from stage 1.

```bash
# SUN RGB-D, 10-stage (4 classes per stage)
CUDA_VISIBLE_DEVICES=0 python tools/train_incremental_scene.py \
    configs/incremental/sunrgbd/tr3d_dynamic_head_4x10_pseudo_memory_ld_design2_reviewing_6111111111.py \
    --work-dir ./incremental_logs/sunrgbd_10stage \
    --seed 200

# ScanNetV2, 5-stage (7 classes per stage)
CUDA_VISIBLE_DEVICES=0 python tools/train_incremental_scene.py \
    configs/incremental/scannet/tr3d_dynamic_head_s5_pseudo_memory_ld_design2_reviewing.py \
    --work-dir ./incremental_logs/scannet_5stage \
    --seed 201
```

To resume from a stage-1 checkpoint instead of retraining the base stage, pass
`--start-stage 2 --checkpoint-path <stage_01.pth>`. Design-2 memory selection then
also needs the stage-1 learning-dynamics scores, which the stage-1 run writes to
`<work-dir>/learning_dynamics/stage_1/learning_dynamics_design2_scores.json`:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_incremental_scene.py \
    configs/incremental/scannet/tr3d_dynamic_head_s5_pseudo_memory_ld_design2_reviewing.py \
    --work-dir ./incremental_logs/scannet_5stage_resumed \
    --start-stage 2 --end-stage 5 \
    --checkpoint-path <stage1_run>/checkpoints/stage_1/latest.pth \
    --seed 201 \
    --cfg-options \
      scene_memory_config.learning_dynamics_design2.stage1_scores_mode=precomputed \
      scene_memory_config.learning_dynamics_design2.stage1_scores_file=<stage1_run>/learning_dynamics/stage_1/learning_dynamics_design2_scores.json
```

## Evaluation

```bash
python tools/eval_incremental.py \
    configs/incremental/sunrgbd/tr3d_dynamic_head_4x10_pseudo_memory_ld_design2_reviewing_6111111111.py \
    checkpoints/sunrgbd_10stage/stage_10.pth \
    --eval mAP
```

Every released checkpoint carries its provenance:

```python
import torch
torch.load('stage_10.pth', map_location='cpu')['meta']['ldmr']
# {'protocol': 'sunrgbd_10stage', 'stage': 10, 'mAP@0.25': 0.1938, ...}
```

The incremental evaluator defaults to seen-class metrics, including early
SUN RGB-D stages. `--eval-class-scope all` explicitly includes future classes
for historical all-class curves; its early-stage averages are not comparable
to reported seen-class results. Both `tools/eval_incremental.py` and
`tools/test.py` accept `--metrics-out /path/to/new/report.json` with `--eval` to
save metrics, evaluated classes, scene count, input hashes and resolved config
details. Existing reports are never overwritten.

For published OpenMMLab TR3D checkpoints with the older head parameter names,
`tools/convert_tr3d_checkpoint.py CONFIG CHECKPOINT NEW_OUTPUT` performs an
explicit, strict inference conversion. It renames the known head parameters
and supplies an exact zero bias for the source's bias-free classifier. Other
schema mismatches are rejected. The converted checkpoint does not carry
optimizer state. Use the checkpoint's original classes and annotation version;
native TR3D scores are not incremental LDMR comparison scores.

## Configs

The six configs backing the results table:

| Protocol | Config |
|---|---|
| SUN RGB-D 3-stage | `configs/incremental/sunrgbd/tr3d_dynamic_head_20x10x10_pseudo_memory_ld_design2_reviewing_521.py` |
| SUN RGB-D 5-stage | `configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_pseudo_memory_ld_design2_reviewing_52211.py` |
| SUN RGB-D 10-stage | `configs/incremental/sunrgbd/tr3d_dynamic_head_4x10_pseudo_memory_ld_design2_reviewing_6111111111.py` |
| ScanNetV2 3-stage | `configs/incremental/scannet/tr3d_dynamic_head_s3_15_10_10_pseudo_memory_ld_design2_reviewing.py` |
| ScanNetV2 5-stage | `configs/incremental/scannet/tr3d_dynamic_head_s5_pseudo_memory_ld_design2_reviewing.py` |
| ScanNetV2 10-stage | `configs/incremental/scannet/tr3d_dynamic_head_s10_4444433333_pseudo_memory_ld_design2_reviewing.py` |

Memory-only ablations (no pseudo labels) and pure-finetuning baselines live
alongside them in the same directories.

### Key hyperparameters

LDMR is configured through two blocks. Memory evolution:

```python
scene_memory_config = dict(
    memory_budget_ratio=0.1,                       # |M| = 10% of the training set
    selection_strategy='learning_dynamics_design2',
    learning_dynamics_design2=dict(
        q_metric='recall',                         # per-class quality signal, Eq. 3
        supply_scaling_mode='cap_log1p',           # log(1+n) count weight, Eq. 9
        redundancy_lambda=0.5,                     # lambda in Eq. 14
    ),
)
```

and intra-stage review:

```python
reviewing = dict(
    enabled=True,
    review_fractions=[0.2, 0.4, 0.6, 0.8],         # I-1 checkpoints -> I sub-stages
    weight_policy=dict(type='ld_drop', eta=3),     # review emphasis eta, Eq. 5
)
```

Values differ per protocol: the SUN RGB-D 3-stage config uses 6 review
checkpoints, and the ScanNetV2 configs use `q_metric='f1'` with `lambda=0.3` and
`eta=5.0`. Each config's docstring records its exact combination and the mAP it
produces.

## Citation

```bibtex
@inproceedings{qian2026ldmr,
  title     = {Breaking the Model Forgetting Cycle in Long-Incremental 3D Object Detection},
  author    = {Qian, Peisheng and Xu, Jie and Yang, Xulei and Zhao, Na},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

Released under [CC BY-NC 4.0](LICENSE), inherited verbatim from
[TR3D](https://github.com/filaPro/tr3d). Non-commercial use only.

This is a derivative work: source files under `mmdet3d/` that carry an
`OpenMMLab` copyright header remain governed by mmdetection3d's Apache-2.0
license. See [NOTICE](NOTICE) for upstream attribution and a statement of
changes.

## Acknowledgements

Built on [TR3D](https://github.com/filaPro/tr3d) and
[mmdetection3d](https://github.com/open-mmlab/mmdetection3d), and uses
[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine). Our incremental
protocols follow [SDCoT](https://github.com/Na-Z/SDCoT). We thank the authors of
ScanNet and SUN RGB-D for the datasets.

(The original `SamsungLabs/tr3d` repository has since been taken down; TR3D now
lives at `filaPro/tr3d` and inside mmdetection3d's `projects/TR3D`.)

This research is supported by the Agency for Science, Technology and Research
(A*STAR) under its MTC Programmatic Funds (Grant No. M23L7b0021), and the
Ministry of Education, Singapore, under its MOE Academic Research Fund Tier 2
(MOE-T2EP20124-0013).
