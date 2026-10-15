"""
Proxy training utilities for RL-driven incremental experiments.

This module intentionally keeps the interface light and highly logged so that
you can probe RL allocation behavior before wiring it into the full training
pipeline.

Design goals:
    - Do not repeat the dataset 15x per epoch; default to 1x repetition
      for proxy runs (data.train.times = 1).
    - Provide a detailed log of what the RL policy requested:
        * Per-class metrics before the stage
        * Derived state features (mAP drop, ratios)
        * Per-class allocation targets T_c
        * Selected memory scenes and their per-class counts
    - Keep actual detector training pluggable via a user-supplied callback.

By default, ``default_proxy_train_fn`` is a NO-OP stub that only logs RL
decisions and returns the input metrics unchanged. This lets you validate the
end-to-end RL wiring before you connect it to a real incremental training
routine.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import math
import os
import copy
import json

from mmdet3d.utils import get_root_logger

from .structures import PerClassMetrics, SceneDescriptor, build_metrics_from_raw
from .state_builder import build_state


@dataclass
class IncrementalProxyTrainConfig:
    """Configuration wrapper passed to proxy training functions.

    Attributes:
        repeat_times: Dataset repetition factor. For RL proxy runs this should
            normally be 1 to avoid the heavy 15x RepeatDataset used in full
            training.
        extra_cfg: Optional free-form object containing any additional
            configuration you want to use inside a custom proxy_train_fn
            (e.g., mmcv.Config instances, stage definitions, work dir roots).
        proxy_train_fn: Optional callable implementing the actual proxy
            training logic. If None, ``default_proxy_train_fn`` will be used.
    """

    repeat_times: int = 1
    extra_cfg: Optional[Any] = None
    proxy_train_fn: Optional[Any] = None


def _summarize_metrics(metrics_k: Dict[int, PerClassMetrics]) -> Dict[str, Any]:
    """Compute simple summary statistics for logging."""
    if not metrics_k:
        return {"num_classes": 0}

    drops = []
    curr_values = []
    prev_values = []

    for cid, m in metrics_k.items():
        curr = float(m.map_curr)
        prev = float(m.map_prev)
        drop = max(0.0, prev - curr)
        curr_values.append(curr)
        prev_values.append(prev)
        drops.append(drop)

    drops_arr = np.asarray(drops, dtype=np.float32)
    curr_arr = np.asarray(curr_values, dtype=np.float32)
    prev_arr = np.asarray(prev_values, dtype=np.float32)

    return {
        "num_classes": len(metrics_k),
        "avg_curr_map": float(curr_arr.mean()),
        "avg_prev_map": float(prev_arr.mean()),
        "avg_drop": float(drops_arr.mean()),
        "max_drop": float(drops_arr.max()),
        "min_drop": float(drops_arr.min()),
    }


def _summarize_allocations(T_c: Mapping[int, int]) -> Dict[str, Any]:
    """Summarize per-class allocation targets."""
    if not T_c:
        return {"num_classes": 0, "total_slots": 0}

    counts = np.asarray(list(T_c.values()), dtype=np.float32)
    return {
        "num_classes": len(T_c),
        "total_slots": int(counts.sum()),
        "min_slots": int(counts.min()),
        "max_slots": int(counts.max()),
        "mean_slots": float(counts.mean()),
    }


def _summarize_scene_pool(scene_pool: Sequence[SceneDescriptor]) -> Dict[str, Any]:
    """Summarize candidate scene pool."""
    if not scene_pool:
        return {"num_scenes": 0, "total_objects": 0}

    total_objects = 0
    per_class_counts: Dict[int, int] = {}
    for desc in scene_pool:
        for cid, cnt in desc.class_counts.items():
            total_objects += int(cnt)
            per_class_counts[cid] = per_class_counts.get(cid, 0) + int(cnt)

    counts_arr = np.asarray(list(per_class_counts.values()), dtype=np.float32)
    return {
        "num_scenes": len(scene_pool),
        "total_objects": int(total_objects),
        "num_classes_with_objects": len(per_class_counts),
        "avg_objects_per_class": float(counts_arr.mean()) if counts_arr.size > 0 else 0.0,
    }


def _aggregate_counts_from_scene_pool(
    scene_pool: Sequence[SceneDescriptor],
) -> Dict[int, int]:
    """Aggregate per-class counts across the entire scene pool."""
    counts: Dict[int, int] = {}
    for desc in scene_pool:
        for cid, cnt in desc.class_counts.items():
            counts[cid] = counts.get(cid, 0) + int(cnt)
    return counts


def _load_scene_snapshots_from_json(json_path: str) -> Dict[str, Dict[str, Any]]:
    """Load scene_snapshots structure from a saved memory bank JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("scene_snapshots", {})


def _rebuild_scene_memory_bank_from_json(
    json_path: str,
    selected_scene_ids: List[str],
    data_info_map: Optional[Dict[str, Any]] = None,
):
    """Rebuild a SceneMemoryBank instance from a saved Stage-1 JSON and a subset of scenes."""
    from mmdet3d.datasets.scene_memory_bank import SceneMemoryBank

    snapshots = _load_scene_snapshots_from_json(json_path)
    bank = SceneMemoryBank(
        memory_budget_ratio=1.0,
        max_memory_scenes=len(selected_scene_ids) + 10,
        selection_strategy="forced",
        debug_mode=False,
    )

    # Iterate selected scenes, pick stage "1" snapshot if present
    for scene_id in selected_scene_ids:
        stage_dict = snapshots.get(scene_id, {})
        snap = stage_dict.get("1") or next(iter(stage_dict.values()), None)
        if snap is None:
            continue
        snapshot = copy.deepcopy(snap)
        # If data_info is missing, try to fill from map; otherwise skip
        if "data_info" not in snapshot:
            if data_info_map and scene_id in data_info_map:
                snapshot["data_info"] = copy.deepcopy(data_info_map[scene_id])
            else:
                continue
        save_stage = int(snapshot.get("save_stage", 1))
        try:
            bank._add_scene_to_memory(scene_id, save_stage, snapshot, importance=1.0)
        except Exception:
            # Fallback: skip problematic scenes
            continue

    return bank, snapshots


def load_metrics_from_log(
    log_path: str,
    model_idx_to_name: Mapping[int, str],
    num_classes: int,
    ap_key_suffix: str = "AP_0.25",
    default_prev_equal_curr: bool = True,
) -> Dict[int, PerClassMetrics]:
    """Load per-class metrics from a stage log.json (last val entry).

    Args:
        log_path: Path to the stage log JSON file (MMCV format).
        model_idx_to_name: Mapping from class id -> class name (used in keys).
        num_classes: Total number of classes to populate.
        ap_key_suffix: Which AP metric to read (default: AP@0.25).
        default_prev_equal_curr: If True, sets map_prev = map_curr for the
            initial metrics to avoid artificial drops at the next stage.

    Returns:
        Dict[int, PerClassMetrics]
    """
    logger = get_root_logger(log_level="INFO")
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning("Failed to read log file %s: %s", log_path, e)
        return {cid: PerClassMetrics(0.0, 0.0, 0, 0) for cid in range(num_classes)}

    last_val = None
    for line in reversed(lines):
        try:
            rec = json.loads(line.strip())
        except Exception:
            continue
        if isinstance(rec, dict) and rec.get("mode") == "val" and "mAP_0.25" in rec:
            last_val = rec
            break

    if last_val is None:
        logger.warning("No validation record found in %s", log_path)
        return {cid: PerClassMetrics(0.0, 0.0, 0, 0) for cid in range(num_classes)}

    map_curr_per_class: Dict[int, float] = {}
    for cid in range(num_classes):
        name = model_idx_to_name.get(cid, f"class_{cid}")
        key = f"{name}_{ap_key_suffix}"
        map_curr_per_class[cid] = float(last_val.get(key, 0.0))

    metrics = {}
    for cid in range(num_classes):
        curr = map_curr_per_class.get(cid, 0.0)
        prev = curr if default_prev_equal_curr else 0.0
        metrics[cid] = PerClassMetrics(
            map_curr=curr,
            map_prev=prev,
            num_seen=0,
            num_mem=0,
        )

    logger.info(
        "Loaded per-class metrics from %s (mean AP=%.4f)",
        log_path,
        float(np.mean(list(map_curr_per_class.values())) if map_curr_per_class else 0.0),
    )
    return metrics


def default_proxy_train_fn(
    detector_ckpt_path: str,
    stage_data: Any,
    memory_scenes: List[SceneDescriptor],
    train_cfg: IncrementalProxyTrainConfig,
    metrics_k: Dict[int, PerClassMetrics],
    T_c: Dict[int, int],
) -> Tuple[Dict[int, PerClassMetrics], Dict[str, Any]]:
    """NO-OP proxy training function with detailed logging.

    This implementation does NOT run any detector training. It only logs:
      - Per-class metrics before the stage
      - Derived state statistics
      - Allocation targets T_c
      - Scene pool and selected memory scenes
    and then returns the input metrics unchanged.

    This is useful for:
      - Validating RL wiring end-to-end
      - Inspecting whether the policy behaves sensibly
      - Debugging state construction and scene selection

    Once you are ready to connect real proxy training, you can replace this
    function with a custom implementation and assign it to
    ``IncrementalProxyTrainConfig.proxy_train_fn``.
    """
    logger = get_root_logger(log_level="INFO")

    logger.info("=== RL Proxy Training (NO-OP) ===")
    logger.info(f"Detector checkpoint: {detector_ckpt_path}")
    if hasattr(stage_data, "stage_id"):
        logger.info(f"Stage id: {stage_data.stage_id}")
    elif isinstance(stage_data, Mapping) and "stage_id" in stage_data:
        logger.info(f"Stage id: {stage_data['stage_id']}")

    # Summarize incoming metrics and state
    metrics_summary = _summarize_metrics(metrics_k)
    logger.info(f"Per-class metrics summary before stage: {metrics_summary}")

    state_vec = build_state(metrics_k)
    logger.info(f"State vector length: {state_vec.size}")
    logger.info(
        f"State stats: mean={float(state_vec.mean()):.4f}, "
        f"std={float(state_vec.std()):.4f}, "
        f"min={float(state_vec.min()):.4f}, "
        f"max={float(state_vec.max()):.4f}"
    )

    # Allocation summary
    alloc_summary = _summarize_allocations(T_c)
    logger.info(f"Allocation targets summary: {alloc_summary}")

    # Scene pool / memory selections
    pool_summary = _summarize_scene_pool(memory_scenes)
    logger.info(f"Selected memory scenes summary: {pool_summary}")
    logger.info(
        "First 10 memory scenes: %s",
        [s.scene_id for s in memory_scenes[:10]],
    )

    # Expose repeat_times explicitly for RL proxy runs
    repeat_times = getattr(train_cfg, "repeat_times", 1)
    logger.info(f"Proxy training repeat_times (data.train.times): {repeat_times}")

    # NO-OP: use metrics_k as new_metrics for now
    new_metrics = metrics_k

    # Construct train_logs with enough detail for offline analysis
    train_logs: Dict[str, Any] = {
        "metrics_before_summary": metrics_summary,
        "alloc_summary": alloc_summary,
            "selected_scene_ids": [s.scene_id for s in memory_scenes],
            "repeat_times": repeat_times,
            "proxy_mode": "no-op",
        }

    return new_metrics, train_logs


def simulated_proxy_train_fn(
    detector_ckpt_path: str,
    stage_data: Any,
    memory_scenes: List[SceneDescriptor],
    train_cfg: IncrementalProxyTrainConfig,
    metrics_k: Dict[int, PerClassMetrics],
    T_c: Dict[int, int],
) -> Tuple[Dict[int, PerClassMetrics], Dict[str, Any]]:
    """Heuristic proxy trainer that simulates forgetting/retention.

    This does NOT train the detector. It adjusts per-class mAP using a simple
    coverage heuristic to make rewards meaningful for RL exploration:
      - Classes covered by selected memory scenes get a small boost.
      - Classes with zero coverage incur a small drop.
    """
    logger = get_root_logger(log_level="INFO")
    logger.info("=== RL Proxy Training (Simulated) ===")
    if hasattr(stage_data, "stage_id"):
        logger.info(f"Stage id: {stage_data.stage_id}")
    elif isinstance(stage_data, Mapping) and "stage_id" in stage_data:
        logger.info(f"Stage id: {stage_data['stage_id']}")

    coverage_counts = _aggregate_counts_from_scene_pool(memory_scenes)

    drop_default = getattr(train_cfg.extra_cfg, "sim_drop", 0.02) if hasattr(train_cfg, "extra_cfg") else 0.02
    boost_default = getattr(train_cfg.extra_cfg, "sim_boost", 0.01) if hasattr(train_cfg, "extra_cfg") else 0.01

    new_metrics: Dict[int, PerClassMetrics] = {}
    for cid, m in metrics_k.items():
        prev = float(m.map_curr)
        cov = float(coverage_counts.get(cid, 0))
        # Boost diminishes with diminishing returns; drop is constant when uncovered
        boost = boost_default * math.log1p(cov)
        drop = drop_default if cov <= 0 else 0.0
        curr = max(0.0, prev - drop + boost)
        new_metrics[cid] = PerClassMetrics(
            map_curr=curr,
            map_prev=prev,
            num_seen=m.num_seen + int(cov),
            num_mem=m.num_mem + int(cov),
        )

    train_logs = {
        "proxy_mode": "simulated",
        "drop_default": drop_default,
        "boost_default": boost_default,
        "coverage_nonzero_classes": len([c for c in coverage_counts if coverage_counts[c] > 0]),
    }

    return new_metrics, train_logs


def proxy_train_fn_real(
    detector_ckpt_path: str,
    stage_data: Any,
    memory_scenes: List[SceneDescriptor],
    train_cfg: IncrementalProxyTrainConfig,
    metrics_k: Dict[int, PerClassMetrics],
    T_c: Dict[int, int],
) -> Tuple[Dict[int, PerClassMetrics], Dict[str, Any]]:
    """Run a short-schedule real stage training using train_incremental_scene components.

    Requirements:
      - train_cfg.extra_cfg must contain:
          * incremental_cfg: the Config for incremental scene-based training
          * stage_definitions: list of stage definitions
          * scene_memory_json: path to a saved scene_memory_bank_stage_1.json
      - detector_ckpt_path: checkpoint to load (e.g., Stage 1 for Stage 2)
      - memory_scenes: RL-selected scenes (SceneDescriptor) for replay

    Behavior:
      - Forces epochs=1 and data.train.times=1 for speed.
      - Rebuilds a SceneMemoryBank from the provided Stage-1 JSON and the
        selected scene IDs, attaches it to the stage train dataset.
      - Calls train_model with validate=True, then parses the val log to build
        new per-class metrics (map_prev = map_curr).
    """
    logger = get_root_logger(log_level="INFO")
    logger.info("=== RL Proxy Training (Real) ===")

    extra = getattr(train_cfg, "extra_cfg", {}) or {}
    incremental_cfg = extra.get("incremental_cfg", None)
    stage_definitions = extra.get("stage_definitions", [])
    scene_memory_json = extra.get("scene_memory_json", None)

    if incremental_cfg is None or scene_memory_json is None:
        logger.warning("Missing incremental_cfg or scene_memory_json; falling back to simulated proxy.")
        return simulated_proxy_train_fn(
            detector_ckpt_path, stage_data, memory_scenes, train_cfg, metrics_k, T_c
        )

    try:
        from tools.train_incremental_scene import prepare_stage_config
        from mmdet3d.datasets import build_dataset, build_dataloader
        from mmdet3d.models import build_model
        from mmdet3d.apis import train_model
        from mmdet3d.apis.test import single_gpu_test
        from mmdet3d.datasets.incremental_mappings import create_mapping_from_config
        import torch
        import os
        import os.path as osp
        import time as _time
        from mmcv import Config as MMCVConfig
    except Exception as e:
        logger.warning(f"Failed to import training dependencies: {e}; using simulated proxy.")
        return simulated_proxy_train_fn(
            detector_ckpt_path, stage_data, memory_scenes, train_cfg, metrics_k, T_c
        )

    def _set_head_class_count(model_cfg, n_classes: int):
        """Utility to set n_classes on the head config (TR3DHead expects n_classes only)."""
        if hasattr(model_cfg, "head"):
            head_cfg = model_cfg.head
            if isinstance(head_cfg, dict):
                head_cfg["n_classes"] = n_classes
                head_cfg.pop("num_classes", None)  # avoid unsupported arg
            else:
                if hasattr(head_cfg, "n_classes"):
                    head_cfg.n_classes = n_classes
                if hasattr(head_cfg, "num_classes"):
                    try:
                        delattr(head_cfg, "num_classes")
                    except Exception:
                        pass

    def _copy_head_weights(model, state_dict):
        """Copy previous-stage head weights into expanded head."""
        if not hasattr(model, "head") or "head.cls_conv.kernel" not in state_dict:
            return
        prev_kernel = state_dict.get("head.cls_conv.kernel")
        prev_bias = state_dict.get("head.cls_conv.bias")
        if prev_kernel is None or prev_bias is None:
            return
        prev_classes = prev_kernel.shape[1]
        new_classes = model.head.cls_conv.kernel.shape[1]
        if prev_classes > new_classes:
            logger.warning(
                "Checkpoint has %d head classes but model only has %d; skipping head copy.",
                prev_classes,
                new_classes,
            )
            return
        try:
            with torch.no_grad():
                model.head.cls_conv.kernel.data[:, :prev_classes] = prev_kernel.to(model.head.cls_conv.kernel.device)
                model.head.cls_conv.bias.data[:, :prev_classes] = prev_bias.view(1, -1).to(model.head.cls_conv.bias.device)
            logger.info(
                "Copied head weights for %d classes into expanded head (%d total).",
                prev_classes,
                new_classes,
            )
        except Exception as e:
            logger.warning(f"Failed to copy head weights: {e}")

    # Resolve stage_id/name
    stage_id = stage_data.get("stage_id") if isinstance(stage_data, Mapping) else getattr(stage_data, "stage_id", None)
    if stage_id is None:
        stage_id = 2
    logger.info(f"Stage id: {stage_id}")

    # Force short schedule
    incremental_cfg = MMCVConfig(copy.deepcopy(incremental_cfg._cfg_dict))
    incremental_cfg.runner.max_epochs = 1
    incremental_cfg.data.train.times = 1

    # Prepare stage config
    stage_idx = stage_id - 1
    stage_definition = stage_definitions[stage_idx]
    work_dir = "./work_dirs/rl_proxy_debug"
    stage_cfg = prepare_stage_config(
        incremental_cfg.base_config, stage_definition, stage_idx, stage_definitions, work_dir,
        incremental_cfg=incremental_cfg, gpu_ids=[0]
    )

    # Dynamic head handling: expand class count per stage (cumulative classes)
    use_dynamic_head = getattr(incremental_cfg, "use_dynamic_head", False)
    cumulative_classes = sum(len(sd.get("class_indices", [])) for sd in stage_definitions[: stage_id])
    prev_stage_classes = sum(len(sd.get("class_indices", [])) for sd in stage_definitions[: stage_id - 1]) if stage_id > 1 else 0
    if hasattr(stage_cfg, "model"):
        _set_head_class_count(stage_cfg.model, cumulative_classes)
        if isinstance(stage_cfg.model.head, dict):
            stage_cfg.model.head.pop("num_classes", None)
    if use_dynamic_head:
        logger.info(
            f"Dynamic head mode: setting n_classes={cumulative_classes} for stage {stage_id} "
            f"(prev classes={prev_stage_classes})"
        )

    # Build mappings for dataset
    mappings = create_mapping_from_config(stage_definitions)
    class_names = [mappings["model_idx_to_name"][i] for i in range(35)]

    # Build (once) a mapping scene_id -> data_info from Stage 1 to enable replay
    data_info_map = extra.get("data_info_map")
    if data_info_map is None:
        try:
            stage1_definition = stage_definitions[0]
            stage1_cfg = prepare_stage_config(
                incremental_cfg.base_config,
                stage1_definition,
                0,
                stage_definitions,
                work_dir,
                incremental_cfg=incremental_cfg,
                gpu_ids=[0],
            )
            stage1_train_cfg = copy.deepcopy(stage1_cfg.data.train)
            innermost = stage1_train_cfg.dataset if hasattr(stage1_train_cfg, "dataset") else stage1_train_cfg
            innermost.type = "IncrementalScanNetDataset"
            innermost.stage_definition = stage1_definition
            innermost.mappings = mappings
            innermost.scene_memory_bank = None
            innermost.evaluation_mode = False
            innermost.all_stage_definitions = stage_definitions
            stage1_dataset = build_dataset(stage1_train_cfg)
            data_info_map = {}
            for di in getattr(stage1_dataset, "data_infos", []):
                sid = None
                if isinstance(di, dict):
                    if "point_cloud" in di and isinstance(di["point_cloud"], dict):
                        sid = di["point_cloud"].get("lidar_idx")
                    sid = sid or di.get("sample_idx") or di.get("scene_id")
                if sid:
                    data_info_map[str(sid)] = di
            # Fallback: if incremental stage-1 yielded no map, try standard ScanNetDataset
            if len(data_info_map) == 0:
                val_cfg = copy.deepcopy(stage1_cfg.data.val)
                val_cfg.type = "ScanNetDataset"
                # Strip incremental-only fields
                for attr in [
                    "evaluation_mode",
                    "stage_definition",
                    "mappings",
                    "object_memory_bank",
                    "scene_memory_bank",
                    "scene_dedup_strategy",
                    "all_stage_definitions",
                    "experiment_dir",
                    "use_pseudo_labels",
                    "pseudo_label_config",
                ]:
                    if hasattr(val_cfg, attr):
                        try:
                            delattr(val_cfg, attr)
                        except Exception:
                            pass
                    if isinstance(val_cfg, dict) and attr in val_cfg:
                        val_cfg.pop(attr, None)
                std_dataset = build_dataset(val_cfg)
                for di in getattr(std_dataset, "data_infos", []):
                    sid = None
                    if isinstance(di, dict):
                        if "point_cloud" in di and isinstance(di["point_cloud"], dict):
                            sid = di["point_cloud"].get("lidar_idx")
                        sid = sid or di.get("sample_idx") or di.get("scene_id")
                    if sid:
                        data_info_map[str(sid)] = di
            extra["data_info_map"] = data_info_map
            logger.info(f"Built Stage-1 data_info map with {len(data_info_map)} entries for replay injection.")
        except Exception as e:
            logger.warning(f"Failed to build data_info_map; replay will be skipped. Error: {e}")
            data_info_map = {}

    # Build scene_id -> data_info map once (from Stage 1) to enable replay
    data_info_map = extra.get("data_info_map")
    if data_info_map is None:
        try:
            stage1_definition = stage_definitions[0]
            stage1_cfg = prepare_stage_config(
                incremental_cfg.base_config,
                stage1_definition,
                0,
                stage_definitions,
                work_dir,
                incremental_cfg=incremental_cfg,
                gpu_ids=[0],
            )
            stage1_train_cfg = copy.deepcopy(stage1_cfg.data.train)
            innermost = stage1_train_cfg.dataset if hasattr(stage1_train_cfg, "dataset") else stage1_train_cfg
            innermost.type = "IncrementalScanNetDataset"
            innermost.stage_definition = stage1_definition
            innermost.mappings = mappings
            innermost.scene_memory_bank = None
            innermost.evaluation_mode = False
            innermost.all_stage_definitions = stage_definitions
            stage1_dataset = build_dataset(stage1_train_cfg)
            data_info_map = {}
            for di in getattr(stage1_dataset, "data_infos", []):
                sid = None
                if isinstance(di, dict):
                    if "point_cloud" in di and isinstance(di["point_cloud"], dict):
                        sid = di["point_cloud"].get("lidar_idx")
                    sid = sid or di.get("sample_idx") or di.get("scene_id")
                if sid:
                    data_info_map[str(sid)] = di
            extra["data_info_map"] = data_info_map
            logger.info(f"Built Stage-1 data_info map with {len(data_info_map)} entries for replay injection.")
        except Exception as e:
            logger.warning(f"Failed to build data_info_map; replay will be skipped. Error: {e}")
            data_info_map = {}

    # Rebuild scene memory bank from selected scenes
    selected_ids = [s.scene_id for s in memory_scenes]
    scene_memory_bank, _ = _rebuild_scene_memory_bank_from_json(scene_memory_json, selected_ids, data_info_map)

    # Build training dataset config (inject stage info and memory bank)
    train_dataset_cfg = copy.deepcopy(stage_cfg.data.train)
    inner = train_dataset_cfg.dataset if hasattr(train_dataset_cfg, "dataset") else train_dataset_cfg
    inner.type = "IncrementalScanNetDataset"
    inner.stage_definition = stage_definition
    inner.mappings = mappings
    if scene_memory_bank is not None and len(scene_memory_bank.memory_scenes) > 0:
        inner.scene_memory_bank = scene_memory_bank
    else:
        inner.scene_memory_bank = None  # Avoid replay if no data_info in snapshots
    inner.scene_dedup_strategy = getattr(incremental_cfg, "scene_dedup_strategy", "keep_both")
    inner.evaluation_mode = False
    inner.all_stage_definitions = stage_definitions
    inner.experiment_dir = stage_cfg.work_dir
    inner.use_pseudo_labels = False

    # Build val dataset config (IncrementalScanNetDataset, cumulative seen classes)
    val_dataset_cfg = copy.deepcopy(stage_cfg.data.val)
    val_dataset_cfg.type = "IncrementalScanNetDataset"
    cum_seen = []
    for sd in stage_definitions[: stage_idx + 1]:
        cum_seen.extend(sd["class_indices"])
    # Preserve numeric order of cumulative seen classes (already frequency-indexed)
    val_dataset_cfg.seen_classes_for_eval = cum_seen
    val_dataset_cfg.current_stage_classes = stage_definition["class_indices"]
    val_dataset_cfg.stage_definitions = stage_definitions
    val_dataset_cfg.mappings = mappings
    val_dataset_cfg.evaluation_mode = True
    val_dataset_cfg.all_stage_definitions = stage_definitions
    # Ensure evaluation class order matches canonical mapping (frequency order)
    val_dataset_cfg.classes = class_names

    stage_cfg.data.train = train_dataset_cfg
    stage_cfg.data.val = val_dataset_cfg

    train_dataset = build_dataset(stage_cfg.data.train)
    train_dataset.CLASSES = class_names
    datasets = [train_dataset]

    # Build model and load checkpoint (with manual head copy for incremental expansion)
    model = build_model(stage_cfg.model, train_cfg=stage_cfg.get("train_cfg"), test_cfg=stage_cfg.get("test_cfg"))
    checkpoint = torch.load(detector_ckpt_path, map_location="cpu")

    model_dict = model.state_dict()
    state_dict = checkpoint["state_dict"]
    filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(filtered, strict=False)
    if use_dynamic_head and hasattr(model, "head"):
        _copy_head_weights(model, state_dict)

    # Set canonical class names for evaluation order
    model.CLASSES = class_names

    # Train (short schedule)
    timestamp = _time.strftime("%Y%m%d_%H%M%S", _time.localtime())
    # Optional: evaluate stage-1 checkpoint before stage 2 starts (verification)
    if stage_id == 2 and not extra.get("stage1_eval_done"):
        try:
            stage1_definition = stage_definitions[0]
            stage1_cfg = prepare_stage_config(
                incremental_cfg.base_config,
                stage1_definition,
                0,
                stage_definitions,
                work_dir,
                incremental_cfg=incremental_cfg,
                gpu_ids=[0],
            )
            stage1_classes = len(stage1_definition.get("class_indices", []))
            _set_head_class_count(stage1_cfg.model, stage1_classes)
            if isinstance(stage1_cfg.model.head, dict):
                stage1_cfg.model.head.pop("num_classes", None)
            # Build model for eval
            eval_model = build_model(
                stage1_cfg.model,
                train_cfg=stage1_cfg.get("train_cfg"),
                test_cfg=stage1_cfg.get("test_cfg"),
            )
            ckpt = torch.load(detector_ckpt_path, map_location="cpu")
            eval_model.load_state_dict(ckpt["state_dict"], strict=True)
            eval_model.CLASSES = class_names
            # Build val dataset/dataloader
            eval_dataset = build_dataset(stage1_cfg.data.val)
            eval_dataset.CLASSES = class_names
            data_loader = build_dataloader(
                eval_dataset,
                samples_per_gpu=getattr(stage1_cfg.data, "samples_per_gpu", 1),
                workers_per_gpu=getattr(stage1_cfg.data, "workers_per_gpu", 2),
                dist=False,
                shuffle=False,
            )
            outputs = single_gpu_test(eval_model, data_loader, show=False)
            eval_kwargs = stage1_cfg.get("evaluation", {}) or {}
            eval_res = eval_dataset.evaluate(outputs, **eval_kwargs)
            logger.info(f"Stage-1 checkpoint eval (pre-Stage2): {eval_res}")
            extra["stage1_eval_done"] = True
        except Exception as e:
            logger.warning(f"Stage-1 pre-eval failed: {e}")

    train_model(
        model,
        datasets,
        stage_cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        meta={"stage_id": stage_id, "proxy_mode": "real"},
    )

    # Parse last val entry for metrics
    val_log = None
    if stage_cfg.work_dir and osp.isdir(stage_cfg.work_dir):
        log_files = [osp.join(stage_cfg.work_dir, f) for f in os.listdir(stage_cfg.work_dir) if f.endswith(".log.json")]
        if log_files:
            val_log = max(log_files, key=os.path.getmtime)
    model_idx_to_name = extra.get("model_idx_to_name", {})
    num_classes = len(model_idx_to_name) if model_idx_to_name else len(metrics_k)
    if val_log:
        new_metrics = load_metrics_from_log(val_log, model_idx_to_name, num_classes, ap_key_suffix="AP_0.25", default_prev_equal_curr=False)
    else:
        logger.warning("No val log found; using previous metrics.")
        new_metrics = metrics_k

    train_logs = {
        "proxy_mode": "real",
        "work_dir": stage_cfg.work_dir,
        "val_log": val_log,
    }
    return new_metrics, train_logs


def ensure_proxy_train_fn(train_cfg: IncrementalProxyTrainConfig) -> IncrementalProxyTrainConfig:
    """Ensure train_cfg has a proxy_train_fn, defaulting to the NO-OP stub."""
    if train_cfg.proxy_train_fn is None:
        train_cfg.proxy_train_fn = default_proxy_train_fn
    # Enforce 1x dataset repetition by default for RL experiments
    if getattr(train_cfg, "repeat_times", None) is None:
        train_cfg.repeat_times = 1
    return train_cfg
