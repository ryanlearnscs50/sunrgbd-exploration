"""
Environment step wrapper for RL-driven memory allocation.

This module provides a high-level function that:
  1) Builds an RL state from per-class metrics for stage k.
  2) Uses a policy to produce class-wise allocation logits.
  3) Converts logits into per-class target object counts T_c.
  4) Selects memory scenes greedily from a candidate pool.
  5) Delegates proxy training + evaluation for stage k+1 to a user-supplied
     function configured via ``train_cfg``.
  6) Computes a reward based on forgetting between stages.

The detector training/evaluation is intentionally abstracted behind a
``proxy_train_fn`` callback supplied via ``train_cfg``. This keeps the RL
components loosely coupled to the rest of the codebase.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from mmdet3d.utils import get_root_logger

from .policy import MemoryAllocPolicy, allocation_from_logits
from .scene_selector import select_scenes_simple
from .state_builder import build_state
from .structures import PerClassMetrics, SceneDescriptor


def _get_proxy_train_fn(train_cfg: Any):
    """Extract proxy training function from a config-like object.

    Expected signatures:
        proxy_train_fn(
            detector_ckpt_path: str,
            stage_data: Any,
            memory_scenes: List[SceneDescriptor],
            train_cfg: Any,
        ) -> Tuple[Dict[int, PerClassMetrics], Dict[str, Any]]
    """
    if train_cfg is None:
        return None

    # Attribute-style access (e.g., SimpleNamespace, Config, IncrementalProxyTrainConfig)
    fn = getattr(train_cfg, "proxy_train_fn", None)
    if fn is not None:
        return fn

    # Dict-style access
    if isinstance(train_cfg, Mapping):
        return train_cfg.get("proxy_train_fn", None)

    return None


def run_stage_with_allocation(
    detector_ckpt_path: str,
    stage_data: Any,
    metrics_k: Dict[int, PerClassMetrics],
    policy: MemoryAllocPolicy,
    scene_pool: List[SceneDescriptor],
    max_scenes: int,
    total_object_slots: int,
    train_cfg: Optional[Any],
) -> Tuple[float, Dict[int, PerClassMetrics], Dict[str, Any]]:
    """Run one incremental stage as an RL environment step.

    Steps:
      1) state = build_state(metrics_k)
      2) logits = policy(state)
      3) T_c = allocation_from_logits(logits, total_object_slots)
      4) memory_scenes = select_scenes_simple(T_c, scene_pool, max_scenes)
      5) Train detector for this stage with (stage_data + memory_scenes)
         using a short proxy training schedule (handled by proxy_train_fn).
      6) Evaluate detector on a validation set and compute new per-class
         metrics (returned by proxy_train_fn).
      7) Compute reward = negative forgetting:
             reward = -sum_c max(0, mAP_prev_c - mAP_curr_c_next)
         where mAP_prev_c comes from metrics_k.map_curr and
         mAP_curr_c_next from the returned new_metrics.map_curr.

    Args:
        detector_ckpt_path: Path to the detector checkpoint at stage k.
        stage_data: Stage-specific data/config as used by the training code.
        metrics_k: Per-class metrics dict for stage k.
        policy: Memory allocation policy network.
        scene_pool: Candidate scenes for replay.
        max_scenes: Maximum number of scenes to select for memory.
        total_object_slots: Global object-level memory budget across classes.
        train_cfg: Lightweight training config for proxy training. Must provide
            a ``proxy_train_fn`` callable, or training will be skipped.

    Returns:
        Tuple of (reward, new_metrics, log_info).
    """
    logger = get_root_logger(log_level="INFO")

    # 1) Build RL state
    state_vec = build_state(metrics_k)
    state_tensor = torch.from_numpy(state_vec).to(
        next(policy.parameters()).device
    )

    # 2) Policy forward
    logits = policy(state_tensor)

    # 3) Allocation targets per class
    T_c = allocation_from_logits(logits.detach(), total_object_slots)

    # 4) Greedy scene selection
    memory_scenes = select_scenes_simple(T_c, scene_pool, max_scenes)

    # 5) Proxy training + evaluation
    proxy_train_fn = _get_proxy_train_fn(train_cfg)
    training_skipped = False
    train_logs: Dict[str, Any] = {}

    if proxy_train_fn is None:
        # Safe default: skip training and reuse previous metrics
        logger.info(
            "RL env: no proxy_train_fn provided; "
            "skipping detector training and reusing previous metrics."
        )
        new_metrics = metrics_k
        training_skipped = True
        train_logs["proxy_train_skipped"] = True
        train_logs["proxy_train_reason"] = (
            "No proxy_train_fn found in train_cfg; "
            "using previous metrics as a placeholder."
        )
    else:
        logger.info("RL env: invoking proxy_train_fn for stage transition.")
        new_metrics, train_logs = proxy_train_fn(
            detector_ckpt_path=detector_ckpt_path,
            stage_data=stage_data,
            memory_scenes=memory_scenes,
            train_cfg=train_cfg,
            metrics_k=metrics_k,
            T_c=T_c,
        )

    # 6) Compute reward based on forgetting between stages
    total_drop = 0.0
    for cid, prev_metrics in metrics_k.items():
        prev_map = float(prev_metrics.map_curr)
        curr_metrics = new_metrics.get(cid, None)
        curr_map = float(curr_metrics.map_curr) if curr_metrics else 0.0
        total_drop += max(0.0, prev_map - curr_map)
    reward = -float(total_drop)

    # 7) Construct logging info
    log_info: Dict[str, Any] = {
        "T_c": T_c,
        "state": state_vec.astype(np.float32),
        "logits": logits.detach().cpu().numpy(),
        "reward": reward,
        "num_memory_scenes": len(memory_scenes),
        "training_skipped": training_skipped,
    }
    log_info.update(train_logs or {})

    # Log high-level summary for probing
    logger.info(
        "RL env summary: reward=%.4f, num_memory_scenes=%d, "
        "training_skipped=%s",
        reward,
        len(memory_scenes),
        training_skipped,
    )

    return reward, new_metrics, log_info
