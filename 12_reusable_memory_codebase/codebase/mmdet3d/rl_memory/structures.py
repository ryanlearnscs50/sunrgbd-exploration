"""
Core data structures for RL-driven memory allocation.

These classes are intentionally lightweight and focus on the information
needed to build RL states and perform scene-level memory selection.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


@dataclass
class PerClassMetrics:
    """Per-class metrics snapshot for a single stage.

    Attributes:
        map_curr: Current stage mAP for this class.
        map_prev: Previous stage mAP for this class.
        num_seen: Total number of training instances seen so far.
        num_mem: Number of instances currently stored in the memory bank.
    """

    map_curr: float
    map_prev: float
    num_seen: int
    num_mem: int


@dataclass
class SceneDescriptor:
    """Lightweight description of a candidate scene for the memory bank.

    Attributes:
        scene_id: Unique scene identifier.
        class_counts: Mapping from class_id -> number of objects in this scene.
        embedding: Optional fixed-dimensional embedding for the scene.
        loss_per_class: Optional mapping class_id -> average loss for that class.
        metadata: Optional additional information for debugging/analysis.
    """

    scene_id: str
    class_counts: Dict[int, int]
    embedding: Optional[np.ndarray] = None
    loss_per_class: Optional[Dict[int, float]] = None
    metadata: Optional[Dict[str, Any]] = None


def build_metrics_from_raw(
    map_curr_per_class: Mapping[int, float],
    map_prev_per_class: Optional[Mapping[int, float]] = None,
    num_seen_per_class: Optional[Mapping[int, int]] = None,
    num_mem_per_class: Optional[Mapping[int, int]] = None,
    default_prev: float = 0.0,
) -> Dict[int, PerClassMetrics]:
    """Construct a metrics dict from raw per-class arrays or mappings.

    This helper is designed to sit between existing evaluation / logging code
    and the RL state builder. It makes minimal assumptions about the source
    format beyond providing per-class values indexed by class id.

    Args:
        map_curr_per_class: Mapping class_id -> current stage mAP.
        map_prev_per_class: Optional mapping class_id -> previous stage mAP.
            If not provided, ``default_prev`` is used.
        num_seen_per_class: Optional mapping class_id -> cumulative instances
            seen so far for this class. If not provided, defaults to 0.
        num_mem_per_class: Optional mapping class_id -> number of instances
            currently stored in the memory bank for this class. If not
            provided, defaults to 0.
        default_prev: Fallback previous mAP when ``map_prev_per_class`` does
            not contain a value for a given class.

    Returns:
        Dict[int, PerClassMetrics]: Per-class metrics suitable for RL.
    """
    metrics: Dict[int, PerClassMetrics] = {}

    all_class_ids: Iterable[int] = map_curr_per_class.keys()
    for cid in all_class_ids:
        map_curr = float(map_curr_per_class.get(cid, 0.0))
        if map_prev_per_class is not None:
            map_prev = float(map_prev_per_class.get(cid, default_prev))
        else:
            map_prev = float(default_prev)

        num_seen = int(num_seen_per_class.get(cid, 0)) if num_seen_per_class else 0
        num_mem = int(num_mem_per_class.get(cid, 0)) if num_mem_per_class else 0

        metrics[cid] = PerClassMetrics(
            map_curr=map_curr,
            map_prev=map_prev,
            num_seen=num_seen,
            num_mem=num_mem,
        )

    return metrics


def scene_descriptors_from_memory_json(
    json_data: Mapping[str, Any],
    stage_id: Optional[int] = None,
) -> List[SceneDescriptor]:
    """Convert a saved scene memory JSON structure into SceneDescriptor objects.

    This handles the unified ``scene_memory_bank_stage_*.json`` format used in
    experiments_logs, where the top-level structure is::

        {
          "scene_snapshots": {
            "scene0191_00": {
              "1": {
                "object_counts": {"2": 2, "4": 1, ...},
                ...
              },
              "2": { ... }
            },
            ...
          }
        }

    Args:
        json_data: Parsed JSON dict.
        stage_id: Optional stage id to select. If None, the latest available
            stage for each scene is used.

    Returns:
        List[SceneDescriptor]: One descriptor per (scene, chosen stage).
    """
    scene_snapshots = json_data.get("scene_snapshots", {})
    descriptors: List[SceneDescriptor] = []

    for scene_id, stage_dict in scene_snapshots.items():
        if not isinstance(stage_dict, Mapping):
            continue

        if stage_id is None:
            # Use the numerically largest stage key if not specified
            try:
                stage_keys = [int(k) for k in stage_dict.keys()]
                if not stage_keys:
                    continue
                chosen_stage = str(max(stage_keys))
            except ValueError:
                # Non-integer keys; skip this scene
                continue
        else:
            chosen_stage = str(stage_id)
            if chosen_stage not in stage_dict:
                continue

        snapshot = stage_dict.get(chosen_stage, {})
        raw_counts = snapshot.get("object_counts", {}) or {}

        class_counts: Dict[int, int] = {}
        for key, value in raw_counts.items():
            try:
                cid = int(key)
            except (TypeError, ValueError):
                continue
            class_counts[cid] = int(value)

        descriptors.append(
            SceneDescriptor(
                scene_id=str(scene_id),
                class_counts=class_counts,
                metadata={"stage_id": int(chosen_stage)},
            )
        )

    return descriptors


def scene_descriptors_from_memory_bank(
    memory_scenes: Mapping[str, Any],
) -> List[SceneDescriptor]:
    """Convert ``SceneMemoryBank.memory_scenes`` into ``SceneDescriptor`` list.

    The SceneMemoryBank class stores scenes as::

        {
          "scene0515_00": {
            "stages": {
              1: {"snapshot": {...}, "importance": 0.8, ...},
              2: {"snapshot": {...}, "importance": 0.7, ...},
            },
            "latest_stage": 2,
            "total_importance": 1.5,
            "present_classes": {0, 2, 4, 5},
          },
          ...
        }

    For each scene, this helper uses the latest stage snapshot and reads
    ``snapshot['object_counts']`` (if present) to populate class_counts.

    Args:
        memory_scenes: Typically ``SceneMemoryBank.memory_scenes``.

    Returns:
        List[SceneDescriptor]: One descriptor per scene.
    """
    descriptors: List[SceneDescriptor] = []

    for scene_id, scene_data in memory_scenes.items():
        if not isinstance(scene_data, Mapping):
            continue

        stages = scene_data.get("stages", {})
        if not stages:
            continue

        latest_stage = scene_data.get("latest_stage", None)
        if latest_stage is None:
            # Fallback: use max key if latest_stage is missing
            try:
                latest_stage = max(stages.keys())
            except ValueError:
                continue

        stage_entry = stages.get(latest_stage, {})
        snapshot = stage_entry.get("snapshot", {})

        raw_counts = snapshot.get("object_counts", {}) or {}
        class_counts: Dict[int, int] = {}
        for key, value in raw_counts.items():
            try:
                cid = int(key)
            except (TypeError, ValueError):
                continue
            class_counts[cid] = int(value)

        descriptors.append(
            SceneDescriptor(
                scene_id=str(scene_id),
                class_counts=class_counts,
                metadata={"latest_stage": int(latest_stage)},
            )
        )

    return descriptors

