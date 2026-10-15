"""
Utilities to convert per-class metrics into fixed-length RL state vectors.
"""

from typing import Dict, Iterable, List

import numpy as np

from .structures import PerClassMetrics


def build_state(metrics_k: Dict[int, PerClassMetrics]) -> np.ndarray:
    """Build a flat RL state vector from per-class metrics.

    For each class c in sorted class id order, this computes a feature vector:

        [map_curr, map_drop, freq_ratio, mem_ratio]

    where:
        map_drop   = max(0, map_prev - map_curr)
        freq_ratio = num_seen / max_num_seen_over_classes
        mem_ratio  = num_mem / max_num_mem_over_classes

    All features are concatenated and returned as a 1D float32 NumPy array.

    Args:
        metrics_k: Mapping class_id -> PerClassMetrics for the current stage.

    Returns:
        np.ndarray of shape [num_classes * 4] with dtype float32.
    """
    if not metrics_k:
        return np.zeros((0,), dtype=np.float32)

    class_ids: List[int] = sorted(metrics_k.keys())

    num_seen_values = np.array(
        [metrics_k[c].num_seen for c in class_ids], dtype=np.float32
    )
    num_mem_values = np.array(
        [metrics_k[c].num_mem for c in class_ids], dtype=np.float32
    )

    max_seen = float(num_seen_values.max()) if num_seen_values.size > 0 else 1.0
    max_mem = float(num_mem_values.max()) if num_mem_values.size > 0 else 1.0

    if max_seen <= 0.0:
        max_seen = 1.0
    if max_mem <= 0.0:
        max_mem = 1.0

    features: Iterable[float] = []
    feature_list: List[float] = []

    for cid in class_ids:
        m = metrics_k[cid]
        map_curr = float(m.map_curr)
        map_drop = float(max(0.0, m.map_prev - m.map_curr))
        freq_ratio = float(m.num_seen) / max_seen
        mem_ratio = float(m.num_mem) / max_mem

        feature_list.extend([map_curr, map_drop, freq_ratio, mem_ratio])

    state = np.asarray(feature_list, dtype=np.float32)
    return state

