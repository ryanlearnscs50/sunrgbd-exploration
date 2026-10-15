"""
Lightweight RL utilities for scene-based memory allocation.

This package provides:
- Data structures for per-class metrics and scene descriptors
- State vector construction for RL policies
- A small policy network that predicts class-wise memory allocations
- Greedy scene selection to satisfy per-class object quotas
- High-level environment and training stubs for RL-based allocators

The RL components are intentionally lightweight and loosely coupled to the
detector training code. They rely on a proxy training API supplied via
configuration rather than importing training scripts directly.
"""

from .structures import (  # noqa: F401
    PerClassMetrics,
    SceneDescriptor,
    build_metrics_from_raw,
    scene_descriptors_from_memory_json,
    scene_descriptors_from_memory_bank,
)
from .proxy_incremental import (  # noqa: F401
    load_metrics_from_log,
    proxy_train_fn_real,
)
from .state_builder import build_state  # noqa: F401
from .policy import MemoryAllocPolicy, allocation_from_logits  # noqa: F401
from .scene_selector import select_scenes_simple  # noqa: F401

__all__ = [
    "PerClassMetrics",
    "SceneDescriptor",
    "build_metrics_from_raw",
    "scene_descriptors_from_memory_json",
    "scene_descriptors_from_memory_bank",
    "build_state",
    "MemoryAllocPolicy",
    "allocation_from_logits",
    "select_scenes_simple",
    "load_metrics_from_log",
    "proxy_train_fn_real",
]
