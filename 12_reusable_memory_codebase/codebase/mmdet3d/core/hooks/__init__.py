"""
Hooks for mmdet3d training pipeline.

Custom hooks for incremental learning and scene discovery.
"""

from .scene_metrics_hook import SceneMetricsHook

__all__ = [
    'SceneMetricsHook'
]