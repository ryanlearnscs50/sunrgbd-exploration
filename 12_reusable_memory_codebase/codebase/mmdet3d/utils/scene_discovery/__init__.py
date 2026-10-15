"""
Scene Discovery Utilities for Incremental Learning

This module provides utilities for discovering optimal scene combinations
for memory banks in incremental learning experiments.

Main components:
- SceneMetricsCollector: Collect and analyze per-scene training metrics
- GradientAlignmentScorer: Compute gradient alignment between scenes and validation data
- FastTrialRunner: Execute fast training trials for scene evaluation
- SceneSubsetSearcher: Find optimal subsets of scenes using various algorithms
"""

from .metrics_collector import SceneMetricsCollector
from .gradient_alignment import GradientAlignmentScorer
from .trial_runner import FastTrialRunner
from .subset_search import GreedySubsetSearch, BeamSubsetSearch, RandomBaselineSearch
from .data_utils import SceneDataLoader

__all__ = [
    'SceneMetricsCollector',
    'GradientAlignmentScorer', 
    'FastTrialRunner',
    'GreedySubsetSearch',
    'BeamSubsetSearch', 
    'RandomBaselineSearch',
    'SceneDataLoader'
]