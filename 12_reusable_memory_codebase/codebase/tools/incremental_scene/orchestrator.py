"""Orchestrator helpers for incremental scene trainer main loop."""

from __future__ import annotations

from typing import List

from .reviewing_ld import compute_review_segment_times


def should_use_sunrgbd_segmented_path(*,
                                      incremental_dataset_type: str,
                                      stage_idx: int,
                                      sunrgbd_reviewing_active: bool,
                                      learning_dynamics_enabled: bool,
                                      legacy_seg_enabled: bool) -> bool:
    """Gate for segmented SUNRGBD stage training path."""
    del stage_idx
    del legacy_seg_enabled
    return bool(
        incremental_dataset_type in ('IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset')
        and (sunrgbd_reviewing_active or learning_dynamics_enabled)
    )


def segmented_mode_label(*,
                         sunrgbd_reviewing_active: bool,
                         learning_dynamics_enabled: bool) -> str:
    if sunrgbd_reviewing_active:
        return 'SUNRGBD Reviewing'
    if learning_dynamics_enabled:
        return 'SUNRGBD Learning-dynamics'
    return 'SUNRGBD Segmented'


def resolve_segment_times(*,
                          stage_epochs: int,
                          repeat_times: int,
                          review_fractions: List[float]) -> List[int]:
    return compute_review_segment_times(
        stage_epochs=int(stage_epochs),
        repeat_times=int(repeat_times),
        review_fractions=list(review_fractions or []),
    )
