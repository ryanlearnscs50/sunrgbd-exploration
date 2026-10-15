from tools.analysis_tools.visualize_object_memory_bank import summarize
from tools.analysis_tools.visualize_object_memory_insertions import box_xy_corners
from tools.analysis_tools.visualize_object_memory_bank_comparison import (
    bank_metrics,
    comparison_summary,
)

import numpy as np


def test_summary_reports_source_diversity_and_sparse_crop_counts():
    payload = {
        'format_version': 3,
        'stage_id': 2,
        'config': {'selection_strategy': 'random'},
        'exemplars': {
            0: [
                {'scene_id': 'a', 'stage_id': 1, 'point_count': 10},
                {'scene_id': 'b', 'stage_id': 1, 'point_count': 40},
            ],
            1: [
                {'scene_id': 'a', 'stage_id': 2, 'point_count': 200},
                {'scene_id': 'c', 'stage_id': 2, 'point_count': 1000},
            ],
        },
    }

    summary = summarize(payload, ['chair', 'table'])

    assert summary['source_scene_diversity'] == {
        'distinct_scenes': 3,
        'duplicate_assignments': 1,
    }
    assert summary['point_count'] == {
        'total': 1250,
        'min': 10,
        'mean': 312.5,
        'median': 120.0,
        'max': 1000,
        'below_20': 1,
        'below_50': 2,
        'below_100': 2,
        'below_500': 3,
    }


def test_box_xy_corners_respect_yaw_and_center():
    box = np.array([2.0, 3.0, 0.0, 4.0, 2.0, 1.0, np.pi / 2])
    corners = box_xy_corners(box)

    np.testing.assert_allclose(corners.mean(axis=0), [2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        corners.max(axis=0) - corners.min(axis=0), [2.0, 4.0], atol=1e-6)


def test_bank_comparison_reports_support_and_sources():
    payload = {
        'exemplars': {
            0: [
                {'scene_id': 'a', 'point_count': 10},
                {'scene_id': 'b', 'point_count': 100},
            ],
            1: [
                {'scene_id': 'a', 'point_count': 1000},
                {'scene_id': 'c', 'point_count': 2000},
            ],
        },
    }
    metrics = bank_metrics(payload)
    summary = comparison_summary(metrics, metrics)['random']

    assert summary == {
        'objects': 4,
        'total_points': 3110,
        'median_points': 550.0,
        'distinct_sources': 3,
        'duplicate_assignments': 1,
        'sparse': {'20': 1, '50': 1, '100': 1, '500': 2},
    }
