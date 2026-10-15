from __future__ import annotations

import pickle

import numpy as np
import pytest

from mmdet3d.datasets.incremental_scannet import IncrementalScanNetDataset


def _canonical_record(label: int = 1, score: float = 0.9):
    return {
        'boxes': np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        'labels': np.array([int(label)], dtype=np.int64),
        'scores': np.array([float(score)], dtype=np.float32),
        'num_detections': 1,
        'label_space': 'nyu40',
        'center_type': 'gravity',
        'axis_aligned': True,
        'box_type': 'upright_depth_6d',
    }


def test_scannet_validate_pseudo_canonical_accepts_valid_payload():
    ds = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
    payload = {'scene001': _canonical_record(label=5)}
    ds._validate_pseudo_canonical(payload)


def test_scannet_validate_pseudo_canonical_rejects_noncanonical_payload():
    ds = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
    bad = _canonical_record(label=5)
    bad['center_type'] = 'bottom'
    with pytest.raises(ValueError, match='center_type must be'):
        ds._validate_pseudo_canonical({'scene001': bad})


def test_scannet_load_pregenerated_fails_fast_on_noncanonical_file(tmp_path):
    p = tmp_path / 'bad_pseudo.pkl'
    bad = _canonical_record(label=5)
    bad['label_space'] = 'model_idx'
    with p.open('wb') as f:
        pickle.dump({'scene001': bad}, f)

    ds = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
    ds.stage_definition = {'stage_id': 2}
    ds.pseudo_label_config = {}
    ds.mappings = {}

    with pytest.raises(Exception, match='Failed to load pre-generated pseudo labels'):
        ds._load_pregenerated_pseudo_labels(str(p))


def test_scannet_inject_pseudo_pre_pipeline_uses_canonical_payload():
    ds = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
    ds.stage_idx = 1
    ds.stage_definition = {'stage_id': 2}
    ds.all_stage_definitions = [
        {'stage_id': 1, 'nyu40_ids': [1]},
        {'stage_id': 2, 'nyu40_ids': [2]},
    ]
    ds.pseudo_vs_gt_iou_thr = 0.25
    ds.pseudo_nms_iou_thr = 0.3
    ds.debug_mode = False
    ds.mappings = {'nyu40_to_model_idx': {1: 0, 2: 1}}

    ds.data_infos = [{
        'point_cloud': {'lidar_idx': 'scene001'},
        'annos': {
            'gt_boxes_upright_depth': np.zeros((0, 6), dtype=np.float32),
            'class': np.zeros((0,), dtype=np.int64),
            'gt_num': 0,
        },
    }]
    ds.pseudo_labels = {'scene001': _canonical_record(label=1)}

    ds._inject_pseudo_labels_pre_pipeline()

    ann = ds.data_infos[0]['annos']
    assert int(ann['gt_num']) == 1
    assert np.asarray(ann['gt_boxes_upright_depth']).shape == (1, 6)
    assert np.asarray(ann['class']).tolist() == [1]

    summary = getattr(ds, 'pseudo_injection_summary', None)
    assert isinstance(summary, dict)
    assert int(summary.get('injected_scenes', 0)) == 1
    assert int(summary.get('total_pseudo_boxes_used', 0)) == 1
