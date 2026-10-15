from __future__ import annotations

import numpy as np

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


def _build_dataset_stub(apply_to_memory_scenes: bool):
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
    ds.apply_pseudo_to_memory_scenes = bool(apply_to_memory_scenes)
    ds.data_infos = [
        {
            'point_cloud': {'lidar_idx': 'scene_nat'},
            'is_replay': False,
            'annos': {
                'gt_boxes_upright_depth': np.zeros((0, 6), dtype=np.float32),
                'class': np.zeros((0,), dtype=np.int64),
                'gt_num': 0,
            },
        },
        {
            'point_cloud': {'lidar_idx': 'scene_rep'},
            'is_replay': True,
            'annos': {
                'gt_boxes_upright_depth': np.zeros((0, 6), dtype=np.float32),
                'class': np.zeros((0,), dtype=np.int64),
                'gt_num': 0,
            },
        },
    ]
    ds.pseudo_labels = {
        'scene_nat': _canonical_record(label=1),
        'scene_rep': _canonical_record(label=1),
    }
    return ds


def test_scannet_pseudo_memory_toggle_off_keeps_replay_unmodified():
    ds = _build_dataset_stub(apply_to_memory_scenes=False)
    ds._inject_pseudo_labels_pre_pipeline()

    assert int(ds.data_infos[0]['annos']['gt_num']) == 1
    assert int(ds.data_infos[1]['annos']['gt_num']) == 0


def test_scannet_pseudo_memory_toggle_on_injects_replay_scenes():
    ds = _build_dataset_stub(apply_to_memory_scenes=True)
    ds._inject_pseudo_labels_pre_pipeline()

    assert int(ds.data_infos[0]['annos']['gt_num']) == 1
    assert int(ds.data_infos[1]['annos']['gt_num']) == 1
