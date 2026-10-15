import numpy as np

from mmdet3d.datasets.incremental_sunrgbd import (
    IncrementalSUNRGBDDataset,
    _aabb6_from_centered_depth_boxes,
)
from mmdet3d.datasets.pseudo_label_utils import filter_pseudo_by_iou_against_gt
from mmdet3d.utils.pregenerate_pseudo_labels_sunrgbd import (
    _bottom_center_to_gravity_center_keep_yaw,
)


def test_sunrgbd_bottom_to_gravity_center_adds_half_height():
    bottom = np.array([[1.0, 2.0, 3.0, 0.8, 0.6, 1.2, 0.3]], dtype=np.float32)
    out = _bottom_center_to_gravity_center_keep_yaw(bottom)
    assert out.shape == (1, 7)
    assert np.isclose(out[0, 2], 3.0 + 1.2 * 0.5)
    assert np.isclose(out[0, 6], 0.3)


def test_sunrgbd_bottom_center_pseudo_matches_gt_and_is_suppressed():
    gt = np.array([[0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.2]], dtype=np.float32)
    bottom = gt.copy()
    bottom[0, 2] = gt[0, 2] - gt[0, 5] * 0.5  # bottom-centred z

    record = {
        "gt_boxes_upright_depth": bottom,
        "class": np.array([3], dtype=np.int64),
        "scores": np.array([0.99], dtype=np.float32),
        "center_type": "bottom",
    }

    # Call the conversion helper without constructing a full dataset instance.
    ds = IncrementalSUNRGBDDataset.__new__(IncrementalSUNRGBDDataset)
    pseudo_boxes, _, _ = ds._extract_pseudo_record(record)
    assert pseudo_boxes.shape == (1, 7)
    assert np.allclose(pseudo_boxes, gt, atol=1e-6)

    gt_aabb6 = _aabb6_from_centered_depth_boxes(gt)
    pseudo_aabb6 = _aabb6_from_centered_depth_boxes(pseudo_boxes)
    keep = filter_pseudo_by_iou_against_gt(gt_aabb6, pseudo_aabb6, iou_thr=0.25)
    assert keep.shape == (1,)
    assert bool(keep[0]) is False

