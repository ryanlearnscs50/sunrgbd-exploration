from __future__ import annotations

import pytest

from mmdet3d.utils.learning_dynamics_scoring import (
    validate_sunrgbd_ld_reviewing_design_consistency,
)


def test_scannet_ld_reviewing_consistency_rejects_mixed_policy():
    with pytest.raises(ValueError):
        validate_sunrgbd_ld_reviewing_design_consistency(
            learning_dynamics_selection=True,
            reviewing_enabled=True,
            reviewing_weight_policy_type='ap_drop',
            ld_iou_mode='0.25',
            reviewing_weight_iou_thr=0.25,
        )


def test_scannet_ld_reviewing_consistency_rejects_iou_mismatch():
    with pytest.raises(ValueError):
        validate_sunrgbd_ld_reviewing_design_consistency(
            learning_dynamics_selection=True,
            reviewing_enabled=True,
            reviewing_weight_policy_type='ld_drop',
            ld_iou_mode='0.50',
            reviewing_weight_iou_thr=0.25,
        )


def test_scannet_ld_reviewing_consistency_accepts_ld_drop_with_matching_iou():
    validate_sunrgbd_ld_reviewing_design_consistency(
        learning_dynamics_selection=True,
        reviewing_enabled=True,
        reviewing_weight_policy_type='ld_drop',
        ld_iou_mode='0.25',
        reviewing_weight_iou_thr=0.25,
    )


def test_scannet_ld_reviewing_consistency_accepts_fixed_policy():
    validate_sunrgbd_ld_reviewing_design_consistency(
        learning_dynamics_selection=True,
        reviewing_enabled=True,
        reviewing_weight_policy_type='fixed',
        ld_iou_mode='0.50',
        reviewing_weight_iou_thr=0.25,
    )
