from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.validate_scannet_alignment_contract import (
    validate_scannet_alignment_contract,
)


def _build_synthetic_scannet_root(
    tmp_path: Path,
    *,
    corrupt_gt_boxes: bool = False,
    identity_axis_align: bool = False,
) -> tuple[Path, Path]:
    data_root = tmp_path / "scannet"
    instance_dir = data_root / "scannet_instance_data_40class"
    instance_dir.mkdir(parents=True, exist_ok=True)

    infos = []
    for i in range(2):
        scene_id = f"scene{i:04d}_00"

        points = np.zeros((128, 6), dtype=np.float32)
        points[:, 0] = np.linspace(0.0, 1.0, 128, dtype=np.float32)
        points[:, 1] = 0.25
        points[:, 2] = 0.5
        points[:, 3:] = 0.1

        axis_align = np.eye(4, dtype=np.float32)
        if not identity_axis_align:
            axis_align[:3, 3] = np.array([5.0 + float(i), -2.0, 1.5], dtype=np.float32)

        unaligned_bbox = np.array([[0.5, 0.25, 0.5, 0.3, 0.3, 0.3, 5]], dtype=np.float32)
        aligned_bbox = unaligned_bbox.copy()
        aligned_bbox[:, :3] = (
            unaligned_bbox[:, :3] @ axis_align[:3, :3].T + axis_align[:3, 3]
        )

        gt_boxes = unaligned_bbox[:, :6] if corrupt_gt_boxes else aligned_bbox[:, :6]

        np.save(instance_dir / f"{scene_id}_vert.npy", points)
        np.save(instance_dir / f"{scene_id}_aligned_bbox.npy", aligned_bbox)
        np.save(instance_dir / f"{scene_id}_unaligned_bbox.npy", unaligned_bbox)
        np.save(instance_dir / f"{scene_id}_axis_align_matrix.npy", axis_align)

        infos.append(
            {
                "point_cloud": {"lidar_idx": scene_id, "num_features": 6},
                "pts_path": f"points/{scene_id}.bin",
                "annos": {
                    "gt_num": int(gt_boxes.shape[0]),
                    "gt_boxes_upright_depth": gt_boxes.astype(np.float32),
                    "class": aligned_bbox[:, -1].astype(np.int64),
                    "axis_align_matrix": axis_align.astype(np.float32),
                },
            }
        )

    ann_file = data_root / "scannet_infos_train_40class_corrected.pkl"
    with ann_file.open("wb") as f:
        pickle.dump(infos, f)
    return data_root, ann_file


def test_scannet_alignment_contract_validator_happy_path(tmp_path: Path):
    data_root, ann_file = _build_synthetic_scannet_root(tmp_path)
    report = validate_scannet_alignment_contract(
        data_root=str(data_root),
        ann_file=str(ann_file),
        sample_scenes=16,
        min_aligned_center_ratio=0.99,
        fail_on_mismatch=False,
    )
    assert report["ok"] is True
    assert int(report["box_mismatch_scene_count"]) == 0
    assert int(report["low_aligned_ratio_scene_count"]) == 0
    assert int(report["raw_false_pass_scene_count"]) == 0
    assert float(report["aligned_center_ratio_mean"]) >= 0.99
    assert float(report["raw_center_ratio_mean"]) < 0.99


def test_scannet_alignment_contract_validator_detects_gt_bbox_mismatch(tmp_path: Path):
    data_root, ann_file = _build_synthetic_scannet_root(tmp_path, corrupt_gt_boxes=True)
    report = validate_scannet_alignment_contract(
        data_root=str(data_root),
        ann_file=str(ann_file),
        sample_scenes=16,
        min_aligned_center_ratio=0.99,
        fail_on_mismatch=False,
    )
    assert report["ok"] is False
    assert int(report["box_mismatch_scene_count"]) > 0
    assert int(report["low_aligned_ratio_scene_count"]) > 0

    with pytest.raises(RuntimeError):
        validate_scannet_alignment_contract(
            data_root=str(data_root),
            ann_file=str(ann_file),
            sample_scenes=16,
            min_aligned_center_ratio=0.99,
            fail_on_mismatch=True,
        )


def test_scannet_alignment_contract_validator_detects_raw_false_pass(tmp_path: Path):
    data_root, ann_file = _build_synthetic_scannet_root(tmp_path, identity_axis_align=True)
    report = validate_scannet_alignment_contract(
        data_root=str(data_root),
        ann_file=str(ann_file),
        sample_scenes=16,
        min_aligned_center_ratio=0.99,
        fail_on_mismatch=False,
    )
    assert report["ok"] is False
    assert int(report["low_aligned_ratio_scene_count"]) == 0
    assert int(report["raw_false_pass_scene_count"]) > 0
