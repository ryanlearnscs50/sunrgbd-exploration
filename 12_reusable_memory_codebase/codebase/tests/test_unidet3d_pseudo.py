"""Artifact and filtering contracts independent of the modern detector runtime."""
import copy
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.pseudo import (load_pseudo, prediction_arrays, scannet_policy,
                                select_pseudo, write_pseudo, geometry_digest)


def record(boxes=None, labels=None, scores=None):
    return dict(boxes=[[0, 0, 0, 2, 2, 2]] if boxes is None else boxes,
                labels=[0] if labels is None else labels, scores=[.9] if scores is None else scores)


@pytest.fixture
def artifact(tmp_path):
    path = tmp_path / 'teacher.json'
    p = ScanNet35Protocol()
    expected = dict(teacher_sha256='a'*64, train_info_sha256='b'*64, geometry_sha256='c'*64)
    write_pseudo(path, {'scene': record(), 'empty': record([], [], [])}, p, 2,
                 teacher_config_sha256='d'*64, **expected)
    return path, p, expected


def test_round_trip_empty_predictions_and_exclusive_output(artifact):
    path, p, expected = artifact
    rows = load_pseudo(path, p, 2, scene_ids=['scene', 'empty'], **expected)
    assert rows['empty']['boxes'].shape == (0, 6)
    np.testing.assert_array_equal(rows['scene']['labels'], [0])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_pseudo(path, {'scene': record()}, p, 2, teacher_config_sha256='d'*64, **expected)
    assert path.read_bytes() == original


@pytest.mark.parametrize('key,value', [('teacher_sha256','e'*64), ('train_info_sha256','e'*64),
    ('geometry_sha256','e'*64), ('stage',3), ('stage',True), ('teacher_stage',True),
    ('coordinates','bottom'), ('label_space','nyu40'), ('classes',['chair']),
    ('policy',{}), ('teacher_config_sha256','unknown')])
def test_provenance_rejection(artifact, key, value):
    path, p, expected = artifact
    payload = json.loads(path.read_text()); payload[key] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_pseudo(path, p, 2, **expected)


def test_coverage_file_changes_and_duplicate_keys(artifact):
    path, p, expected = artifact
    with pytest.raises(ValueError, match='coverage'):
        load_pseudo(path, p, 2, scene_ids=['scene'], **expected)
    with pytest.raises(ValueError, match='changed'):
        load_pseudo(path, p, 2, artifact_sha256='e'*64, **expected)
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError, match='duplicate'):
        load_pseudo(path, p, 2, **expected)


@pytest.mark.parametrize('change', [dict(labels=[7]), dict(labels=[True]), dict(labels=[0.0]),
    dict(labels=[-1]), dict(scores=[float('nan')]), dict(scores=[1.1]),
    dict(boxes=[[0,0,0,0,1,1]]), dict(boxes=[[0,0,0,1,1,1,0]]), dict(scores=[])])
def test_invalid_predictions(change):
    value = record(); value.update(change)
    with pytest.raises(ValueError):
        prediction_arrays(value, 7)


def test_confidence_boundary_and_class_agnostic_gt_filter():
    boxes = [[0,0,0,1,1,1], [5,0,0,1,1,1], [10,0,0,1,1,1]]
    output = select_pseudo(record(boxes, [0,1,2], [.45, .449, .9]),
                           np.array([[10,0,0,1,1,1]]), 7, scannet_policy(2))
    np.testing.assert_array_equal(output['labels'], [0])


def test_nms_uses_maintained_geometry_and_preserves_score_alignment():
    reference = runpy.run_path(str(ROOT/'mmdet3d/datasets/pseudo_label_utils.py'))
    boxes = np.array([[0,0,0,2,2,2], [.1,0,0,2,2,2], [8,0,0,2,2,2]], dtype=np.float32)
    scores = np.array([.6, .95, .8], dtype=np.float32)
    indices = reference['nms_indices_iou'](boxes, scores, .3)
    output = select_pseudo(record(boxes, [0,0,0], scores), np.empty((0,6)), 7, scannet_policy(2))
    np.testing.assert_array_equal(output['boxes'], boxes[sorted(indices)])
    np.testing.assert_array_equal(output['scores'], scores[sorted(indices)])
    output = select_pseudo(record(boxes[:2], [0,1], scores[:2]), np.empty((0,6)), 7, scannet_policy(2))
    assert len(output['boxes']) == 2  # class-wise NMS


def test_cap_precedes_gt_filter():
    policy = scannet_policy(2); policy['max_pseudo_per_scene'] = 1
    output = select_pseudo(record([[0,0,0,1,1,1], [5,0,0,1,1,1]], [0,1], [.9,.8]),
                           np.array([[0,0,0,1,1,1]]), 7, policy)
    assert len(output['boxes']) == 0


def test_geometry_digest_ignores_hidden_annotations_but_detects_points(tmp_path):
    points = tmp_path/'points'; points.write_bytes(b'original points')
    groups = tmp_path/'groups'; groups.write_bytes(b'original groups')
    row = dict(lidar_points=dict(lidar_path=str(points)), super_pts_path=str(groups),
               instances=[dict(bbox_label_3d=0)])
    before = geometry_digest([row])
    row['instances'] = []; row['pts_semantic_mask_path'] = 'not read'
    assert geometry_digest([row]) == before
    points.write_bytes(b'changed points')
    assert geometry_digest([row]) != before


@pytest.mark.parametrize('setting', ['scannet35_s3_freqorder_15_10_10',
                                   'scannet35_s10_freqorder_4444433333'])
def test_unvalidated_pseudo_protocol_is_rejected(tmp_path, setting):
    with pytest.raises(ValueError, match='only the S5'):
        write_pseudo(tmp_path / 'teacher.json', {'scene': record()},
                     ScanNet35Protocol(setting), 2, teacher_sha256='a'*64,
                     train_info_sha256='b'*64, geometry_sha256='c'*64,
                     teacher_config_sha256='d'*64)
    assert not (tmp_path / 'teacher.json').exists()
