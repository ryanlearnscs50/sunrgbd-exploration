from mmcv import ConfigDict

import pytest

from tools.train_incremental_scene import (
    _attach_object_memory_insertion,
    _detach_object_memory_insertion,
    _filter_sunrgbd_infos_to_model_label_space,
    _validate_object_memory_resume,
)

import numpy as np


def _type(transform):
    return transform.get('type')


def test_insertion_is_after_scannet_alignment_and_before_sampling():
    dataset = ConfigDict(pipeline=[
        ConfigDict(type='LoadPointsFromFile'),
        ConfigDict(type='LoadAnnotations3D'),
        ConfigDict(type='GlobalAlignment'),
        ConfigDict(type='PointSample'),
        ConfigDict(type='GlobalRotScaleTrans'),
    ])
    bank = object()

    _attach_object_memory_insertion(
        dataset, bank, dict(insertion_probability=1.0))

    assert list(map(_type, dataset.pipeline)) == [
        'LoadPointsFromFile', 'LoadAnnotations3D', 'GlobalAlignment',
        'InsertExemplarObjects', 'PointSample', 'GlobalRotScaleTrans']
    assert dataset.pipeline[3]['memory_bank'] is bank


def test_insertion_is_not_duplicated_when_pipeline_is_rewired():
    dataset = ConfigDict(pipeline=[
        ConfigDict(type='LoadPointsFromFile'),
        ConfigDict(type='LoadAnnotations3D'),
    ])
    bank = object()
    _attach_object_memory_insertion(dataset, bank)
    _attach_object_memory_insertion(dataset, bank)

    assert list(map(_type, dataset.pipeline)).count('InsertExemplarObjects') == 1


def test_insertion_is_removed_for_natural_learning_dynamics_pool():
    bank = object()
    dataset = ConfigDict(
        object_memory_bank=bank,
        pipeline=[
            ConfigDict(type='LoadPointsFromFile'),
            ConfigDict(type='LoadAnnotations3D'),
            ConfigDict(type='InsertExemplarObjects', memory_bank=bank),
            ConfigDict(type='PointSample'),
        ])

    _detach_object_memory_insertion(dataset)

    assert dataset.object_memory_bank is None
    assert list(map(_type, dataset.pipeline)) == [
        'LoadPointsFromFile', 'LoadAnnotations3D', 'PointSample']


class _ResumeBank:
    def __init__(self, stage, classes):
        self.loaded_stage_id = stage
        self._classes = classes

    def get_stored_classes(self):
        return self._classes


def test_object_memory_resume_requires_previous_stage_and_exact_classes():
    stages = [
        dict(stage_id=1, class_indices=[0, 1]),
        dict(stage_id=2, class_indices=[2, 3]),
        dict(stage_id=3, class_indices=[4, 5]),
    ]
    _validate_object_memory_resume(
        _ResumeBank(2, [0, 1, 2, 3]), 3, stages, 'bank.pkl')

    with pytest.raises(ValueError, match='stage mismatch'):
        _validate_object_memory_resume(
            _ResumeBank(1, [0, 1]), 3, stages, 'bank.pkl')
    with pytest.raises(ValueError, match='class coverage mismatch'):
        _validate_object_memory_resume(
            _ResumeBank(2, [0, 1, 2]), 3, stages, 'bank.pkl')


def test_carrier_scene_eval_filters_future_labels_without_mutating_source():
    source = [{
        'annos': {
            'gt_num': 3,
            'class': np.asarray([0, 15, 17], dtype=np.int64),
            'name': np.asarray(['zero', 'fifteen', 'future'], dtype=object),
            'gt_boxes_upright_depth': np.arange(21, dtype=np.float32).reshape(3, 7),
            'bbox': [[0], [1], [2]],
            'index': np.asarray([8, 9, 10], dtype=np.int32),
            'scene_scalar': 'preserved',
        },
    }]

    filtered = _filter_sunrgbd_infos_to_model_label_space(source, 16)

    assert filtered[0]['annos']['gt_num'] == 2
    assert filtered[0]['annos']['class'].tolist() == [0, 15]
    assert filtered[0]['annos']['name'].tolist() == ['zero', 'fifteen']
    assert filtered[0]['annos']['gt_boxes_upright_depth'].shape == (2, 7)
    assert filtered[0]['annos']['bbox'] == [[0], [1]]
    assert filtered[0]['annos']['index'].tolist() == [0, 1]
    assert filtered[0]['annos']['scene_scalar'] == 'preserved'
    assert source[0]['annos']['gt_num'] == 3
    assert source[0]['annos']['class'].tolist() == [0, 15, 17]
