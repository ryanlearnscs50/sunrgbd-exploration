"""Explicit box/pseudo pipeline contracts; synthetic data is not accuracy evidence."""
import copy
import json
import os
from pathlib import Path
import pickle
import runpy
import subprocess
import sys

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet3d.registry import DATASETS, MODELS

from test_host_runtime import config
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.configuration import stage_config
from ldmr_unidet3d.pseudo import (geometry_digest, load_pseudo, write_pseudo,
                                select_pseudo, scannet_policy)
from ldmr_unidet3d.sequence import sha256


def paths(config):
    return [Path(config[split+'_dataloader']['dataset']['datasets'][0]['ann_file'])
            for split in ('train', 'val')]


def box_config(config, artifact=None, teacher_hash=None):
    train, val = paths(config)
    mode = config['ldmr_protocol']['memory_mode']
    return stage_config(config, ScanNet35Protocol(), 2, train_info=train, val_info=val,
        work_dir=train.parent/'box_training', memory_mode=mode, epochs=1,
        crop_path=train.parent/'crops.pkl', target_mode='boxes',
        pseudo_path=artifact, teacher_sha256=teacher_hash)


def test_teacher_inputs_ignore_hidden_annotations_and_missing_mask_files(config):
    train, _ = paths(config)
    with train.open('rb') as stream:
        records = pickle.load(stream)['data_list']
    build = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              'tools/generate_pseudo.py'))['inference_dataset']
    base = config['test_dataloader']['dataset']['datasets'][0]
    observations = []
    manifests = []
    for corrupt in (False, True):
        rows = copy.deepcopy(records)
        if corrupt:
            for row in rows:
                row['instances'] = [dict(bbox_3d=[float('nan')]*6, bbox_label_3d=34)]
                row['pts_semantic_mask_path'] = '/missing/hidden_semantics.bin'
                row['pts_instance_mask_path'] = '/missing/hidden_instances.bin'
                row['is_replay'] = True
                row['ldmr_replay_from_stage'] = 999
        manifest = train.parent / f'teacher_input_{corrupt}.pkl'
        cfg = build(base, rows, ScanNet35Protocol().decoder_classes(1), manifest)
        with manifest.open('rb') as stream:
            manifests.append(pickle.load(stream))
        observations.append(DATASETS.build(cfg)[0])
    assert manifests[0] == manifests[1]
    first, second = observations
    assert torch.equal(first['inputs']['points'], second['inputs']['points'])
    assert torch.equal(first['data_samples'].gt_pts_seg.sp_pts_mask,
                       second['data_samples'].gt_pts_seg.sp_pts_mask)
    for observation in observations:
        assert 'pts_semantic_mask' not in observation['data_samples'].gt_pts_seg
        assert 'pts_instance_mask' not in observation['data_samples'].gt_pts_seg


def test_pseudo_and_visible_boxes_share_flip_rotation_and_scale(config):
    train, _ = paths(config)
    with train.open('rb') as stream:
        records = pickle.load(stream)['data_list']
    teacher = dict(boxes=[[2.5, 2.5, 2.5, 1, 2, 1]], labels=[0], scores=[.9])
    artifact = train.parent / 'augmentation_teacher.json'
    protocol = ScanNet35Protocol()
    write_pseudo(artifact, {'fixture': teacher}, protocol, 2,
                 teacher_sha256='a'*64, train_info_sha256=sha256(train),
                 geometry_sha256=geometry_digest(records), teacher_config_sha256='b'*64)
    cfg = box_config(config, artifact, 'a'*64)
    pipeline = cfg['train_dataloader']['dataset']['datasets'][0]['pipeline']
    for step in pipeline:
        if step['type'] == 'RandomFlip3D':
            step.update(flip_ratio_bev_horizontal=1., flip_ratio_bev_vertical=1.)
        if step['type'] == 'GlobalRotScaleTrans':
            step.update(rot_range=[np.pi/4, np.pi/4], scale_ratio_range=[2., 2.],
                        translation_std=[0., 0., 0.])
    gt = np.asarray([v['bbox_3d'] for v in records[0]['instances']], dtype=np.float32)
    pseudo = select_pseudo(teacher, gt, 7, scannet_policy(2))
    expected = np.concatenate([gt, pseudo['boxes']])
    expected[:, :2] *= -1  # both depth-frame flips
    cosine = np.sqrt(.5)
    rotation = np.array([[cosine, -cosine], [cosine, cosine]])
    expected[:, :2] = expected[:, :2] @ rotation.T
    expected[:, 3:5] = cosine * expected[:, 3:5].sum(1, keepdims=True)
    expected *= 2
    sample = DATASETS.build(cfg['train_dataloader']['dataset'])[0]
    boxes = sample['data_samples'].gt_instances_3d.bboxes_3d
    actual = torch.cat([boxes.gravity_center, boxes.tensor[:, 3:6]], dim=1).numpy()
    np.testing.assert_allclose(actual[:len(expected)], expected, atol=1e-5)
    assert not boxes.with_yaw


@pytest.mark.parametrize('use_pseudo', [False, True])
def test_box_pipeline_hidden_masks_geometry_and_cuda_gradients(config, use_pseudo):
    train, _ = paths(config)
    artifact = train.parent/'teacher.json' if use_pseudo else None
    with train.open('rb') as f:
        records = pickle.load(f)['data_list']
    if use_pseudo:
        write_pseudo(artifact, {'fixture': dict(boxes=[[2.5,2.5,2.5,1,1,1]], labels=[0], scores=[.9])},
            ScanNet35Protocol(), 2, teacher_sha256='a'*64, train_info_sha256=sha256(train),
            geometry_sha256=geometry_digest(records), teacher_config_sha256='b'*64)
    cfg = box_config(config, artifact, 'a'*64 if use_pseudo else None)
    assert cfg['model']['bbox_by_mask'] == [False]
    assert cfg['model']['target_by_distance'] == [True]
    pipeline = cfg['train_dataloader']['dataset']['datasets'][0]['pipeline']
    assert not any(step['type']=='ElasticTransfrom' for step in pipeline)
    observations = []
    for corrupt in (False, True):
        if corrupt:
            # Keep geometry and visible box annotations fixed. Destroy all
            # destination semantic/instance identity, including hidden classes.
            for row in records:
                points = np.fromfile(row['lidar_points']['lidar_path'], dtype=np.float32).reshape(-1,6)
                np.full(len(points), 1, np.int64).tofile(row['pts_semantic_mask_path'])
                np.full(len(points), -1, np.int64).tofile(row['pts_instance_mask_path'])
        np.random.seed(19)
        dataset = DATASETS.build(cfg['train_dataloader']['dataset'])
        sample = dataset[0]
        observations.append(sample)
        assert 'elastic_coords' not in sample['inputs']
        assert 'pts_semantic_mask' not in sample['data_samples'].gt_pts_seg
        assert 'pts_instance_mask' not in sample['data_samples'].gt_pts_seg
    first, second = observations
    assert torch.equal(first['inputs']['points'], second['inputs']['points'])
    a, b = [v['data_samples'].gt_instances_3d for v in observations]
    assert torch.equal(a.labels_3d, b.labels_3d)
    assert torch.equal(a.bboxes_3d.tensor, b.bboxes_3d.tensor)
    mode = cfg['ldmr_protocol']['memory_mode']
    expected_pseudo = int(use_pseudo and mode != 'scene')  # saved GT suppresses duplicate teacher box
    assert first['data_samples'].get('ldmr_pseudo_count', 0) == expected_pseudo
    assert 7 in a.labels_3d.tolist()
    if use_pseudo or mode == 'scene':
        assert 0 in a.labels_3d.tolist()
    model = MODELS.build(cfg['model']).cuda()
    model.init_weights()
    inputs = model.data_preprocessor(pseudo_collate([second]), training=True)
    original = copy.deepcopy(inputs['data_samples'][0].gt_instances_3d)
    xyz = inputs['inputs']['points'][0][:,:3]
    criterion = model.criterion
    captured = []
    def inspect(outputs, instances, names):
        observed = instances[0]
        assert torch.allclose(observed.bboxes_3d.gravity_center,
                              original.bboxes_3d.gravity_center - xyz.min(0).values)
        assert torch.equal(observed.bboxes_3d.tensor[:,3:6], original.bboxes_3d.tensor[:,3:6])
        assert torch.equal(observed.labels_3d, original.labels_3d)
        captured.append(True)
        return criterion(outputs, instances, names)
    model.criterion = inspect
    loss = sum(model(**inputs, mode='loss').values())
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert captured and gradients and all(torch.isfinite(g).all() for g in gradients)


def test_teacher_cli_strips_annotations_and_binds_provenance(config):
    train, val = paths(config)
    protocol = ScanNet35Protocol()
    cfg = stage_config(config, protocol, 1, train_info=train, val_info=val,
        work_dir=train.parent/'unused_teacher_work', memory_mode='none', epochs=1)
    cfg_path = train.parent/'teacher_config.py'
    Config(cfg).dump(str(cfg_path))
    model = MODELS.build(cfg['model'])
    model.init_weights()
    checkpoint = train.parent/'teacher.pth'
    torch.save(dict(state_dict=model.state_dict(), meta=dict(
        ldmr_protocol_id=protocol.identity, ldmr_stage=1, epoch=1)), checkpoint)
    del model
    output = train.parent/'predictions.json'
    tool = Path(__file__).resolve().parents[1]/'tools/generate_pseudo.py'
    command = [sys.executable, str(tool), '--teacher-config', str(cfg_path),
               '--checkpoint', str(checkpoint), '--train-info', str(train),
               '--upstream-root', os.environ['UNIDET3D_ROOT'], '--stage', '2', '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    with train.open('rb') as f:
        records = pickle.load(f)['data_list']
    expected = dict(teacher_sha256=sha256(checkpoint), train_info_sha256=sha256(train),
                    geometry_sha256=geometry_digest(records))
    predictions = load_pseudo(output, protocol, 2, scene_ids=['fixture'], **expected)
    assert np.all(predictions['fixture']['labels'] < 7)
    # Reuse the actual generated artifact through the real box training pipeline.
    dataset = DATASETS.build(box_config(config, output, sha256(checkpoint))['train_dataloader']['dataset'])
    assert len(dataset[0]['data_samples'].gt_instances_3d.labels_3d) >= 1
    before = output.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and output.read_bytes() == before
