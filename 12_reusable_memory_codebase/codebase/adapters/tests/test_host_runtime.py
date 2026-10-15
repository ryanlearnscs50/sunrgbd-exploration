"""Run ONLY with the isolated host Python and adapters/upstream on PYTHONPATH.

Synthetic fixtures verify host contracts, not real-data accuracy/superpoints.
"""
import os
import copy
import shutil
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS, MODELS

from ldmr_unidet3d import host
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.configuration import stage_config
from ldmr_unidet3d.data import convert_info, stage_records
from ldmr_unidet3d.objects import extract_crop, save_crops


@pytest.mark.parametrize('include_valid', [True, False])
def test_postprocessing_drops_boxes_without_volumetric_point_support(include_valid):
    from types import SimpleNamespace, MethodType
    from mmengine.config import ConfigDict
    from unidet3d.unidet3d import UniDet3D
    proxy = SimpleNamespace(
        test_cfg=ConfigDict(topk_insts=3, iou_thr=[.5], low_sp_thr=.18, up_sp_thr=.81),
        decoder=SimpleNamespace(datasets=['scannet']), fast_nms=[True],
        use_superpoints=[True],
        _single_scene_multiclass_nms=lambda boxes, scores, labels, *_: (boxes, scores, labels))
    proxy.trim_bboxes_by_superpoints = MethodType(UniDet3D.trim_bboxes_by_superpoints, proxy)
    points = torch.tensor([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                           for z in (-.5, .5)] + [[5., 5., 5.]])
    boxes = torch.tensor([[0., 0., 0., 2., 2., 2.],
                          [10., 10., 10., 2., 2., 2.],  # no point support
                          [5., 5., 5., 2., 2., 2.]])  # a single point
    out = dict(cls_preds=[torch.tensor([[3., 0.], [2., 0.], [1., 0.]])], bboxes=[boxes])
    if not include_valid:
        out = {key: [value[0][1:]] for key, value in out.items()}
    result = host.LDMRUniDet3D.predict_by_feat(proxy, out, [torch.arange(9)],
                                             [points], ['scannet'])[0]
    assert torch.isfinite(result[0].tensor).all()
    assert len(result[0]) == int(include_valid)
    if not include_valid:
        assert result[1].numel() == result[2].numel() == 0
        return
    assert torch.equal(result[0].gravity_center, torch.zeros(1, 3))
    assert torch.equal(result[0].tensor[:, 3:6], torch.ones(1, 3))
    assert result[1].tolist() == [0]
    assert torch.equal(result[2], out['cls_preds'][0].softmax(-1)[:1, 0])


@pytest.mark.parametrize('field', ['cls_preds', 'bboxes'])
def test_postprocessing_rejects_nonfinite_decoder_outputs(field):
    from types import SimpleNamespace
    out = dict(cls_preds=[torch.zeros(1, 2)], bboxes=[torch.ones(1, 6)])
    out[field][0][0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite decoder'):
        host.LDMRUniDet3D.predict_by_feat(SimpleNamespace(), out, [torch.arange(2)],
                                        [torch.zeros(2, 3)], ['scannet'])


def test_prediction_preserves_masks_even_when_upstream_raises(monkeypatch):
    from types import SimpleNamespace
    from unidet3d.unidet3d import UniDet3D
    sample = SimpleNamespace(gt_pts_seg=SimpleNamespace(sp_pts_mask=torch.arange(4)))
    original = sample.gt_pts_seg.sp_pts_mask
    def failing_predict(self, inputs, samples, **kwargs):
        samples[0].gt_pts_seg.sp_pts_mask += 100
        raise RuntimeError('inference failed')
    monkeypatch.setattr(UniDet3D, 'predict', failing_predict)
    model = object.__new__(host.LDMRUniDet3D)
    with pytest.raises(RuntimeError, match='inference failed'):
        model.predict({}, [sample])
    assert sample.gt_pts_seg.sp_pts_mask is original
    assert torch.equal(original, torch.arange(4))


@pytest.mark.parametrize('raises', [False, True])
def test_loss_preserves_samples_including_upstream_failure(monkeypatch, raises):
    from mmengine.structures import BaseDataElement, InstanceData
    from unidet3d.unidet3d import UniDet3D
    sample = BaseDataElement(
        gt_pts_seg=BaseDataElement(sp_pts_mask=torch.arange(4)),
        gt_instances_3d=InstanceData(bboxes_3d=torch.ones(1, 6)))
    original_groups = sample.gt_pts_seg.sp_pts_mask
    original_targets = sample.gt_instances_3d
    def mutating_loss(self, inputs, samples, **kwargs):
        samples[0].gt_pts_seg.sp_pts_mask += 100
        samples[0].gt_instances_3d.bboxes_3d.zero_()
        samples[0].gt_instances_3d.query_masks = torch.ones(1, 4, dtype=torch.bool)
        if raises:
            raise RuntimeError('loss failed')
        return {'loss': inputs['parameter'].square()}
    monkeypatch.setattr(UniDet3D, 'loss', mutating_loss)
    model = object.__new__(host.LDMRUniDet3D)
    parameter = torch.tensor(2., requires_grad=True)
    if raises:
        with pytest.raises(RuntimeError, match='loss failed'):
            model.loss({'parameter': parameter}, [sample])
    else:
        model.loss({'parameter': parameter}, [sample])['loss'].backward()
        assert parameter.grad.item() == 4
    assert sample.gt_pts_seg.sp_pts_mask is original_groups
    assert torch.equal(original_groups, torch.arange(4))
    assert sample.gt_instances_3d is original_targets
    assert torch.equal(original_targets.bboxes_3d, torch.ones(1, 6))
    assert 'query_masks' not in original_targets


def test_postprocessing_preserves_upstream_when_topk_fits():
    from types import SimpleNamespace
    from mmengine.config import ConfigDict
    from unidet3d.unidet3d import UniDet3D
    proxy = SimpleNamespace(
        test_cfg=ConfigDict(topk_insts=20, iou_thr=[.5]),
        decoder=SimpleNamespace(datasets=['scannet']), fast_nms=[True],
        use_superpoints=[False],
        _single_scene_multiclass_nms=lambda boxes, scores, labels, *_: (boxes, scores, labels))
    generator = torch.Generator().manual_seed(31)
    out = dict(cls_preds=[torch.randn(10, 5, generator=generator)],
               bboxes=[torch.rand(10, 6, generator=generator)])
    args = (out, [torch.arange(10)], [torch.zeros(10, 3)], ['scannet'])
    expected = UniDet3D.predict_by_feat(proxy, *args)[0]
    actual = host.LDMRUniDet3D.predict_by_feat(proxy, *args)[0]
    assert torch.equal(actual[0].tensor, expected[0].tensor)
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])


@pytest.fixture(params=['none', 'scene', 'object'])
def config(tmp_path, request):
    mode = request.param
    protocol = ScanNet35Protocol()
    rng = np.random.RandomState(23)
    # Three components; source groups are fixtures, not generated superpoints.
    points = rng.uniform([-4, -4, 0, 0, 0, 0], [4, 4, 4, 255, 255, 255], (2048, 6)).astype('float32')
    raw = np.full(2048, 1, dtype=np.int64)
    raw[:512] = 18
    raw[512:1024] = 5
    instances = np.zeros(2048, dtype=np.int64)
    instances[:512] = 3
    instances[512:1024] = 7
    points[:512, :3] = rng.uniform([-3, -3, 0], [-2, -2, 1], (512, 3))
    points[512:1024, :3] = rng.uniform([2, 2, 2], [3, 3, 3], (512, 3))
    sp = (np.arange(2048, dtype=np.int64) // 8) * 3 + 7
    for name, value in [('points', points), ('semantic', raw), ('instance', instances), ('superpoints', sp)]:
        value.tofile(tmp_path / f'{name}.bin')
    info = dict(point_cloud=dict(lidar_idx='fixture', num_features=6),
                pts_path='points.bin', pts_semantic_mask_path='semantic.bin',
                pts_instance_mask_path='instance.bin', annos=dict(
                    **{'class':np.array([18, 5])}, axis_align_matrix=np.eye(4),
                    gt_boxes_upright_depth=np.array([[-2.5, -2.5, .5, 1, 1, 1], [2.5, 2.5, 2.5, 1, 1, 1]])))
    record = convert_info(info, protocol, data_root=tmp_path,
                          superpoint_path=tmp_path/'superpoints.bin', box_label_space='nyu40')
    for split in ('train', 'val'):
        records = stage_records([record], protocol, 2, evaluation=(split == 'val'))
        if split == 'val':
            # Independent fixture files/IDs exercise the split guard. These
            # synthetic copies are not a real-data generalization experiment.
            records = copy.deepcopy(records)
            records[0]['sample_idx'] = 'validation_fixture'
            shutil.copyfile(tmp_path / 'points.bin', tmp_path / 'val_points.bin')
            records[0]['lidar_points']['lidar_path'] = str(tmp_path / 'val_points.bin')
        if mode == 'scene' and split == 'train':
            records[0]['instances'] = record['instances']
            records[0].update(is_replay=True, ldmr_replay_from_stage=1,
                              ldmr_include_current=True)
        with (tmp_path/f'{split}.pkl').open('wb') as f:
            pickle.dump(dict(metainfo=dict(classes=protocol.decoder_classes(2),
                ldmr_protocol_id=protocol.identity, ldmr_stage=2, ldmr_split=split,
                ldmr_memory_mode=mode), data_list=records), f)
    crop = extract_crop(points, raw, instances, sp, bbox=[2.5, 2.5, 2.5, 1, 1, 1],
        instance_id=7, class_id=0, scene_id='fixture_source', object_idx=1, stage=1,
        protocol=protocol, min_points=20)
    save_crops(tmp_path/'crops.pkl', [crop], protocol, 1)
    root = Path(os.environ['UNIDET3D_ROOT'])
    base = Config.fromfile(str(root/'configs/unidet3d_1xb8_scannet.py'))
    cfg = stage_config(base.to_dict(), protocol, 2, train_info=tmp_path/'train.pkl',
        val_info=tmp_path/'val.pkl', work_dir=tmp_path/'run', memory_mode=mode, epochs=1,
        crop_path=tmp_path/'crops.pkl')
    # Deliberately exceed the fixture's query/class pairs to exercise the cap.
    cfg['model']['test_cfg']['topk_insts'] = 10000
    init_default_scope('mmdet3d')
    return cfg


def test_host_registered_pipeline_masks_and_geometry(config):
    np.random.seed(7)
    dataset = DATASETS.build(config['train_dataloader']['dataset'])
    sample = dataset[0]
    data = sample['data_samples']
    labels = data.gt_instances_3d.labels_3d
    mode = config['ldmr_protocol']['memory_mode']
    expected = [7] if mode == 'none' else [0, 7]
    assert sorted(labels.tolist()) == expected
    points = sample['inputs']['points']
    inst = data.gt_pts_seg.pts_instance_mask
    sp = data.gt_pts_seg.sp_pts_mask
    assert len(points) == len(inst) == len(sp)
    assert (len(points) > 2048) if mode == 'object' else (len(points) == 2048)
    assert data.gt_instances_3d.sp_masks.shape == (len(expected), int(sp.max()) + 1)
    assert sorted(torch.unique(inst).tolist()) == [-1] + list(range(len(expected)))
    assert torch.isfinite(points).all()
    val = DATASETS.build(config['val_dataloader']['dataset'])
    val_sample = val[0]['data_samples']
    assert sorted(val_sample.eval_ann_info['gt_labels_3d'].tolist()) == [0, 7]
    assert torch.equal(torch.unique(val_sample.gt_pts_seg.sp_pts_mask), torch.arange(256))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for sparse detector')
def test_host_forward_backward_and_predict(config):
    dataset = DATASETS.build(config['train_dataloader']['dataset'])
    model = MODELS.build(config['model']).cuda()
    model.init_weights()
    model.train()
    batch = pseudo_collate([dataset[0], dataset[0]])
    inputs = model.data_preprocessor(batch, training=True)
    original_samples = copy.deepcopy(inputs['data_samples'])
    losses = model(**inputs, mode='loss')
    for sample, original in zip(inputs['data_samples'], original_samples):
        assert torch.equal(sample.gt_pts_seg.sp_pts_mask, original.gt_pts_seg.sp_pts_mask)
        assert set(sample.gt_instances_3d.keys()) == set(original.gt_instances_3d.keys())
        for key, value in original.gt_instances_3d.items():
            actual = getattr(sample.gt_instances_3d, key)
            if hasattr(value, 'tensor'):
                assert torch.equal(actual.tensor, value.tensor)
            else:
                assert torch.equal(actual, value)
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    grads = {name: parameter.grad for name, parameter in model.named_parameters()
             if parameter.grad is not None}
    assert grads and all(torch.isfinite(grad).all() for grad in grads.values())
    assert any(name.startswith('decoder.outs_cls.') for name in grads)
    assert any(name.startswith('unet.') for name in grads)
    model.eval()
    val = DATASETS.build(config['val_dataloader']['dataset'])
    with torch.no_grad():
        batch = model.data_preprocessor(pseudo_collate([val[0], val[0]]), training=False)
        original_groups = [s.gt_pts_seg.sp_pts_mask.clone() for s in batch['data_samples']]
        result = model(**batch, mode='predict')
    assert len(result) == 2
    for i, item in enumerate(result):
        assert torch.equal(item.gt_pts_seg.sp_pts_mask, original_groups[i])
        labels = item.pred_instances_3d.labels_3d
        assert torch.all((labels >= 0) & (labels < 14))
        assert torch.isfinite(item.pred_instances_3d.bboxes_3d.tensor).all()
        assert (item.pred_instances_3d.bboxes_3d.tensor[:, 3:6] > 0).all()
        assert torch.isfinite(item.pred_instances_3d.scores_3d).all()
        assert torch.equal(item.pred_instances_3d.points, batch['inputs']['points'][i])


def test_stage_cli_constructs_dataset_and_model_in_fresh_process(config):
    tool = Path(__file__).resolve().parents[1] / 'tools/train_stage.py'
    train = config['train_dataloader']['dataset']['datasets'][0]['ann_file']
    val = config['val_dataloader']['dataset']['datasets'][0]['ann_file']
    mode = config['ldmr_protocol']['memory_mode']
    command = [sys.executable, str(tool), '--upstream-root', os.environ['UNIDET3D_ROOT'],
               '--train-info', train, '--val-info', val, '--work-dir', config['work_dir'],
               '--stage', '2', '--memory-mode', mode, '--validate-only']
    if mode == 'object':
        command += ['--crop-path', str(Path(train).parent / 'crops.pkl')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'model built: 14 seen classes' in result.stdout


def test_spawn_workers_preserve_seeded_replay_and_aligned_masks(config):
    """Two CPU loader workers exercise serialization and per-worker replay RNG."""
    from mmengine.runner import Runner
    cfg = copy.deepcopy(config['train_dataloader'])
    # Repeat the synthetic record so each worker processes a sample. This is
    # a worker/pipeline test, not independent training scenes or accuracy data.
    cfg['dataset'] = dict(type='RepeatDataset', times=2, dataset=cfg['dataset'])
    cfg.update(batch_size=1, num_workers=2, persistent_workers=False,
               multiprocessing_context='spawn', drop_last=False)
    observations = []
    for _ in range(2):
        torch.manual_seed(29)
        loader = Runner.build_dataloader(cfg, seed=29)
        batches = list(loader)  # exhaust iterator and shut down nonpersistent workers
        assert len(batches) == 2
        current = []
        for batch in batches:
            points = batch['inputs']['points'][0]
            sample = batch['data_samples'][0]
            labels = sample.gt_instances_3d.labels_3d
            inst = sample.gt_pts_seg.pts_instance_mask
            sp = sample.gt_pts_seg.sp_pts_mask
            assert len(points) == len(inst) == len(sp)
            assert torch.isfinite(points).all()
            assert sample.gt_instances_3d.sp_masks.shape == (len(labels), int(sp.max()) + 1)
            assert 7 in labels.tolist()
            assert all(label in (0, 7) for label in labels.tolist())
            if config['ldmr_protocol']['memory_mode'] == 'object':
                assert 0 in labels.tolist() and len(points) > 2048
            current.append((points.clone(), labels.clone(), inst.clone(), sp.clone()))
        observations.append(current)
    for first, second in zip(*observations):
        assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_explicit_box_supervision_survives_hidden_mask_changes(config):
    """Prospective host route only: no teacher generation or production switch."""
    from unidet3d.structures import InstanceData_ as InstanceData
    np.random.seed(7)
    dataset_cfg = copy.deepcopy(config['train_dataloader']['dataset'])
    for child in dataset_cfg['datasets']:
        child['pipeline'] = [step for step in child['pipeline'] if step['type'] != 'ElasticTransfrom']
        for step in child['pipeline']:
            if step['type'] == 'Pack3DDetInputs_':
                step['keys'] = list(step['keys']) + ['gt_bboxes_3d']
    dataset = DATASETS.build(dataset_cfg)
    model = MODELS.build(config['model']).cuda()
    model.init_weights()
    model.train()
    model.bbox_by_mask = [False]
    model.target_by_distance = [True]
    model.query_thr = 1000000
    inputs = model.data_preprocessor(pseudo_collate([dataset[0]]), training=True)
    assert 'elastic_coords' not in inputs['inputs']
    sample = inputs['data_samples'][0]
    xyz = inputs['inputs']['points'][0][:, :3]
    old = sample.gt_instances_3d
    # A synthetic teacher box depends only on visible geometry, never hidden GT.
    minimum, maximum = xyz.min(0).values, xyz.max(0).values
    size = maximum - minimum
    gravity = (minimum + maximum) / 2
    box = old.bboxes_3d.tensor.new_zeros((1, old.bboxes_3d.tensor.shape[1]))
    box[0, :6] = torch.cat([gravity, size])
    box[:, 2] -= size[2] / 2
    boxes = old.bboxes_3d.new_box(torch.cat([old.bboxes_3d.tensor, box], dim=0))
    labels = torch.cat([old.labels_3d, old.labels_3d.new_tensor([0])])
    sample.gt_instances_3d = InstanceData(
        labels_3d=labels, bboxes_3d=boxes,
        sp_masks=torch.zeros(len(labels), old.sp_masks.shape[1], dtype=torch.bool, device='cuda'))
    expected_center = boxes.gravity_center - minimum
    expected_size = boxes.tensor[:, 3:6].clone()
    criterion = model.criterion
    captured = []
    def inspect(outputs, instances, datasets):
        inst = instances[0]
        assert torch.allclose(inst.bboxes_3d.gravity_center, expected_center)
        assert torch.equal(inst.bboxes_3d.tensor[:, 3:6], expected_size)
        assert torch.equal(inst.labels_3d, labels)
        assert inst.query_masks.shape[0] == len(labels)
        assert inst.query_masks[-1].any()
        captured.append(inst.query_masks.detach().clone())
        return criterion(outputs, instances, datasets)
    model.criterion = inspect
    for ignored in (-1, 123):
        # Corrupt hidden semantics/instances deliberately: this route must not
        # use them to reconstruct teacher boxes or query assignments.
        sample.gt_pts_seg.pts_instance_mask.fill_(ignored)
        sample.gt_pts_seg.pts_semantic_mask.fill_(ignored)
        loss = sum(model(**inputs, mode='loss').values())
        assert torch.isfinite(loss)
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert len(captured) == 2 and torch.equal(captured[0], captured[1])
