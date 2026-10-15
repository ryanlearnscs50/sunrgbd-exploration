import copy
import os
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.data import convert_info, stage_records, verify_point_files


def test_mesh_superpoints_require_exact_row_identity():
    from ldmr_unidet3d.data import verify_mesh_rows
    points = np.array([[0, 0, 0, 1, 2, 3], [1, 0, 0, 1, 2, 3],
                       [0, 1, 0, 1, 2, 3]], dtype=np.float32)
    vertices = points[:, :3].copy()
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    verify_mesh_rows(points, vertices, faces)
    for bad in (vertices[::-1], vertices[:2], vertices + .001):
        with pytest.raises(ValueError, match='XYZ row'):
            verify_mesh_rows(points, bad, faces)
    for bad in (np.array([[0, 1, 3]]), np.array([[-1, 1, 2]]), faces.astype(float)):
        with pytest.raises(ValueError, match='triangle'):
            verify_mesh_rows(points, vertices, bad)


def legacy():
    return dict(point_cloud=dict(lidar_idx='scene', num_features=6),
                pts_path='points.bin', pts_instance_mask_path='instance.bin',
                pts_semantic_mask_path='semantic.bin', annos=dict(
                    **{'class': np.array([4, 17, 33, 0])},
                    gt_boxes_upright_depth=np.tile([0, 0, 1, 1, 1, 1], (4, 1)),
                    axis_align_matrix=np.eye(4)))


def converted(tmp_path):
    return convert_info(legacy(), ScanNet35Protocol(), data_root=tmp_path,
                        superpoint_path=tmp_path/'superpoints.bin',
                        box_label_space='nyu40_zero_based')


def test_conversion_label_offset_boxes_alignment_and_no_mutation(tmp_path):
    info = legacy()
    before = copy.deepcopy(info)
    out = converted(tmp_path)
    assert [x['bbox_label_3d'] for x in out['instances']] == [0, 7, 14]
    assert out['instances'][0]['bbox_3d'] == [0, 0, 1, 1, 1, 1]
    assert out['ldmr_semantic_space'] == 'nyu40'
    np.testing.assert_array_equal(info['annos']['class'], before['annos']['class'])
    with pytest.raises(ValueError, match='explicitly'):
        convert_info(info, ScanNet35Protocol(), data_root=tmp_path,
                     superpoint_path='sp', box_label_space='auto')


def test_stage_scene_eligibility_matches_ldmr(tmp_path):
    p = ScanNet35Protocol()
    out = converted(tmp_path)
    future_only = copy.deepcopy(out)
    future_only.update(sample_idx='future', instances=[out['instances'][2]])
    records = [out, future_only]
    train = stage_records(records, p, 2)
    val = stage_records(records, p, 2, evaluation=True)
    assert len(train) == len(val) == 1
    assert [x['bbox_label_3d'] for x in train[0]['instances']] == [7]
    assert [x['bbox_label_3d'] for x in val[0]['instances']] == [0, 7]
    assert len(out['instances']) == 3
    with pytest.raises(ValueError, match='duplicate'):
        stage_records([out, out], p, 2)


def test_validate_files_and_reject_mask_misalignment(tmp_path):
    record = converted(tmp_path)
    np.zeros((2, 6), np.float32).tofile(tmp_path/'points.bin')
    np.array([5, 18], np.int64).tofile(tmp_path/'semantic.bin')
    np.array([0, 1], np.int64).tofile(tmp_path/'instance.bin')
    np.array([9, 10], np.int64).tofile(tmp_path/'superpoints.bin')
    assert verify_point_files(record) == 2
    np.array([9], np.int64).tofile(tmp_path/'superpoints.bin')
    with pytest.raises(ValueError, match='one int64'):
        verify_point_files(record)


@pytest.mark.parametrize('trailing_bytes', [1, 2, 3, 4, 23])
def test_point_file_rejects_partial_float_or_row(tmp_path, trailing_bytes):
    record = converted(tmp_path)
    np.zeros((2, 6), np.float32).tofile(tmp_path / 'points.bin')
    with (tmp_path / 'points.bin').open('ab') as stream:
        stream.write(b'\x00' * trailing_bytes)
    with pytest.raises(ValueError, match='byte length'):
        verify_point_files(record)


@pytest.mark.parametrize('alias', ['same_path', 'symlink', 'hardlink', 'same_id', 'disjoint'])
def test_split_validation_checks_scene_and_file_identity(tmp_path, alias):
    from ldmr_unidet3d.data import validate_disjoint_splits
    train = converted(tmp_path)
    np.zeros((2, 6), np.float32).tofile(tmp_path / 'points.bin')
    val = copy.deepcopy(train)
    val['sample_idx'] = 'other_scene'
    other = tmp_path / 'other.bin'
    if alias == 'symlink':
        other.symlink_to(tmp_path / 'points.bin')
    elif alias == 'hardlink':
        import os
        os.link(tmp_path / 'points.bin', other)
    else:
        np.ones((2, 6), np.float32).tofile(other)
    if alias != 'same_path':
        val['lidar_points']['lidar_path'] = str(other)
    if alias == 'same_id':
        val['sample_idx'] = train['sample_idx']
    if alias == 'disjoint':
        validate_disjoint_splits([train], [val])
    else:
        with pytest.raises(ValueError, match='overlap'):
            validate_disjoint_splits([train], [val])


def test_host_config_modes_and_stage_visibility(tmp_path):
    from ldmr_unidet3d.configuration import stage_config
    import runpy
    upstream = Path(os.environ.get(
        'UNIDET3D_ROOT', Path(__file__).resolve().parents[2] / 'reference/UniDet3D'))
    reference = upstream / 'configs/unidet3d_1xb8_scannet.py'
    assert reference.is_file(), (
        'Set UNIDET3D_ROOT to the pinned UniDet3D checkout; missing config: '
        f'{reference}')
    base = {k: v for k, v in runpy.run_path(str(reference)).items() if not k.startswith('__')}
    p = ScanNet35Protocol()
    for mode in ('none', 'scene', 'object'):
        cfg = stage_config(base, p, 2, train_info='train.pkl', val_info='val.pkl',
                           work_dir=tmp_path, memory_mode=mode, epochs=12, crop_path='crops.pkl')
        assert cfg['model']['decoder']['datasets_classes'] == [list(p.names[:14])]
        assert cfg['model']['criterion']['matcher']['type'] == 'LDMRUniMatcher'
        assert cfg['load_from'] is None
        pipeline = cfg['train_dataloader']['dataset']['datasets'][0]['pipeline']
        types = [t['type'] for t in pipeline]
        assert 'PointSegClassMapping' not in types
        assert 'PointDetClassMappingScanNet' not in types
        assert types.index('LDMRScanNetTargets') < types.index('Pack3DDetInputs_')
        if mode == 'object':
            assert types.index('GlobalAlignment') < types.index('LDMRInsertObjects') < types.index('RandomFlip3D')
        else:
            assert 'LDMRInsertObjects' not in types
    assert base['model']['type'] == 'UniDet3D'


@pytest.mark.parametrize('recipe,mode', [('ldmr', 'scene'), ('tr3d_obj', 'object')])
@pytest.mark.parametrize('validation_interval', [None, 30])
def test_reference_recipe_preserves_exposure_and_nominal_validation(tmp_path, recipe, mode, validation_interval):
    from types import SimpleNamespace
    import runpy
    from ldmr_unidet3d.recipes import apply_reference_recipe
    from ldmr_unidet3d.configuration import stage_config
    from ldmr_unidet3d.reviewing import review_boundaries
    upstream = Path(os.environ.get(
        'UNIDET3D_ROOT', Path(__file__).resolve().parents[2] / 'reference/UniDet3D'))
    base = {key: value for key, value in runpy.run_path(
        str(upstream / 'configs/unidet3d_1xb8_scannet.py')).items()
        if not key.startswith('__')}
    args = SimpleNamespace(recipe=recipe, memory_mode=mode,
        stage_setting='scannet35_s5_freqorder', epochs=12, batch_size=8,
        ld_epochs=None, reviewing=False, object_policy='ldmr_weighted',
        validation_interval=validation_interval)
    apply_reference_recipe(args)
    cfg = stage_config(base, ScanNet35Protocol(), 1,
        train_info='train.pkl', val_info='val.pkl', work_dir=tmp_path,
        memory_mode=mode, epochs=args.epochs, target_mode=args.target_mode,
        object_policy=args.object_policy, validation_interval=args.validation_interval)
    assert cfg['train_cfg']['max_epochs'] == 180
    assert cfg['train_cfg']['val_interval'] == (validation_interval or 15)
    assert cfg['default_hooks']['checkpoint']['interval'] == (validation_interval or 15)
    assert cfg['default_hooks']['checkpoint']['save_last']
    assert cfg['param_scheduler']['end'] == 180
    assert cfg['optim_wrapper'] == base['optim_wrapper']
    assert args.pseudo and args.target_mode == 'boxes'
    assert args.reviewing == (mode == 'scene')
    if mode == 'scene':
        assert review_boundaries(args.epochs) == [0, 36, 72, 108, 144, 180]
    else:
        assert args.objects_per_class == 100 and args.max_objects == 3500
        from ldmr_unidet3d.recipes import tr3d_obj_settings
        bank, pseudo = tr3d_obj_settings()
        from ldmr_unidet3d.pseudo import scannet_policy
        for stage in range(2, 6):
            policy = scannet_policy(stage, policy_name='tr3d_obj')
            assert all(value == pseudo[key] for key, value in policy.items())
        next_cfg = stage_config(base, ScanNet35Protocol(), 2,
            train_info='train.pkl', val_info='val.pkl', work_dir=tmp_path,
            memory_mode=mode, epochs=args.epochs, target_mode=args.target_mode,
            object_policy=args.object_policy, crop_path='crops.pkl')
        pipeline = next_cfg['train_dataloader']['dataset']['datasets'][0]['pipeline']
        insert = next(step for step in pipeline if step['type'] == 'LDMRInsertObjects')
        assert insert['max_exemplars'] == bank['insertion']['max_exemplars_per_scene']
        assert insert['probability'] == bank['insertion']['insertion_probability']
        assert insert['sampling_policy'] == bank['insertion']['sampling_policy']
        assert insert['collision_iou'] == bank['insertion']['collision_threshold']


def test_model_state_expansion_rejects_unrelated_checkpoint():
    import torch
    from ldmr_unidet3d.checkpoints import expand_checkpoint
    p = ScanNet35Protocol()
    old = {'decoder.outs_cls.2.weight':torch.arange(24).reshape(8, 3).float(),
           'decoder.outs_cls.2.bias':torch.arange(8).float(), 'backbone.test':torch.ones(2)}
    new = {'decoder.outs_cls.2.weight':torch.zeros(15, 3),
           'decoder.outs_cls.2.bias':torch.zeros(15), 'backbone.test':torch.zeros(2)}
    previous = dict(state_dict=old, meta=dict(ldmr_protocol_id=p.identity, ldmr_stage=1))
    result = expand_checkpoint(previous, new, p, 2)
    assert set(result) == {'state_dict', 'meta'}  # no old optimizer carried over
    assert torch.equal(result['state_dict']['backbone.test'], old['backbone.test'])
    assert torch.equal(result['state_dict']['decoder.outs_cls.2.weight'][-1], old['decoder.outs_cls.2.weight'][-1])
    with pytest.raises(ValueError, match='protocol'):
        expand_checkpoint(dict(state_dict=old, meta={}), new, p, 2)


@pytest.mark.parametrize('damage', [None, 'stage', 'protocol', 'shape', 'dtype', 'nan', 'missing'])
def test_evaluation_checkpoint_validation_rejects_silent_misloads(damage):
    import torch
    from ldmr_unidet3d.checkpoints import validate_checkpoint_state
    protocol = ScanNet35Protocol()
    expected = {'weight': torch.ones(3)}
    checkpoint = dict(state_dict={'weight': torch.ones(3)},
                      meta=dict(ldmr_protocol_id=protocol.identity, ldmr_stage=2))
    if damage == 'stage':
        checkpoint['meta']['ldmr_stage'] = 1
    elif damage == 'protocol':
        checkpoint['meta']['ldmr_protocol_id'] = 'other'
    elif damage == 'shape':
        checkpoint['state_dict']['weight'] = torch.ones(4)
    elif damage == 'dtype':
        checkpoint['state_dict']['weight'] = torch.ones(3, dtype=torch.float64)
    elif damage == 'nan':
        checkpoint['state_dict']['weight'][0] = float('nan')
    elif damage == 'missing':
        checkpoint['state_dict'] = {}
    if damage is None:
        validate_checkpoint_state(checkpoint, expected, protocol, 2)
    else:
        with pytest.raises(ValueError):
            validate_checkpoint_state(checkpoint, expected, protocol, 2)


def test_merge_labels_preserves_ldmr_default_without_exposing_middle_stages(tmp_path):
    from ldmr_unidet3d.scene import merge_replay_records
    from ldmr_unidet3d import stage_targets
    p = ScanNet35Protocol()
    record = converted(tmp_path)
    natural = stage_records([record], p, 3)[0]
    replay = copy.deepcopy(record)
    replay.update(is_replay=True, ldmr_replay_from_stage=1,
                  instances=[record['instances'][0]])
    merged = merge_replay_records([natural], [replay])
    assert len(merged) == 1
    assert sorted(i['bbox_label_3d'] for i in merged[0]['instances']) == [0, 14]
    assert merged[0]['ldmr_include_current'] is True
    target = stage_targets(np.array([5, 18, 34]), np.array([1, 2, 3]), np.array([0, 1, 2]),
                           p, 3, replay_from_stage=1, include_current=True)
    assert target['gt_labels_3d'].tolist() == [0, 14]  # stage-2 pillow remains hidden
    assert len(merge_replay_records([natural], [replay], 'keep_both')) == 2
    with pytest.raises(ValueError, match='unique-scene'):
        merge_replay_records([natural], [replay, replay])
