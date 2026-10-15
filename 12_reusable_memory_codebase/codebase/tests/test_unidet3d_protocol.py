"""Host-independent checks; not a UniDet3D runtime pass."""
import runpy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol, remap_classifier, stage_targets


@pytest.mark.parametrize('setting,sizes', [
    ('scannet35_s5_freqorder', [7] * 5),
    ('scannet35_s3_freqorder_15_10_10', [15, 10, 10]),
    ('scannet35_s10_freqorder_4444433333', [4] * 5 + [3] * 5),
])
def test_exact_ldmr_mapping_and_all_stages(setting, sizes):
    source = runpy.run_path(str(ROOT / 'configs/_base_/class_mappings/'
                               'scannet_dynamic_head_mappings.py'))
    p = ScanNet35Protocol(setting)
    assert p.stages == source['get_stage_definitions']('frequency', setting)
    assert len(p.names) == len(set(p.names)) == 35
    assert [len(s['class_indices']) for s in p.stages] == sizes
    cursor = 0
    raw = np.arange(41)
    for stage, count in enumerate(sizes, 1):
        labels = p.map_raw(raw, stage)
        assert sorted(labels[labels >= 0]) == list(range(cursor, cursor + count))
        seen = p.map_raw(raw, stage, evaluation=True)
        assert sorted(seen[seen >= 0]) == list(range(cursor + count))
        if stage > 1:
            replay = p.map_raw(raw, stage, replay_from_stage=stage - 1)
            assert sorted(replay[replay >= 0]) == list(range(cursor))
        cursor += count


@pytest.mark.parametrize('stage', [0, 6, True, 1.5])
def test_invalid_stage(stage):
    with pytest.raises(ValueError, match='stage'):
        ScanNet35Protocol().seen_count(stage)


def test_identity_and_raw_validation():
    p = ScanNet35Protocol()
    assert p.identity == ScanNet35Protocol().identity
    assert p.identity != ScanNet35Protocol('scannet35_s3_freqorder_15_10_10').identity
    with pytest.raises(ValueError, match='integer'):
        p.map_raw(np.array([5.]), 1)
    with pytest.raises(ValueError, match='precede'):
        p.map_raw(np.array([5]), 2, replay_from_stage=2)


def test_natural_and_replay_visibility():
    # Chair stage 1, pillow stage 2, sink stage 3, wall ignored.
    raw = np.array([5, 5, 18, 18, 34, 1])
    instances = np.array([10, 10, 40, 40, 80, 90])
    sp = np.array([7, 7, 50, 50, 70, 70])
    saved = [a.copy() for a in (raw, instances, sp)]
    p = ScanNet35Protocol()
    natural = stage_targets(raw, instances, sp, p, 2)
    replay = stage_targets(raw, instances, sp, p, 2, replay_from_stage=1)
    assert natural['gt_labels_3d'].tolist() == [7]
    assert natural['pts_instance_mask'].tolist() == [-1, -1, 0, 0, -1, -1]
    assert replay['gt_labels_3d'].tolist() == [0]
    assert replay['pts_instance_mask'].tolist() == [0, 0, -1, -1, -1, -1]
    assert replay['gt_sp_masks'].tolist() == [[True, False, False]]
    for a, b in zip((raw, instances, sp), saved):
        np.testing.assert_array_equal(a, b)


def test_all_visible_empty_and_majority_tie():
    p = ScanNet35Protocol()
    out = stage_targets(np.array([5, 5]), np.array([4, 4]), np.array([9, 9]), p, 1)
    assert out['pts_instance_mask'].tolist() == [0, 0]
    assert out['gt_sp_masks'].tolist() == [[True]]
    out = stage_targets(np.array([5, 1]), np.array([4, 8]), np.array([9, 9]), p, 1)
    assert out['gt_sp_masks'].tolist() == [[False]]
    out = stage_targets(np.array([1]), np.array([4]), np.array([9]), p, 1)
    assert out['gt_sp_masks'].shape == (0, 1)
    empty = np.array([], dtype=np.int64)
    out = stage_targets(empty, empty, empty, p, 1)
    assert out['gt_sp_masks'].shape == (0, 0)


@pytest.mark.parametrize('raw,inst,sp', [
    ([5, 18], [1, 1], [0, 0]),
    ([5, 5], [1], [0, 0]),
    ([5, 5], [-2, -2], [0, 0]),
    ([5, 5], [1, 1], [-1, 0]),
])
def test_invalid_annotations(raw, inst, sp):
    with pytest.raises(ValueError):
        stage_targets(*(np.array(v) for v in (raw, inst, sp)), ScanNet35Protocol(), 1)


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_head_growth_preserves_predictions(backend):
    p = ScanNet35Protocol()
    for stage in range(1, 5):
        old, new = p.decoder_classes(stage), p.decoder_classes(stage + 1)
        rng = np.random.RandomState(stage)
        ow = rng.randn(len(old) + 1, 3).astype('float32')
        ob = rng.randn(len(old) + 1).astype('float32')
        nw = rng.randn(len(new) + 1, 3).astype('float32')
        nb = rng.randn(len(new) + 1).astype('float32')
        arrays = [ow, ob, nw, nb]
        if backend == 'torch':
            arrays = [torch.tensor(a, requires_grad=True) for a in arrays]
        weight, bias = remap_classifier(arrays[0], arrays[1], old, arrays[2], arrays[3], new)
        if backend == 'torch':
            weight, bias = weight.numpy(), bias.numpy()
        features = np.array([.2, .8, -.4], dtype='float32')
        for name in old + ['no_obj']:
            src = (sorted(old) + ['no_obj']).index(name)
            dst = (sorted(new) + ['no_obj']).index(name)
            np.testing.assert_allclose(weight[dst] @ features + bias[dst],
                                       ow[src] @ features + ob[src])
        for name in set(new) - set(old):
            dst = sorted(new).index(name)
            np.testing.assert_array_equal(weight[dst], nw[dst])
            assert bias[dst] == nb[dst]


def test_classifier_rejects_incompatible_vocabulary():
    with pytest.raises(ValueError, match='retain'):
        remap_classifier(np.zeros((2, 3)), np.zeros(2), ['chair'],
                         np.zeros((2, 3)), np.zeros(2), ['table'])


def test_real_scene_bank_bridge_and_checkpoint(tmp_path):
    from mmdet3d.datasets.scene_memory_bank import SceneMemoryBank
    from ldmr_unidet3d.scene import replay_records
    p = ScanNet35Protocol()
    bank = SceneMemoryBank(max_memory_scenes=2, quota_strategy='stage_ratio',
                           stage_scene_counts={1: 4, 2: 4},
                           selection_strategy='random', random_seed=42,
                           debug_mode=False, min_objects_per_scene=1)
    infos = [dict(sample_idx=f's{i}', annos=dict(
        gt_num=2, **{'class': np.array([5, 18])},
        gt_boxes_upright_depth=np.ones((2, 6)))) for i in range(4)]
    mapping = dict(nyu40_to_model_idx=dict(zip(p.raw_ids, range(35))),
                   valid_nyu40_ids=list(p.raw_ids))
    bank.add_stage_scenes(1, infos, list(range(7)), mapping)
    host = {f's{i}': dict(ldmr_protocol_id=p.identity, instances=[
        dict(bbox_label_3d=0, bbox_3d=[0, 0, 0, 1, 1, 1]),
        dict(bbox_label_3d=7, bbox_3d=[2, 2, 2, 1, 1, 1])]) for i in range(4)}
    before = replay_records(bank, host, p, 2)
    assert len(before) == 2
    assert all(r['ldmr_replay_from_stage'] == 1 for r in before)
    assert all([i['bbox_label_3d'] for i in r['instances']] == [0] for r in before)
    assert all(len(r['instances']) == 2 for r in host.values())
    path = tmp_path / 'bank.pkl'
    bank.save_checkpoint(path, 1, dict(protocol=p.identity))
    restored = SceneMemoryBank.load_checkpoint(path, expected_context=dict(protocol=p.identity))
    assert replay_records(restored, host, p, 2) == before
    with pytest.raises(ValueError, match='missing'):
        replay_records(restored, {}, p, 2)
    for r in host.values():
        r['ldmr_protocol_id'] = 'wrong'
    with pytest.raises(ValueError, match='protocol'):
        replay_records(restored, host, p, 2)
