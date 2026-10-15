import copy
from pathlib import Path
import pickle
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol, stage_targets
from ldmr_unidet3d.objects import (
    extract_crop, paste_crop, validate_crop, selected_payloads, save_crops, load_crops)


def source():
    # A chair at z=3 and one in-box point from a future-class instance.
    xyz = [[-.4, -.4, 2.6], [.4, .4, 3.4], [0, 0, 3], [0, 0, 2.8], [8, 8, 0]]
    pts = np.column_stack([xyz, np.full((5, 3), 127)]).astype('float32')
    return pts, np.array([5, 5, 5, 18, 1]), np.array([7, 7, 7, 9, 0]), np.array([20, 20, 30, 30, 40])


def crop(**kwargs):
    args = dict(bbox=np.array([0, 0, 3, 1, 1, 1]), instance_id=7, class_id=0,
                scene_id='chair_scene', object_idx=2, stage=1,
                protocol=ScanNet35Protocol(), min_points=1, floor_percentile=0)
    args.update(kwargs)
    return extract_crop(*source(), **args)


def destination():
    return (np.array([[10, 10, 0, 100, 100, 100], [10, 10, 1, 80, 80, 80]], dtype='float32'),
            np.array([5, 18]), np.array([9999, 20000]), np.array([777, 888]))


def test_crop_matches_ldmr_geometry_keeps_context_unlabeled():
    from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank
    c = crop()
    expected = ObjectMemoryBank.extract_object_points(source()[0], c['bbox'])
    np.testing.assert_array_equal(c['points'], expected)
    assert c['pts_semantic_mask'].tolist() == [5, 5, 5, 0]
    assert c['pts_instance_mask'].tolist() == [0, 0, 0, -1]
    assert c['source_floor_offset'] == 2.5
    assert c['sp_pts_mask'].tolist() == [0, 0, 1, 1]
    assert crop(min_points=5) is None


def test_paste_disjoint_ids_elevation_no_future_leakage_or_mutation():
    p = ScanNet35Protocol()
    arrays = destination()
    saved = [a.copy() for a in arrays]
    c = crop()
    out = paste_crop(*arrays, c, center=[0, 0, 3], occupied_boxes=np.empty((0, 6)),
                     protocol=p, stage=2)
    assert out['points'].shape == (6, 6)
    assert out['pts_instance_mask'].tolist() == [0, 1, 2, 2, 2, -1]
    assert out['sp_pts_mask'].tolist() == [0, 1, 2, 2, 3, 3]
    assert out['ldmr_inserted_mask'].tolist() == [False, False, True, True, True, False]
    target = stage_targets(out['pts_semantic_mask'], out['pts_instance_mask'],
                           out['sp_pts_mask'], p, 2, inserted_mask=out['ldmr_inserted_mask'])
    assert target['gt_labels_3d'].tolist() == [7, 0]  # current pillow, pasted chair
    assert target['pts_instance_mask'][0] == -1  # natural old chair stays hidden
    np.testing.assert_array_equal(out['points'][2:, :3], source()[0][:4, :3])
    for left, right in zip(arrays, saved):
        np.testing.assert_array_equal(left, right)
    # A second paste also gets fresh IDs, even if its source superpoints repeat.
    twice = paste_crop(out['points'], out['pts_semantic_mask'], out['pts_instance_mask'],
                       out['sp_pts_mask'], c, center=[3, 0, 3],
                       occupied_boxes=out['placed_box'][None], protocol=p, stage=2,
                       inserted_mask=out['ldmr_inserted_mask'])
    assert twice['pts_instance_mask'][-4:].tolist() == [3, 3, 3, -1]
    assert twice['ldmr_inserted_mask'].sum() == 6


def test_collision_and_invalid_replay():
    p = ScanNet35Protocol()
    c = crop()
    assert paste_crop(*destination(), c, center=[0, 0, 3],
                      occupied_boxes=np.array([[0, 0, 3, 1, 1, 1]]), protocol=p, stage=2) is None
    with pytest.raises(ValueError, match='prior-stage'):
        paste_crop(*destination(), c, center=[0, 0, 3], occupied_boxes=np.empty((0, 6)),
                   protocol=p, stage=1)
    with pytest.raises(ValueError, match='axis-aligned'):
        crop(bbox=[0, 0, 3, 1, 1, 1, .2])
    with pytest.raises(ValueError, match='declared class'):
        crop(instance_id=9)


@pytest.mark.parametrize('mutation', ['future', 'context', 'shape', 'frame', 'weight'])
def test_crop_validation(mutation):
    p = ScanNet35Protocol()
    c = crop()
    if mutation == 'future': c['stage_id'] = 2
    if mutation == 'context': c['pts_semantic_mask'][-1] = 18
    if mutation == 'shape': c['sp_pts_mask'] = c['sp_pts_mask'][:-1]
    if mutation == 'frame': c['coordinate_frame'] = 'bottom_center'
    if mutation == 'weight': c['replay_weight'] = float('nan')
    with pytest.raises(ValueError): validate_crop(c, p)


def test_existing_bank_selection_export_and_roundtrip(tmp_path):
    from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank
    p = ScanNet35Protocol()
    candidates = [crop(scene_id=f's{i}', object_idx=i) for i in range(8)]
    bank = ObjectMemoryBank(exemplars_per_class=2, max_total_exemplars=2,
                            selection_strategy='random', min_points=1)
    bank.add_exemplars(0, candidates, stage_id=1)
    selected = selected_payloads(bank, candidates, p)
    assert len(selected) == 2
    assert [x['object_idx'] for x in selected] == [x['object_idx'] for x in bank.get_all_exemplars()]
    path = tmp_path / 'crops.pkl'
    save_crops(path, selected, p, 1)
    restored = load_crops(path, p, 1)
    for a, b in zip(selected, restored):
        for key in ('points', 'pts_semantic_mask', 'pts_instance_mask', 'sp_pts_mask'):
            np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(ValueError, match='stage mismatch'): load_crops(path, p, 2)
    corrupted = copy.deepcopy(selected)
    corrupted[0]['pts_semantic_mask'][-1] = 18
    before = path.read_bytes()
    with pytest.raises(ValueError): save_crops(path, corrupted, p, 1)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='missing masks'): selected_payloads(bank, [], p)


def test_inserted_flags_cannot_expose_future_or_alias_natural_instances():
    p = ScanNet35Protocol()
    with pytest.raises(ValueError, match='previous classes'):
        stage_targets(np.array([34]), np.array([0]), np.array([0]), p, 2,
                      inserted_mask=np.array([True]))
    with pytest.raises(ValueError, match='share an instance'):
        stage_targets(np.array([5, 5]), np.array([0, 0]), np.array([0, 0]), p, 2,
                      inserted_mask=np.array([False, True]))
