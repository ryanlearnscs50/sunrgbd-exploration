"""Host review decisions must match the original coverage-preserving policy."""
from pathlib import Path
import runpy
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.reviewing import review_boundaries, review_indices, review_weights
from ldmr_unidet3d.trajectories import make_snapshot
from ldmr_unidet3d.learning_dynamics import reference_functions


@pytest.mark.parametrize('epochs,expected', [(1, [0, 1]), (2, [0, 1, 2]),
    (5, [0, 1, 2, 3, 4, 5]), (12, [0, 2, 4, 7, 9, 12])])
def test_original_review_boundaries(epochs, expected):
    assert review_boundaries(epochs) == expected


def test_weights_use_consecutive_drop_with_saved_visibility():
    protocol = ScanNet35Protocol()
    snapshots = []
    for step, tp in enumerate([1, 0, 1]):
        seats = [dict(scene_id='memory', save_stage=1, classes={cid: dict(
            tp=tp if cid == 0 else 0, fp=0, fn=1-tp if cid == 0 else 0,
            denom=int(cid == 0), gt_count=int(cid == 0), tp_count=tp if cid == 0 else 0,
            dt_count=tp if cid == 0 else 0, count=int(cid == 0),
            q=2*tp/(tp+1+1e-9) if cid == 0 else None) for cid in range(7)})]
        snapshots.append(make_snapshot(seats, protocol=protocol, stage=2, step=step,
            checkpoint_sha256=str(step)*64, measurement_sha256='c'*64, q_metric='f1'))
    actual = review_weights(snapshots[:2], protocol)
    expected = reference_functions()['scoring']['compute_reviewing_entry_weights_ld_drop'](
        snapshots[0]['seats'], snapshots[1]['seats'], old_classes=list(range(7)),
        iou_mode='0.25', q_metric='f1', eta=5, object_count_cap=20,
        normalize_by_gt_weight=True, w_entry_max=10)
    assert actual == expected
    assert actual['weights_by_uid']['memory_stage1'] == pytest.approx(6)
    assert review_weights(snapshots, protocol)['weights_by_uid']['memory_stage1'] == 1
    snapshots[1]['seats'][0]['scene_id'] = 'missing'
    with pytest.raises(ValueError, match='coverage'):
        review_weights(snapshots[:2], protocol)


def test_sampling_preserves_merged_memory_and_reduces_natural_visits():
    rows = [dict(sample_idx=f'n{i}') for i in range(10)] + [
        dict(sample_idx='a', is_replay=True, ldmr_replay_from_stage=1),
        dict(sample_idx='b', is_replay=True, ldmr_replay_from_stage=1, ldmr_include_current=True)]
    neutral, _ = review_indices(rows, dict(a_stage1=1, b_stage1=1), seed=200, stage=2, review_k=1)
    boosted, debug = review_indices(rows, dict(a_stage1=6, b_stage1=1), seed=200, stage=2, review_k=1)
    assert len(boosted) == len(neutral) == len(rows)
    assert {10, 11}.issubset(boosted)
    assert sum(i >= 10 for i in boosted) > sum(i >= 10 for i in neutral)
    assert debug['memory_never_seen_count'] == 0
    assert review_indices(rows, dict(a_stage1=6, b_stage1=1), seed=200, stage=2, review_k=1)[0] == boosted


def test_review_seals_require_online_artifacts(tmp_path):
    from ldmr_unidet3d.sequence import stage_files
    paths = stage_files(tmp_path, 2, 5, 'scene', 5, ld_epochs=list(range(6)), reviewing=True)
    assert tmp_path / 'training/reviewing/epoch_5_sampling.json' in paths
    assert tmp_path / 'training/reviewing/review_4/snapshot_4.json' in paths
    assert tmp_path / 'training/reviewing/review_4.json' in paths


def test_object_weights_match_original_bank_inheritance():
    from ldmr_unidet3d.reviewing import object_review_weights
    from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank
    crops = [dict(scene_id=s,stage_id=t,class_id=0) for s,t in [('a',1),('a',1),('a',2),('b',1)]]
    class Proxy:
        source_seat_id = staticmethod(ObjectMemoryBank.source_seat_id)
        def get_all_exemplars(self): return crops
        def _rebuild_cache(self): pass
    weights = {'a_stage1':6.,'a_stage2':2.,'b_stage1':1.}
    ObjectMemoryBank.apply_source_seat_replay_weights(Proxy(),weights)
    assert object_review_weights(crops,weights)==[c['replay_weight'] for c in crops]==[6.,6.,2.,1.]
    with pytest.raises(ValueError,match='exactly'):
        object_review_weights(crops,dict(weights,unrelated_stage1=1.))
