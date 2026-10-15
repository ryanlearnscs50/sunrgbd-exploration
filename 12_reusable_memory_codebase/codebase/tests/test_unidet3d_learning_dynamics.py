"""Numerical LD bridge and malformed-record regressions without CUDA kernels."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.learning_dynamics import measure_scene, reference_functions


class Boxes:
    """CPU AABB container implementing the box API used by the shared matcher."""
    def __init__(self, values, box_dim=6, origin=(.5, .5, .5), **kwargs):
        self.tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1, box_dim).clone()
        self.tensor[:, :3] += self.tensor[:, 3:6] * (torch.tensor([.5, .5, .5]) - torch.tensor(origin))

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, index):
        return Boxes(self.tensor[index])

    def convert_to(self, mode):
        return self

    @staticmethod
    def overlaps(a, b):
        low = torch.maximum(a.tensor[:, None, :3] - a.tensor[:, None, 3:6]/2,
                            b.tensor[None, :, :3] - b.tensor[None, :, 3:6]/2)
        high = torch.minimum(a.tensor[:, None, :3] + a.tensor[:, None, 3:6]/2,
                             b.tensor[None, :, :3] + b.tensor[None, :, 3:6]/2)
        inter = (high-low).clamp(min=0).prod(-1)
        return inter / (a.tensor[:, 3:6].prod(-1)[:, None] + b.tensor[:, 3:6].prod(-1)[None, :] - inter)


BOX = [0, 0, 0, 2, 2, 2]


def measure(prediction, boxes=None, labels=None, **kwargs):
    return measure_scene(prediction, [BOX] if boxes is None else boxes,
        [0] if labels is None else labels, protocol=ScanNet35Protocol(),
        stage=2, replay_from_stage=1, scene_id='scene', box_type_3d=Boxes,
        box_mode_3d=None, **kwargs)


@pytest.mark.parametrize('metric,quality', [('f1', 2/3), ('recall', 1)])
def test_duplicate_prediction_and_empty_classes(metric, quality):
    result = measure(dict(boxes=[BOX, BOX], labels=[0, 0], scores=[.9, .8]), q_metric=metric)
    row = result['classes'][0]
    assert (row['tp'], row['fp'], row['fn']) == (1, 1, 0)
    assert row['q'] == pytest.approx(quality)
    assert result['classes'][1]['q'] is None
    assert result['save_stage'] == 1


def test_empty_predictions_are_forgetting_and_use_reference_review_weights():
    before = measure(dict(boxes=[BOX], labels=[0], scores=[.9]))
    after = measure(dict(boxes=[], labels=[], scores=[]))
    assert after['classes'][0]['fn'] == 1
    scorer = reference_functions()['scoring']
    weights = scorer['compute_reviewing_entry_weights_ld_drop'](
        [before], [after], old_classes=list(range(7)), iou_mode='0.25', q_metric='recall',
        eta=5, w_entry_max=10)
    assert weights['weights_by_uid']['scene_stage1'] == pytest.approx(6)
    design = scorer['compute_learning_dynamics_design1_scores'](
        {0: [before], 1: [after]}, class_ids=list(range(7)), q_metric='recall', design_version=2)
    assert design['class_need'][0] > 0


def test_iou_threshold_is_inclusive():
    # A centered half-size-in-XY prediction occupies exactly one quarter of GT.
    result = measure(dict(boxes=[[0, 0, 0, 1, 1, 2]], labels=[0], scores=[.9]))
    assert result['classes'][0]['tp'] == 1


def test_hidden_gt_is_rejected():
    with pytest.raises(ValueError, match='stage-visible'):
        measure(dict(boxes=[], labels=[], scores=[]), labels=[7])


@pytest.mark.parametrize('mutation', ['labels', 'count', 'missing_boxes', 'scores', 'nan', 'negative', 'label_float', 'eps', 'missing_scores'])
def test_shared_matcher_rejects_corruption(mutation):
    gt = dict(gt_num=1, gt_boxes_upright_depth=np.array([BOX]), **{'class': np.array([0])})
    dt = dict(boxes_3d=Boxes([BOX]), labels_3d=np.array([0]), scores_3d=np.array([.9]))
    eps = 1e-9
    if mutation == 'labels': gt['class'] = np.array([0, 1])
    if mutation == 'count': gt['gt_num'] = 0
    if mutation == 'missing_boxes': del dt['boxes_3d']
    if mutation == 'scores': dt['scores_3d'] = np.array([])
    if mutation == 'nan': dt['scores_3d'][0] = np.nan
    if mutation == 'negative': dt['boxes_3d'].tensor[0, 3] = -1
    if mutation == 'label_float': dt['labels_3d'] = np.array([.5])
    if mutation == 'eps': eps = float('nan')
    if mutation == 'missing_scores': del dt['scores_3d']
    with pytest.raises(ValueError):
        reference_functions()['match']([gt], [dt], seat_keys=[('scene', 1)],
            eval_class_indices=[0], iou_thr=.25, box_type_3d=Boxes, box_mode_3d=None, eps=eps)


def test_shared_matcher_accepts_explicit_empty_lists():
    result = reference_functions()['match'](
        [dict(gt_num=0, gt_boxes_upright_depth=[], **{'class': []})],
        [dict(labels_3d=[], scores_3d=[])], seat_keys=[('empty', 1)],
        eval_class_indices=[0], iou_thr=.25, box_type_3d=Boxes, box_mode_3d=None)
    assert result[0]['classes'][0]['q'] is None
    assert result[0]['classes'][0]['dt_count'] == 0


def empty_match(keys, classes=(0,)):
    return reference_functions()['match'](
        [dict(gt_num=0, gt_boxes_upright_depth=[], **{'class': []}) for _ in keys],
        [dict(labels_3d=[], scores_3d=[]) for _ in keys], seat_keys=keys,
        eval_class_indices=classes, iou_thr=.25, box_type_3d=Boxes, box_mode_3d=None)


@pytest.mark.parametrize('key', [('scene', 1.9), ('scene', True),
    ('scene', 0), ('scene', -1), ('', 1), (None, 1),
    {'scene_id': 'scene', 'save_stage': 1.5}, 'scene_stage01', '_stage1'])
def test_matcher_rejects_lossy_or_invalid_seat_identity(key):
    with pytest.raises(ValueError, match='seat'):
        empty_match([key])


def test_matcher_rejects_duplicate_seat_across_representations():
    with pytest.raises(ValueError, match='uplicate'):
        empty_match([('scene', 1), 'scene_stage1'])


@pytest.mark.parametrize('classes', [[False], [0.9], [-1], ['0'], [0, 0]])
def test_matcher_rejects_lossy_or_duplicate_class_scope(classes):
    with pytest.raises(ValueError, match='class'):
        empty_match([('scene', 1)], classes)


def test_matcher_preserves_canonical_legacy_seats_and_numpy_indices():
    result = empty_match(['scene_stage1', ('scene', np.int64(2)),
                          {'scene_id': 'second', 'save_stage': 1}], [np.int64(0)])
    assert [(row['scene_id'], row['save_stage']) for row in result] == [
        ('scene', 1), ('scene', 2), ('second', 1)]
