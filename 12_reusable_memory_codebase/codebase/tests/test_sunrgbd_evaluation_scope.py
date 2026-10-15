import numpy as np
import pytest
import torch

from mmdet3d.core.bbox import DepthInstance3DBoxes, Box3DMode
from mmdet3d.datasets.sunrgbd_dataset import SUNRGBDDataset


def fixture(monkeypatch):
    # Exact-match IoU on CPU isolates class filtering and the actual AP reducer.
    monkeypatch.setattr(DepthInstance3DBoxes, 'overlaps', staticmethod(
        lambda a, b: torch.ones((len(a), len(b)))))
    dataset = SUNRGBDDataset.__new__(SUNRGBDDataset)
    dataset.CLASSES = ('seen', 'future')
    boxes = np.array([[0, 0, 0, 1, 1, 1, 0], [2, 2, 2, 1, 1, 1, 0]], np.float32)
    dataset.data_infos = [dict(annos=dict(gt_num=2, **{'class': np.array([0, 1])},
                                        gt_boxes_upright_depth=boxes))]
    dataset.box_type_3d = DepthInstance3DBoxes
    dataset.box_mode_3d = Box3DMode.DEPTH
    predictions = [dict(boxes_3d=DepthInstance3DBoxes(boxes[:1], origin=(.5, .5, .5)),
                        labels_3d=torch.tensor([0]), scores_3d=torch.tensor([.99]))]
    return dataset, predictions


def test_seen_scope_excludes_future_gt_from_denominator(monkeypatch):
    dataset, predictions = fixture(monkeypatch)
    full = dataset.evaluate(predictions)
    dataset.seen_classes_for_eval = [0]
    seen = dataset.evaluate(predictions)
    assert full['mAP_0.25'] == pytest.approx(.5)
    assert seen['mAP_0.25'] == pytest.approx(1.)
    assert 'future_AP_0.25' not in seen
    assert dataset.data_infos[0]['annos']['gt_num'] == 2
    dataset.seen_classes_for_eval = [0, 1]
    assert dataset.evaluate(predictions) == full


@pytest.mark.parametrize('seen', [[], [0, 0], [2], [-1], [True], [0.5]])
def test_invalid_scope_is_rejected(monkeypatch, seen):
    dataset, predictions = fixture(monkeypatch)
    dataset.seen_classes_for_eval = seen
    with pytest.raises(ValueError, match='unique valid'):
        dataset.evaluate(predictions)
