"""Host measurement bridge to maintained LDMR matching and quality scoring.

No training hook or memory selection is enabled by this module. Callers supply
unaugmented, aligned gravity-center AABBs and LDMR frequency-order labels.
"""
from functools import lru_cache
from pathlib import Path
import runpy

import numpy as np

from .pseudo import prediction_arrays

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def reference_functions():
    # Load files directly: importing legacy mmdet3d would shadow the host runtime.
    return dict(match=runpy.run_path(str(ROOT / 'mmdet3d/core/evaluation/scene_class_stats.py'))[
        'compute_scene_class_match_stats'], scoring=runpy.run_path(str(
        ROOT / 'mmdet3d/utils/learning_dynamics_scoring.py')))


def measure_scene(prediction, gt_boxes, gt_labels, *, protocol, stage,
                  scene_id, box_type_3d, box_mode_3d, replay_from_stage=None,
                  iou_thr=0.25, q_metric='recall', eps=1e-9):
    """Measure current GT or saved replay visibility using the legacy matcher.

    Predictions may contain all seen classes. GT must already be filtered to
    natural current classes or the saved replay visibility; reject hidden GT.
    Empty predictions are explicit records, never missing measurements.
    """
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError('LD scene identity must be a nonempty string')
    visible = protocol.visible_classes(stage, replay_from_stage)
    boxes, labels, scores = prediction_arrays(prediction, protocol.seen_count(stage))
    gt = np.asarray(gt_boxes, dtype=np.float32)
    target = np.asarray(gt_labels)
    if gt.shape == (0,):
        gt = gt.reshape(0, 6)
    if target.shape == (0,):
        target = target.astype(np.int64)
    if (gt.ndim != 2 or gt.shape[1] != 6 or target.shape != (len(gt),)
            or target.dtype.kind not in 'iu' or not np.isfinite(gt).all()
            or np.any(gt[:, 3:] <= 0) or not np.isin(target, visible).all()):
        raise ValueError('LD GT must be finite AABBs with stage-visible integer labels')
    # Both runtimes use box containers with explicit origin conversion.
    detections = box_type_3d(boxes, box_dim=6, with_yaw=False,
                             origin=(0.5, 0.5, 0.5))
    return reference_functions()['match'](
        [dict(gt_num=len(gt), gt_boxes_upright_depth=gt, **{'class': target})],
        [dict(boxes_3d=detections, labels_3d=labels, scores_3d=scores)],
        seat_keys=[(scene_id, stage if replay_from_stage is None else replay_from_stage)],
        eval_class_indices=visible, iou_thr=iou_thr, box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d, q_metric=q_metric, eps=eps)[0]
