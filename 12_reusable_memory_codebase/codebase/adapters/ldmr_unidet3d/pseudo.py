"""Strict, host-independent teacher prediction artifacts and LDMR filtering.

Boxes are aligned ScanNet gravity-center AABBs. Labels use LDMR frequency order,
not the host classifier's internally sorted rows. No point GT is used here.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _geometry():
    # This module is NumPy-only; avoid importing either mmdet3d runtime package.
    return runpy.run_path(str(ROOT / 'mmdet3d/datasets/pseudo_label_utils.py'))


def scannet_policy(stage, stage_setting='scannet35_s5_freqorder', policy_name='ldmr'):
    """Resolve the maintained LDMR ScanNet policy, not a second set of defaults."""
    if stage_setting != 'scannet35_s5_freqorder':
        raise ValueError('host pseudo supervision currently supports only the S5 protocol')
    source = ROOT / 'configs/incremental/scannet/tr3d_dynamic_head_s5_pseudo_memory_ld_design2_reviewing.py'
    config = runpy.run_path(str(source))['pseudo_label_config']
    if type(stage) is not int or stage < 2 or stage > 5:
        raise ValueError('pseudo supervision requires a later ScanNet stage')
    if policy_name == 'tr3d_obj':
        from .recipes import tr3d_obj_settings
        config = tr3d_obj_settings()[1]
        return {key: config[key] for key in (
            'confidence_threshold', 'nms_threshold', 'max_pseudo_per_scene',
            'pseudo_vs_gt_iou_thr', 'apply_to_memory_scenes')}
    if policy_name != 'ldmr':
        raise ValueError('unknown pseudo-label policy')
    return dict(confidence_threshold=config['stage_thresholds'].get(stage, config['confidence_threshold']),
                nms_threshold=config['nms_threshold'],
                max_pseudo_per_scene=config['max_pseudo_per_scene'],
                pseudo_vs_gt_iou_thr=config['pseudo_vs_gt_iou_thr'],
                apply_to_memory_scenes=config['apply_to_memory_scenes'])


def prediction_arrays(record, old_count):
    """Reject malformed or wrong-class predictions before filtering."""
    if not isinstance(record, dict) or set(record) != {'boxes', 'labels', 'scores'}:
        raise ValueError('prediction record requires boxes, labels and scores')
    boxes = np.asarray(record['boxes'], dtype=np.float32)
    labels = np.asarray(record['labels'])
    scores = np.asarray(record['scores'], dtype=np.float32)
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 6)
    if labels.shape == (0,):
        labels = labels.astype(np.int64)
    if (boxes.ndim != 2 or boxes.shape[1] != 6 or labels.shape != (len(boxes),)
            or scores.shape != (len(boxes),) or labels.dtype.kind not in 'iu'
            or not np.isfinite(boxes).all() or np.any(boxes[:, 3:] <= 0)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
            or np.any((labels < 0) | (labels >= old_count))):
        raise ValueError('predictions must be finite positive AABBs, old-class integer labels and probability scores')
    return boxes.copy(), labels.astype(np.int64, copy=True), scores.copy()


def select_pseudo(record, gt_boxes, old_count, policy):
    """LDMR confidence -> class NMS -> cap -> visible-GT conflict filtering.

    Use the maintained IoU/NMS helpers verbatim, including their threshold/tie
    behavior. The final GT filter cannot consult hidden old/future annotations.
    """
    boxes, labels, scores = prediction_arrays(record, old_count)
    gt = np.asarray(gt_boxes, dtype=np.float32)
    if (gt.ndim != 2 or gt.shape[1] != 6 or not np.isfinite(gt).all()
            or np.any(gt[:, 3:] <= 0)):
        raise ValueError('visible GT must be finite positive gravity-center AABBs')
    keep = scores >= policy['confidence_threshold']
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    kept = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        local = _geometry()['nms_indices_iou'](boxes[indices], scores[indices], policy['nms_threshold'])
        kept.extend(indices[local].tolist())
    kept = np.asarray(sorted(kept), dtype=np.int64)
    boxes, labels, scores = boxes[kept], labels[kept], scores[kept]
    if len(boxes) > policy['max_pseudo_per_scene']:
        keep = np.argsort(scores)[::-1][:policy['max_pseudo_per_scene']]
        boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    keep = _geometry()['filter_pseudo_by_iou_against_gt'](gt, boxes, policy['pseudo_vs_gt_iou_thr'])
    return dict(boxes=boxes[keep], labels=labels[keep], scores=scores[keep])


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate pseudo artifact key: {key}')
        result[key] = value
    return result


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_payload(payload, protocol, stage, *, teacher_sha256, train_info_sha256,
                     geometry_sha256, scene_ids=None, policy_name='ldmr'):
    protocol.seen_count(stage)
    if stage == 1:
        raise ValueError('stage 1 cannot use pseudo supervision')
    if not isinstance(payload, dict):
        raise ValueError('pseudo artifact must be a mapping')
    expected = dict(version=1, protocol_id=protocol.identity, stage=stage,
                    teacher_stage=stage - 1, teacher_sha256=teacher_sha256,
                    train_info_sha256=train_info_sha256, geometry_sha256=geometry_sha256,
                    teacher_config_sha256=payload.get('teacher_config_sha256'),
                    coordinates='aligned_depth_gravity_aabb6', label_space='ldmr_frequency',
                    classes=protocol.decoder_classes(stage - 1), policy=scannet_policy(stage, protocol.stage_setting, policy_name))
    if not all(_digest(value) for value in (teacher_sha256, train_info_sha256, geometry_sha256, payload.get('teacher_config_sha256'))):
        raise ValueError('teacher and train manifest SHA256 are required')
    if (set(payload) != set(expected) | {'predictions'}
            or any(payload.get(key) != value for key, value in expected.items())
            or any(type(payload.get(key)) is not int for key in ('version', 'stage', 'teacher_stage'))):
        raise ValueError('pseudo artifact provenance, protocol, geometry or policy mismatch')
    records = payload['predictions']
    if not isinstance(records, dict) or not records or any(not isinstance(k, str) or not k for k in records):
        raise ValueError('pseudo artifact requires explicit scene IDs, including empty predictions')
    if scene_ids is not None and set(records) != set(scene_ids):
        raise ValueError('pseudo artifact scene coverage differs from the training manifest')
    result = {}
    for key, record in records.items():
        values = prediction_arrays(record, protocol.seen_count(stage - 1))
        result[key] = dict(zip(('boxes', 'labels', 'scores'), values))
    return result


def load_pseudo(path, protocol, stage, *, teacher_sha256, train_info_sha256,
                geometry_sha256, artifact_sha256=None, scene_ids=None, policy_name='ldmr'):
    raw = Path(path).read_bytes()
    if artifact_sha256 is not None and hashlib.sha256(raw).hexdigest() != artifact_sha256:
        raise ValueError('pseudo artifact changed after configuration')
    payload = json.loads(raw, object_pairs_hook=_unique_keys)
    return validate_payload(payload, protocol, stage, teacher_sha256=teacher_sha256,
                            train_info_sha256=train_info_sha256, geometry_sha256=geometry_sha256, scene_ids=scene_ids, policy_name=policy_name)


def geometry_digest(records):
    """Bind teacher input geometry/superpoints, never hidden annotation masks."""
    from .sequence import sha256
    paths = {str(Path(path).resolve()) for row in records for path in
             (row['lidar_points']['lidar_path'], row['super_pts_path'])}
    hashes = {path: sha256(path) for path in sorted(paths)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def write_pseudo(path, predictions, protocol, stage, *, teacher_sha256,
                 train_info_sha256, geometry_sha256, teacher_config_sha256, policy_name='ldmr'):
    """Publish a validated JSON artifact exclusively; no partial final file."""
    import os
    import tempfile
    path = Path(path)
    payload = dict(version=1, protocol_id=protocol.identity, stage=stage,
        teacher_stage=stage - 1, teacher_sha256=teacher_sha256,
        train_info_sha256=train_info_sha256, geometry_sha256=geometry_sha256,
        teacher_config_sha256=teacher_config_sha256,
        coordinates='aligned_depth_gravity_aabb6', label_space='ldmr_frequency',
        classes=protocol.decoder_classes(stage - 1), policy=scannet_policy(stage, protocol.stage_setting, policy_name),
        predictions={key: {name: np.asarray(value).tolist() for name, value in record.items()}
                     for key, record in predictions.items()})
    validate_payload(payload, protocol, stage, teacher_sha256=teacher_sha256,
                     train_info_sha256=train_info_sha256, geometry_sha256=geometry_sha256, policy_name=policy_name)
    encoded = json.dumps(payload, allow_nan=False, indent=2) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
