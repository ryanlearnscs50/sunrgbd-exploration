"""Point-aligned object replay for ScanNet, alongside LDMR's existing selector.

Unlike legacy box-only crops, only points belonging to the source instance are
supervised. Box context is retained as ignored points. This module imports no
host framework and never reads or modifies shared datasets.
"""
from copy import deepcopy
from numbers import Integral
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _aligned(points, semantic, instances, superpoints):
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 6 or points.dtype.kind != 'f'
            or not np.isfinite(points).all()):
        raise ValueError('points must be finite XYZRGB floating-point rows')
    arrays = [np.asarray(a) for a in (semantic, instances, superpoints)]
    for a in arrays:
        if a.shape != (len(points),) or a.dtype.kind not in 'iu':
            raise ValueError('masks must be point-aligned integer arrays')
    if np.any((arrays[0] < 0) | (arrays[0] > 40)):
        raise ValueError('semantic IDs must be raw NYU40 (0 through 40)')
    if np.any(arrays[1] < -1) or np.any(arrays[2] < 0):
        raise ValueError('invalid instance/superpoint IDs')
    return points, *arrays


def extract_crop(points, semantic, instances, superpoints, *, bbox, instance_id,
                 class_id, scene_id, object_idx, stage, protocol,
                 crop_margin=0.0, min_points=20, floor_percentile=1.0,
                 crop_tolerance=1e-6):
    """Build a richer candidate before calling the unchanged LDMR selector.

    All inputs are axis-aligned DEPTH coordinates, raw NYU40 semantic IDs, and
    point-aligned masks. Boxes use gravity centers. Instance ID is explicit;
    annotation row number must never be assumed to equal a point-instance ID.
    """
    points, semantic, instances, superpoints = _aligned(
        points, semantic, instances, superpoints)
    instance_id = _integer(instance_id, 'instance_id')
    class_id = _integer(class_id, 'class_id')
    object_idx = _integer(object_idx, 'object_idx')
    min_points = _integer(min_points, 'min_points', 1)
    if class_id not in protocol.visible_classes(stage):
        raise ValueError('only current-stage classes may populate the bank')
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError('scene_id must be nonempty')
    if not np.isfinite(crop_margin) or crop_margin < 0:
        raise ValueError('crop_margin must be finite and nonnegative')
    if not np.isfinite(crop_tolerance) or crop_tolerance < 0:
        raise ValueError('crop_tolerance must be finite and nonnegative')
    if not np.isfinite(floor_percentile) or not 0 <= floor_percentile <= 100:
        raise ValueError('invalid floor percentile')
    box = np.asarray(bbox, dtype=np.float32)
    if (box.shape not in ((6,), (7,)) or not np.isfinite(box).all()
            or np.any(box[3:6] <= 0) or (len(box) == 7 and box[6] != 0)):
        raise ValueError('ScanNet crop requires a positive axis-aligned gravity box')
    source_member = instances == instance_id
    raw_id = protocol.raw_ids[class_id]
    if not source_member.any() or not np.all(semantic[source_member] == raw_id):
        raise ValueError('source instance does not match the declared class')
    xyz = points[:, :3].astype(np.float64) - box[:3].astype(np.float64)
    keep = np.all(np.abs(xyz) <= box[3:6].astype(np.float64) * (0.5 + crop_margin) + crop_tolerance, axis=1)
    if keep.sum() < min_points or not source_member[keep].any():
        return None
    crop = points[keep].astype(np.float32, copy=True)
    crop[:, :3] = xyz[keep]
    member = source_member[keep]
    _, sp = np.unique(superpoints[keep], return_inverse=True)
    payload = dict(
        format_version=1, protocol_id=protocol.identity, scene_id=scene_id,
        object_idx=object_idx, class_id=class_id, stage_id=int(stage),
        source_instance_id=instance_id, bbox=box.copy(), points=crop,
        point_count=len(crop), nyu40_id=int(raw_id),
        coordinate_frame='object_local_gravity_center',
        source_floor_offset=float(box[2] - box[5] * .5 -
                                  np.percentile(points[:, 2], floor_percentile)),
        pts_semantic_mask=np.where(member, raw_id, 0).astype(np.int64),
        pts_instance_mask=np.where(member, 0, -1).astype(np.int64),
        sp_pts_mask=sp.astype(np.int64), replay_weight=1.0)
    validate_crop(payload, protocol)
    return payload


def validate_crop(crop, protocol):
    if crop.get('format_version') != 1 or crop.get('protocol_id') != protocol.identity:
        raise ValueError('incompatible masked-crop format/protocol')
    points, sem, inst, sp = _aligned(crop['points'], crop['pts_semantic_mask'],
                                   crop['pts_instance_mask'], crop['sp_pts_mask'])
    cid = _integer(crop['class_id'], 'class_id')
    if cid not in protocol.visible_classes(crop['stage_id']):
        raise ValueError('crop class is outside its population stage')
    if crop.get('coordinate_frame') != 'object_local_gravity_center':
        raise ValueError('invalid crop coordinate frame')
    _integer(crop['object_idx'], 'object_idx')
    _integer(crop['source_instance_id'], 'source_instance_id')
    if not isinstance(crop['scene_id'], str) or not crop['scene_id']:
        raise ValueError('invalid source scene')
    box = np.asarray(crop['bbox'])
    if (box.shape not in ((6,), (7,)) or not np.isfinite(box).all()
            or np.any(box[3:6] <= 0) or (len(box) == 7 and box[6] != 0)):
        raise ValueError('invalid axis-aligned crop box')
    if (len(points) != crop['point_count'] or not np.any(inst == 0)
            or not np.all(np.isin(inst, [-1, 0]))
            or not np.all(sem[inst == 0] == protocol.raw_ids[cid])
            or not np.all(sem[inst == -1] == 0)
            or crop['nyu40_id'] != protocol.raw_ids[cid]):
        raise ValueError('crop membership/count/class mismatch')
    if not np.array_equal(np.unique(sp), np.arange(len(np.unique(sp)))):
        raise ValueError('crop superpoints must be compact')
    if (not np.isfinite(crop['source_floor_offset'])
            or not np.isfinite(crop.get('replay_weight', 1.0))
            or crop.get('replay_weight', 1.0) < 0):
        raise ValueError('invalid floor offset/replay weight')


def selected_payloads(bank, candidates, protocol):
    """Join masks to the bank's retained candidates, without changing selection.

    Pass the same candidates to ObjectMemoryBank.add_exemplars first. Supply
    candidates for every retained class (including existing rich payloads).
    Discarded candidate arrays are not retained in the returned export.
    """
    lookup = {}
    for crop in candidates:
        validate_crop(crop, protocol)
        key = (crop['scene_id'], crop['object_idx'], crop['class_id'])
        if key in lookup:
            raise ValueError('duplicate object candidate identity')
        lookup[key] = crop
    output = []
    for selected in bank.get_all_exemplars():
        key = (selected['scene_id'], selected['object_idx'], selected['class_id'])
        if key not in lookup:
            raise ValueError(f'missing masks for selected object {key}')
        crop = lookup[key]
        if (not np.array_equal(selected['points'], crop['points'])
                or not np.array_equal(selected['bbox'], crop['bbox'])
                or selected['stage_id'] != crop['stage_id']):
            raise ValueError('selected crop geometry/stage differs from mask payload')
        result = deepcopy(crop)
        result['replay_weight'] = float(selected.get('replay_weight', 1.0))
        validate_crop(result, protocol)
        output.append(result)
    return output


def save_crops(path, crops, protocol, completed_stage):
    """Atomic trusted-local pickle for selected rich crops, not selector state."""
    protocol.seen_count(completed_stage)
    keys = set()
    for crop in crops:
        validate_crop(crop, protocol)
        key = (crop['scene_id'], crop['object_idx'], crop['class_id'])
        if key in keys or crop['stage_id'] > completed_stage:
            raise ValueError('duplicate or future crop in export')
        keys.add(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temp = f.name
            pickle.dump(dict(format_version=1, protocol_id=protocol.identity,
                             completed_stage=completed_stage, crops=list(crops)),
                        f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)


def load_crops(path, protocol, expected_stage):
    """Load only trusted local artifacts. Fail closed on protocol/stage mismatch."""
    protocol.seen_count(expected_stage)
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    if (payload.get('format_version') != 1
            or payload.get('protocol_id') != protocol.identity
            or payload.get('completed_stage') != expected_stage):
        raise ValueError('masked crop export format/protocol/stage mismatch')
    crops = payload['crops']
    keys = set()
    for crop in crops:
        validate_crop(crop, protocol)
        key = (crop['scene_id'], crop['object_idx'], crop['class_id'])
        if key in keys or crop['stage_id'] > expected_stage:
            raise ValueError('duplicate or future crop in export')
        keys.add(key)
    return crops


def paste_crop(points, semantic, instances, superpoints, crop, *, center,
               occupied_boxes, protocol, stage, inserted_mask=None, collision_iou=0.0):
    """Paste once at a proposed center; return None if its AABB overlaps.

    occupied_boxes are gravity-centered visible/pseudo boxes, never hidden GT.
    Placement proposals belong to the caller. No random yaw for ScanNet's AABB
    path. Caller can preserve elevation with floor + size_z/2 + floor_offset.
    Returns copies of all aligned fields and a mask for pasted *target* points.
    """
    points, semantic, instances, superpoints = _aligned(
        points, semantic, instances, superpoints)
    validate_crop(crop, protocol)
    protocol.seen_count(stage)
    if crop['stage_id'] >= stage:
        raise ValueError('only prior-stage objects may be replayed')
    if inserted_mask is None:
        inserted_mask = np.zeros(len(points), dtype=bool)
    inserted_mask = np.asarray(inserted_mask)
    if inserted_mask.dtype != np.bool_ or inserted_mask.shape != (len(points),):
        raise ValueError('inserted mask must align with destination points')
    center = np.asarray(center, dtype=np.float32)
    occupied = np.asarray(occupied_boxes, dtype=np.float32)
    if (center.shape != (3,) or not np.isfinite(center).all()
            or occupied.ndim != 2 or occupied.shape[1] != 6
            or not np.isfinite(occupied).all() or np.any(occupied[:, 3:] <= 0)):
        raise ValueError('invalid placement center or occupied gravity boxes')
    size = np.asarray(crop['bbox'][3:6])
    overlap = np.minimum(center + size / 2, occupied[:, :3] + occupied[:, 3:] / 2) - np.maximum(
        center - size / 2, occupied[:, :3] - occupied[:, 3:] / 2)
    if not np.isfinite(collision_iou) or not 0 <= collision_iou <= 1:
        raise ValueError("collision IoU must be in [0, 1]")
    intersection = np.maximum(overlap, 0).prod(axis=1)
    union = size.prod() + occupied[:, 3:].prod(axis=1) - intersection
    if np.any(intersection / union > collision_iou):
        return None
    # Compact IDs before offsetting to prevent collisions and integer overflow.
    dest_inst = np.full(len(points), -1, dtype=np.int64)
    valid = instances >= 0
    unique, inverse = np.unique(instances[valid], return_inverse=True)
    dest_inst[valid] = inverse
    _, dest_sp = np.unique(superpoints, return_inverse=True)
    crop_points = crop['points'].astype(points.dtype, copy=True)
    crop_points[:, :3] += center
    member = crop['pts_instance_mask'] == 0
    return dict(
        points=np.concatenate([points, crop_points]),
        pts_semantic_mask=np.concatenate([semantic.astype(np.int64), crop['pts_semantic_mask']]),
        pts_instance_mask=np.concatenate([dest_inst, np.where(member, len(unique), -1)]),
        sp_pts_mask=np.concatenate([dest_sp, crop['sp_pts_mask'] + len(np.unique(dest_sp))]),
        ldmr_inserted_mask=np.concatenate([inserted_mask, member]),
        placed_box=np.concatenate([center, size]))
