"""Validated checkpoint-bound LD trajectories; no detector or training hooks.

A missing measurement is an error, not a zero-quality detection. Scoring stays
in the maintained LDMR implementation after this boundary verifies coverage.
"""
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

from .learning_dynamics import reference_functions


_FIELDS = {'version', 'protocol_id', 'stage', 'step', 'checkpoint_sha256',
           'measurement_sha256', 'iou_thr', 'q_metric', 'eps', 'seats'}
_STATS = {'tp', 'fp', 'fn', 'denom', 'gt_count', 'tp_count', 'dt_count', 'q', 'count'}


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _number(value):
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _seat_map(snapshot, protocol):
    seats = snapshot['seats']
    if not isinstance(seats, list) or not seats:
        raise ValueError('LD snapshot requires nonempty explicit seat coverage')
    result = {}
    for seat in seats:
        if not isinstance(seat, dict) or set(seat) != {'scene_id', 'save_stage', 'classes'}:
            raise ValueError('invalid LD seat schema')
        sid, saved = seat['scene_id'], seat['save_stage']
        if not isinstance(sid, str) or not sid or not _integer(saved, 1) or saved > snapshot['stage']:
            raise ValueError('invalid LD scene/saved-stage identity')
        key = (sid, saved)
        if key in result:
            raise ValueError('duplicate LD seat identity')
        visible = protocol.visible_classes(snapshot['stage'],
            None if saved == snapshot['stage'] else saved)
        classes = seat['classes']
        if not isinstance(classes, dict):
            raise ValueError('LD classes must be a mapping')
        normalized = {}
        for cid, stat in classes.items():
            if type(cid) is int:
                label = cid
            elif isinstance(cid, str) and cid.isascii() and cid.isdecimal() and str(int(cid)) == cid:
                label = int(cid)
            else:
                raise ValueError('LD class keys must be canonical integers')
            if label in normalized or not isinstance(stat, dict) or set(stat) != _STATS:
                raise ValueError('duplicate LD class or invalid statistic schema')
            for field in _STATS - {'q'}:
                value = stat[field]
                if not _number(value) or value < 0 or int(value) != value:
                    raise ValueError('LD counts must be finite nonnegative integers')
            gt, tp, fp, fn, dt = (stat[k] for k in ('gt_count', 'tp', 'fp', 'fn', 'dt_count'))
            if (tp + fn != gt or tp + fp != dt or stat['tp_count'] != tp
                    or stat['denom'] != gt or stat['count'] != gt):
                raise ValueError('inconsistent LD TP/FP/FN/GT counts')
            q = stat['q']
            if gt == 0:
                if q is not None:
                    raise ValueError('LD quality is undefined for absent GT')
            else:
                numerator = tp if snapshot['q_metric'] == 'recall' else 2 * tp
                denominator = tp + fn if snapshot['q_metric'] == 'recall' else 2 * tp + fp + fn
                expected = numerator / (denominator + snapshot['eps'])
                if not _number(q) or not math.isclose(q, expected, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('LD quality disagrees with counts or measurement policy')
            normalized[label] = deepcopy(stat)
        if set(normalized) != set(visible):
            raise ValueError('LD class coverage differs from saved visibility')
        result[key] = normalized
    return result


def validate_snapshot(snapshot, protocol):
    """Validate a JSON-compatible snapshot and return normalized seat records."""
    if not isinstance(snapshot, dict) or set(snapshot) != _FIELDS:
        raise ValueError('invalid LD snapshot schema')
    if type(snapshot['version']) is not int or snapshot['version'] != 1:
        raise ValueError('unsupported LD snapshot version')
    if snapshot['protocol_id'] != protocol.identity:
        raise ValueError('LD protocol mismatch')
    protocol.seen_count(snapshot['stage'])
    if not _integer(snapshot['stage'], 1) or not _integer(snapshot['step']):
        raise ValueError('LD stage/step must be integer indices')
    if not all(_digest(snapshot[name]) for name in ('checkpoint_sha256', 'measurement_sha256')):
        raise ValueError('LD checkpoint and measurement SHA256 required')
    if (not _number(snapshot['iou_thr']) or not 0 < snapshot['iou_thr'] < 1
            or snapshot['q_metric'] not in ('f1', 'recall')
            or not _number(snapshot['eps']) or snapshot['eps'] <= 0):
        raise ValueError('invalid LD quality policy')
    reference_functions()['scoring']['_normalize_iou_mode'](snapshot['iou_thr'])
    records = _seat_map(snapshot, protocol)
    return [dict(scene_id=sid, save_stage=saved, classes=records[(sid, saved)])
            for sid, saved in sorted(records)]


def make_snapshot(seats, *, protocol, stage, step, checkpoint_sha256,
                  measurement_sha256, iou_thr=.25, q_metric='recall', eps=1e-9):
    """Bind measured seats to checkpoint and measurement-context digests.

    measurement_sha256 must bind the fixed evaluation inputs, visible GT,
    inference config/seed and measurement source versions. It is not a hash of
    the changing predictions. Callers own creating this evaluation context.
    """
    value = dict(version=1, protocol_id=protocol.identity, stage=stage, step=step,
        checkpoint_sha256=checkpoint_sha256, measurement_sha256=measurement_sha256,
        iou_thr=iou_thr, q_metric=q_metric, eps=eps, seats=deepcopy(seats))
    value['seats'] = validate_snapshot(value, protocol)
    return value


def validate_trajectory(snapshots, *, protocol, expected_steps):
    """Reject missing endpoints/intervals, changed GT and mixed provenance."""
    if (not isinstance(expected_steps, (list, tuple)) or not expected_steps
            or any(not _integer(step) for step in expected_steps)
            or list(expected_steps) != sorted(set(expected_steps))
            or expected_steps[0] != 0):
        raise ValueError('expected LD steps must be strictly increasing from zero')
    if not isinstance(snapshots, (list, tuple)) or len(snapshots) != len(expected_steps):
        raise ValueError('missing or extra LD snapshots')
    metrics, baseline, reference = {}, None, None
    for index, (snapshot, step) in enumerate(zip(snapshots, expected_steps)):
        records = validate_snapshot(snapshot, protocol)
        if snapshot['step'] != step:
            raise ValueError('LD snapshot does not match the planned step')
        context = {key: snapshot[key] for key in
                   ('protocol_id', 'stage', 'measurement_sha256', 'iou_thr', 'q_metric', 'eps')}
        coverage = {(row['scene_id'], row['save_stage']):
                    {cid: stat['gt_count'] for cid, stat in row['classes'].items()} for row in records}
        if baseline is None:
            baseline, reference = coverage, context
        elif context != reference or coverage != baseline:
            raise ValueError('LD trajectory measurement context, seat coverage or GT changed')
        metrics[index] = records
    return metrics


def score_design2(snapshots, *, protocol, expected_steps):
    """Use the reference Design-2 scorer only on complete validated trajectories."""
    metrics = validate_trajectory(snapshots, protocol=protocol, expected_steps=expected_steps)
    context = snapshots[0]
    classes = sorted({cid for row in metrics[0] for cid in row['classes']})
    return reference_functions()['scoring']['compute_learning_dynamics_design1_scores'](
        metrics, class_ids=classes, new_classes=list(protocol.visible_classes(context['stage'])),
        q_metric=context['q_metric'], eps=context['eps'], design_version=2)


def write_snapshot(path, snapshot, protocol):
    """Publish validated JSON exclusively; preserve existing and partial outputs."""
    seats = validate_snapshot(snapshot, protocol)
    snapshot = dict(snapshot, seats=seats)
    path = Path(path)
    encoded = (json.dumps(snapshot, allow_nan=False, sort_keys=True, indent=2) + '\n').encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def load_snapshot(path, protocol, *, expected_sha256):
    """Load only the exact sealed artifact requested by its consumer."""
    raw = Path(path).read_bytes()
    if not _digest(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('LD snapshot artifact hash mismatch')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate LD JSON key')
            result[key] = value
        return result
    snapshot = json.loads(raw, object_pairs_hook=unique)
    snapshot['seats'] = validate_snapshot(snapshot, protocol)
    return snapshot
