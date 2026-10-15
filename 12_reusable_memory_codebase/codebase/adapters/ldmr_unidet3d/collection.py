"""Fixed training-seat pools and checkpoint schedules for LD collection."""
from copy import deepcopy
import hashlib
import json


def checkpoint_schedule(specifications):
    from pathlib import Path
    schedule = []
    for spec in specifications:
        step, separator, path = spec.partition('=')
        if not separator or not step.isascii() or not step.isdecimal() or str(int(step)) != step:
            raise ValueError('checkpoint must be canonical epoch=/path/to/checkpoint')
        schedule.append((int(step), Path(path).resolve()))
    steps = [step for step, _ in schedule]
    if len(steps) < 2 or steps[0] != 0 or steps != sorted(set(steps)):
        raise ValueError('LD schedule requires at least two increasing epochs starting at zero')
    if len({path for _, path in schedule}) != len(schedule):
        raise ValueError('LD schedule cannot reuse a checkpoint path')
    return schedule


def validate_checkpoint_epoch(checkpoint, stage, epoch):
    meta = checkpoint['meta']
    if epoch == 0:
        if (meta.get('epoch', 0) != 0 or type(meta.get('epoch', 0)) is not int
                or (stage > 1 and meta.get('ldmr_initialized_from_stage') != stage - 1)):
            raise ValueError('epoch zero requires the saved stage initialization')
    elif type(meta.get('epoch')) is not int or meta['epoch'] != epoch:
        raise ValueError('checkpoint epoch differs from the planned measurement')


def fixed_pool(records, protocol, stage, scene_ids=None):
    """Split merged replay/current annotations into explicit visibility seats.

    Geometry enters inference once per scene; visible annotations stay outside
    the detector. Input records must come from a validated training manifest.
    """
    if scene_ids is not None:
        if (not isinstance(scene_ids, list) or not scene_ids
                or any(not isinstance(s, str) or not s for s in scene_ids)
                or len(set(scene_ids)) != len(scene_ids)):
            raise ValueError('scene IDs must be a nonempty unique string list')
        if not set(scene_ids).issubset({str(r['sample_idx']) for r in records}):
            raise ValueError('requested LD scenes missing from training manifest')
    geometry, seats = {}, {}
    fields = ('sample_idx', 'lidar_points', 'super_pts_path', 'axis_align_matrix',
              'ldmr_protocol_id', 'ldmr_semantic_space')
    for record in records:
        sid = str(record['sample_idx'])
        if scene_ids is not None and sid not in scene_ids:
            continue
        clean = {key: deepcopy(record[key]) for key in fields}
        if sid in geometry and geometry[sid] != clean:
            raise ValueError('same scene has conflicting LD inference geometry')
        geometry[sid] = clean
        saved = record['ldmr_replay_from_stage'] if record.get('is_replay') else stage
        versions = [saved]
        if record.get('ldmr_include_current') and saved != stage:
            versions.append(stage)
        for saved in versions:
            visible = protocol.visible_classes(stage, None if saved == stage else saved)
            instances = [deepcopy(i) for i in record['instances'] if i['bbox_label_3d'] in visible]
            key = (sid, saved)
            seat = dict(scene_id=sid, save_stage=saved, instances=instances)
            if key in seats:
                # Duplicate natural/replay records must not silently double-count.
                raise ValueError('duplicate LD seat in training pool')
            seats[key] = seat
    if not seats:
        raise ValueError('LD training pool is empty')
    return [geometry[sid] for sid in sorted(geometry)], [seats[key] for key in sorted(seats)]


def context_digest(context):
    return hashlib.sha256(json.dumps(context, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def object_source_pool(records, exemplars, protocol, stage):
    """Natural candidates plus unique old object carrier seats, for inference only.

    Carrier scenes never enter the object training dataset. Multiple population
    stages from one scene remain separate seats but share inference geometry.
    """
    from .data import stage_records
    import numpy as np
    stage_records(records, protocol, len(protocol.stages), evaluation=True)
    by_id = {r['sample_idx']: r for r in records}
    pool = stage_records(records, protocol, stage)
    seats = set()
    for obj in exemplars:
        sid, saved = obj['scene_id'], obj['stage_id']
        if type(saved) is not int or not 1 <= saved < stage:
            raise ValueError('object carrier must belong to a preceding stage')
        if sid not in by_id:
            raise ValueError('missing object carrier scene')
        row = by_id[sid]
        idx = obj['object_idx']
        if (type(idx) is not int or not 0 <= idx < len(row['instances'])
                or row['instances'][idx]['bbox_label_3d'] != obj['class_id']
                or obj['class_id'] not in protocol.visible_classes(saved)
                or not np.allclose(row['instances'][idx]['bbox_3d'], obj['bbox'][:6], atol=1e-4)):
            raise ValueError('object carrier annotation differs from selected exemplar')
        if (sid, saved) in seats:
            continue
        seats.add((sid, saved))
        carrier = deepcopy(row)
        visible = protocol.visible_classes(stage, saved)
        carrier['instances'] = [i for i in carrier['instances'] if i['bbox_label_3d'] in visible]
        carrier.update(is_replay=True, ldmr_replay_from_stage=saved)
        pool.append(carrier)
    return pool


def training_schedule(epochs, total_epochs):
    """Explicit LD boundaries include initialization and the final epoch."""
    if (not isinstance(epochs, (list, tuple)) or len(epochs) < 2
            or any(type(e) is not int for e in epochs)
            or list(epochs) != sorted(set(epochs))
            or epochs[0] != 0 or epochs[-1] != total_epochs):
        raise ValueError('LD epochs must increase from zero through the final training epoch')
    return list(epochs)


def reusable_prefix(directory, context, checkpoints, protocol):
    """Validate a completed collection before retaining its measured prefix.

    The new schedule extends the old one; all measurement inputs, GT, sources,
    policy and checkpoint bytes must remain identical. Keep provenance linking
    to the original artifacts even though snapshots bind the extended schedule.
    """
    if directory is None:
        return {}, None
    from pathlib import Path
    from .sequence import sha256, verify_hashes
    from .trajectories import load_snapshot, validate_trajectory
    directory = Path(directory).resolve()
    complete_path = directory / 'complete.json'
    receipt_hash = sha256(complete_path)
    complete = json.loads(complete_path.read_text())
    if complete.get('status') != 'complete':
        raise ValueError('reused collection must be complete')
    hashes = {str(directory / name): digest for name, digest in complete['artifacts'].items()}
    hashes[str(complete_path)] = receipt_hash
    verify_hashes(hashes, 'reused collection')
    old = json.loads((directory / 'context.json').read_text())
    steps = old['expected_steps']
    if (len(steps) >= len(context['expected_steps'])
            or steps != context['expected_steps'][:len(steps)]
            or complete['steps'] != steps
            or complete['measurement_sha256'] != context_digest(old)
            or {k: v for k, v in old.items() if k != 'expected_steps'} !=
               {k: v for k, v in context.items() if k != 'expected_steps'}):
        raise ValueError('reused collection is not an identical measurement-context prefix')
    if (len(complete['checkpoints']) != len(steps) or
            any(checkpoints.get(path) != digest for path, digest in complete['checkpoints'].items())):
        raise ValueError('reused collection checkpoint provenance changed')
    verify_hashes(complete['checkpoints'], 'reused checkpoints')
    snapshots = [load_snapshot(directory / f'snapshot_{step}.json', protocol,
                 expected_sha256=complete['artifacts'][f'snapshot_{step}.json']) for step in steps]
    validate_trajectory(snapshots, protocol=protocol, expected_steps=steps)
    for index, snapshot in enumerate(snapshots):
        if (snapshot['measurement_sha256'] != complete['measurement_sha256']
                or snapshot['checkpoint_sha256'] != list(checkpoints.values())[index]):
            raise ValueError('reused snapshot provenance changed')
    return {s['step']: s for s in snapshots}, dict(directory=str(directory), hashes=hashes)
