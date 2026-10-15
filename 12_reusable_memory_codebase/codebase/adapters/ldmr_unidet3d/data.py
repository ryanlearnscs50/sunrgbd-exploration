"""Explicit conversion of legacy ScanNet infos; all source access is read-only."""
from copy import deepcopy
from pathlib import Path
import numpy as np


def verify_mesh_rows(points, vertices, faces):
    """Require exact original mesh rows before attaching mesh superpoints.

    A nearest-neighbor join can silently confuse repeated vertices or sampled
    points. Prepared masks are safe here only with identical XYZ row order.
    """
    points, vertices, faces = map(np.asarray, (points, vertices, faces))
    if (points.ndim != 2 or points.shape[1] != 6
            or vertices.shape != (len(points), 3) or not len(points)
            or not np.isfinite(points).all()
            or not np.array_equal(points[:, :3], vertices)):
        raise ValueError('prepared points must match every mesh XYZ row exactly')
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.dtype.kind not in 'iu'
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError('mesh faces must contain valid triangle vertex indices')


def convert_info(info, protocol, *, data_root, superpoint_path, box_label_space):
    """Return one host record without guessing label numbering or touching files.

    Caller must verify the files (including real superpoints) before training.
    Points are unaligned XYZRGB; boxes are aligned gravity-center AABBs.
    """
    if box_label_space not in ('nyu40', 'nyu40_zero_based'):
        raise ValueError('box_label_space must explicitly name NYU40 numbering')
    root = Path(data_root).absolute()
    annos = info['annos']
    labels = np.asarray(annos['class'])
    boxes = np.asarray(annos['gt_boxes_upright_depth'])
    if (labels.ndim != 1 or labels.dtype.kind not in 'iu'
            or boxes.shape != (len(labels), 6) or not np.isfinite(boxes).all()
            or np.any(boxes[:, 3:] <= 0)):
        raise ValueError('malformed legacy detection annotations')
    lower, upper = (0, 39) if box_label_space == 'nyu40_zero_based' else (1, 40)
    if np.any((labels < lower) | (labels > upper)):
        raise ValueError('box labels outside declared NYU40 numbering')
    raw = labels.astype(np.int64) + (box_label_space == 'nyu40_zero_based')
    align = np.asarray(annos['axis_align_matrix'], dtype=np.float32)
    if (align.shape != (4, 4) or not np.isfinite(align).all()
            or not np.allclose(align[3], [0, 0, 0, 1])
            or not np.allclose(align[:3, :3].T @ align[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(align[:3, :3]), 1, atol=1e-3)):
        raise ValueError('axis alignment must be a finite rigid transform')
    if info['point_cloud']['num_features'] != 6:
        raise ValueError('ScanNet adapter requires XYZRGB (six features)')
    mapping = dict(zip(protocol.raw_ids, range(len(protocol.raw_ids))))
    instances = [dict(bbox_3d=box.astype(float).tolist(), bbox_label_3d=mapping[int(label)])
                 for box, label in zip(boxes, raw) if int(label) in mapping]
    return dict(
        sample_idx=str(info['point_cloud']['lidar_idx']),
        lidar_points=dict(num_pts_feats=6, lidar_path=str(root / info['pts_path'])),
        pts_instance_mask_path=str(root / info['pts_instance_mask_path']),
        pts_semantic_mask_path=str(root / info['pts_semantic_mask_path']),
        super_pts_path=str(Path(superpoint_path).absolute()),
        axis_align_matrix=align.tolist(), instances=instances,
        ldmr_protocol_id=protocol.identity, ldmr_semantic_space='nyu40')


def stage_records(records, protocol, stage, *, evaluation=False):
    """Match LDMR scene eligibility: current-class train / seen-class evaluation."""
    allowed = set(protocol.visible_classes(stage, evaluation=evaluation))
    output = []
    scene_ids = set()
    for record in records:
        if record.get('ldmr_protocol_id') != protocol.identity:
            raise ValueError('record protocol mismatch')
        if record.get('is_replay', False):
            raise ValueError('natural stage filtering cannot accept replay records')
        scene_id = record['sample_idx']
        if scene_id in scene_ids:
            raise ValueError('duplicate natural scene ID')
        scene_ids.add(scene_id)
        for inst in record['instances']:
            label = inst['bbox_label_3d']
            if type(label) is not int or not 0 <= label < len(protocol.names):
                raise ValueError('record labels are not LDMR class indices')
        filtered = [inst for inst in record['instances'] if inst['bbox_label_3d'] in allowed]
        if filtered:
            result = deepcopy(record)
            result['instances'] = deepcopy(filtered)
            output.append(result)
    return output


def verify_point_files(record):
    """Validate lengths and IDs without writing or modifying any point payload."""
    if record.get('ldmr_semantic_space') != 'nyu40':
        raise ValueError('point semantics must be explicitly declared raw NYU40')
    point_path = Path(record['lidar_points']['lidar_path'])
    if point_path.stat().st_size % (6 * np.dtype(np.float32).itemsize):
        raise ValueError('invalid point file byte length for XYZRGB float32 rows')
    pts = np.fromfile(point_path, dtype=np.float32)
    if len(pts) % 6 or not np.isfinite(pts).all():
        raise ValueError('invalid point file')
    count = len(pts) // 6
    if not count:
        raise ValueError('empty point file')
    for key in ('pts_instance_mask_path', 'pts_semantic_mask_path', 'super_pts_path'):
        path = Path(record[key])
        if path.stat().st_size != count * 8:
            raise ValueError(f'{key} does not contain one int64 ID per point')
        values = np.fromfile(path, dtype=np.int64)
        if key == 'pts_semantic_mask_path' and np.any((values < 0) | (values > 40)):
            raise ValueError('semantic file is outside raw NYU40 IDs')
        if key == 'pts_instance_mask_path' and np.any(values < -1):
            raise ValueError('invalid negative instance ID')
        if key == 'super_pts_path' and np.any(values < 0):
            raise ValueError('invalid negative superpoint ID')
    return count


def validate_disjoint_splits(train_records, val_records):
    """Reject scene leakage, including renamed IDs sharing a point file.

    Follow symlinks and compare filesystem identities to catch hard links.
    This does not detect independently copied files with renamed scene IDs.
    """
    if {r['sample_idx'] for r in train_records} & {r['sample_idx'] for r in val_records}:
        raise ValueError('train and validation scene IDs overlap')
    identities = []
    for records in (train_records, val_records):
        files = set()
        for record in records:
            stat = Path(record['lidar_points']['lidar_path']).stat()
            files.add((stat.st_dev, stat.st_ino))
        identities.append(files)
    if identities[0] & identities[1]:
        raise ValueError('train and validation point files overlap')


def validate_stage_manifest(path, protocol, stage, memory_mode, *, evaluation):
    """Check explicit stage metadata before constructing a host dataset."""
    import pickle
    with open(path, 'rb') as f:
        payload = pickle.load(f)  # trusted local prepared metadata only
    meta = payload.get('metainfo', {})
    if (meta.get('ldmr_protocol_id') != protocol.identity or meta.get('ldmr_stage') != stage
            or meta.get('ldmr_split') != ('val' if evaluation else 'train')):
        raise ValueError('stage manifest protocol/stage/split mismatch')
    records = payload['data_list']
    if not records:
        raise ValueError('stage manifest is empty')
    if not evaluation and meta.get('ldmr_memory_mode') != memory_mode:
        raise ValueError('stage manifest memory mode mismatch')
    for record in records:
        if record.get('ldmr_protocol_id') != protocol.identity:
            raise ValueError('record protocol mismatch')
        replay = record.get('is_replay', False)
        if replay and 'ldmr_replay_from_stage' not in record:
            raise ValueError('replay record lacks snapshot stage')
        if replay and (evaluation or memory_mode != 'scene'):
            raise ValueError('unexpected scene replay record')
        allowed = set(protocol.visible_classes(stage,
            record.get('ldmr_replay_from_stage') if replay else None,
            evaluation=evaluation))
        if record.get('ldmr_include_current', False):
            if not replay:
                raise ValueError('merged record lacks replay identity')
            allowed.update(protocol.visible_classes(stage))
        if not record['instances'] or any(inst['bbox_label_3d'] not in allowed for inst in record['instances']):
            raise ValueError('manifest contains wrong-stage box annotations')
        verify_point_files(record)
    return payload
