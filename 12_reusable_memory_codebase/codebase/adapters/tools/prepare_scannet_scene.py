#!/usr/bin/env python3
"""Prepare one real ScanNet scene in a NEW output directory, never in-place.

Requires the host torch/segmentator/plyfile environment. Reads LDMR's standalone
ScanNet exporter to preserve raw NYU40 labels and alignment conventions.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'adapters'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--label-map', required=True, type=Path)
    parser.add_argument('--scene-id', required=True)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--stage-setting', default='scannet35_s5_freqorder')
    args = parser.parse_args()
    if not re.fullmatch(r'scene[0-9]{4}_[0-9]{2}', args.scene_id):
        parser.error('invalid ScanNet scene ID')
    raw = args.raw_root.resolve()
    output = args.output_root.resolve()
    if output == raw or raw in output.parents or output in raw.parents:
        parser.error('output must be separate from the shared raw dataset')
    scene = raw / args.scene_id
    names = [args.scene_id + suffix for suffix in (
        '_vh_clean_2.ply', '.aggregation.json', '_vh_clean_2.0.010000.segs.json', '.txt')]
    paths = [scene / name for name in names]
    # Actual open, not os.access: fail before creating any output if unreadable.
    for path in paths + [args.label_map]:
        with path.open('rb') as f:
            f.read(1)
    destination = output / args.scene_id
    if destination.exists():
        parser.error(f'output already exists; refusing to overwrite {destination}')

    import numpy as np
    import torch
    import segmentator
    from plyfile import PlyData
    from ldmr_unidet3d import ScanNet35Protocol
    from ldmr_unidet3d.data import convert_info

    sys.path.insert(0, str(REPO / 'data/scannet'))
    try:
        spec = importlib.util.spec_from_file_location('ldmr_scannet_export', REPO / 'data/scannet/load_scannet_data.py')
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        points, semantic, instances, _, boxes, _, alignment = source.export(
            *(str(path) for path in paths), str(args.label_map), None)
    finally:
        sys.path.pop(0)
    with paths[0].open('rb') as f:
        mesh = PlyData.read(f)
    vertices = np.column_stack([mesh['vertex'][k] for k in ('x', 'y', 'z')]).astype(np.float32)
    faces = np.asarray(list(mesh['face']['vertex_indices']), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('segmentator requires a triangle mesh')
    np.testing.assert_array_equal(vertices, points[:, :3])
    sp = segmentator.segment_mesh(torch.from_numpy(vertices), torch.from_numpy(faces)).numpy()
    if sp.shape != (len(points),) or sp.dtype.kind not in 'iu' or np.any(sp < 0):
        raise ValueError('invalid superpoint output')
    _, sp = np.unique(sp, return_inverse=True)
    good = (np.all(boxes[:, 3:6] > 0, axis=1) & np.isfinite(boxes).all(axis=1)
            & (boxes[:, -1] >= 1) & (boxes[:, -1] <= 40))
    boxes = boxes[good]
    legacy = dict(point_cloud=dict(lidar_idx=args.scene_id, num_features=6),
                  pts_path='points.bin', pts_instance_mask_path='instance.bin',
                  pts_semantic_mask_path='semantic.bin', annos=dict(
                      **{'class':boxes[:, -1].astype(np.int64)},
                      gt_boxes_upright_depth=boxes[:, :6], axis_align_matrix=alignment))
    protocol = ScanNet35Protocol(args.stage_setting)
    info = convert_info(legacy, protocol, data_root=destination,
                        superpoint_path=destination/'superpoints.bin', box_label_space='nyu40')
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.prepare-', dir=output))
    try:
        for name, array, dtype in (
            ('points.bin', points, np.float32), ('semantic.bin', semantic, np.int64),
            ('instance.bin', instances, np.int64), ('superpoints.bin', sp, np.int64)):
            np.asarray(array, dtype=dtype).tofile(temporary / name)
        payload = dict(metainfo=dict(dataset='scannet', classes=list(protocol.names)), data_list=[info])
        with (temporary/'info.pkl').open('wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        provenance = dict(scene_id=args.scene_id, point_count=len(points),
            superpoint_count=len(np.unique(sp)), protocol_id=protocol.identity,
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + [args.label_map]},
            segmentator_module=str(segmentator.__file__),
            segmentator_module_sha256=hashlib.sha256(Path(segmentator.__file__).read_bytes()).hexdigest(),
            point_row_order='original mesh vertex order; no subsampling')
        (temporary/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(destination / 'info.pkl')


if __name__ == '__main__':
    main()
