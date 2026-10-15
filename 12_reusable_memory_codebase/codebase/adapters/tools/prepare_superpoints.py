#!/usr/bin/env python3
"""Attach genuine mesh superpoints to an existing prepared ScanNet split.

Runs in the isolated host runtime. Requires exact mesh XYZ row identity with
prepared XYZRGB; never guesses a correspondence or modifies source files.
Use --scene-id repeatedly for a bounded integration subset. Output is a new
directory containing converted full-protocol infos, superpoints and provenance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--mesh-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--box-label-space', required=True,
                        choices=['nyu40', 'nyu40_zero_based'])
    parser.add_argument('--scene-id', action='append')
    parser.add_argument('--stage-setting', default='scannet35_s5_freqorder')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for source in (args.data_root.resolve(), args.mesh_root.resolve(), args.input.resolve().parent):
        if output == source or source in output.parents or output in source.parents:
            parser.error('output must be separate from the source dataset directories')
    if output.exists():
        parser.error('output exists; refusing to overwrite')

    import numpy as np
    import torch
    import segmentator
    from plyfile import PlyData
    from ldmr_unidet3d import ScanNet35Protocol
    from ldmr_unidet3d.data import convert_info, verify_mesh_rows, verify_point_files

    with args.input.open('rb') as stream:
        infos = pickle.load(stream)  # trusted local legacy metadata
    by_id = {str(info['point_cloud']['lidar_idx']): info for info in infos}
    if len(by_id) != len(infos):
        raise ValueError('duplicate source scene IDs')
    selected = args.scene_id or list(by_id)
    if len(selected) != len(set(selected)) or not selected:
        raise ValueError('scene selection must be nonempty and unique')
    for scene in selected:
        if not re.fullmatch(r'scene[0-9]{4}_[0-9]{2}', scene) or scene not in by_id:
            raise ValueError(f'invalid or missing scene ID: {scene}')
    protocol = ScanNet35Protocol(args.stage_setting)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.superpoints-', dir=output.parent))
    try:
        records, evidence = [], []
        for scene in selected:
            info = by_id[scene]
            mesh_path = args.mesh_root / scene / (scene + '_vh_clean_2.ply')
            mesh = PlyData.read(str(mesh_path))
            vertices = np.column_stack([mesh['vertex'][k] for k in ('x', 'y', 'z')]).astype(np.float32)
            faces = np.asarray(list(mesh['face']['vertex_indices']), dtype=np.int64)
            points_path = args.data_root / info['pts_path']
            points = np.fromfile(points_path, np.float32).reshape(-1, 6)
            verify_mesh_rows(points, vertices, faces)
            groups = segmentator.segment_mesh(torch.from_numpy(vertices), torch.from_numpy(faces)).numpy()
            if groups.shape != (len(points),) or groups.dtype.kind not in 'iu' or np.any(groups < 0):
                raise ValueError('invalid segmentator output')
            _, groups = np.unique(groups, return_inverse=True)
            name = scene + '.superpoints.bin'
            groups.astype(np.int64).tofile(temporary / name)
            record = convert_info(info, protocol, data_root=args.data_root,
                                  superpoint_path=temporary / name,
                                  box_label_space=args.box_label_space)
            verify_point_files(record)
            evidence.append(dict(scene=scene, points=len(points), superpoints=len(np.unique(groups)),
                mesh_path=str(mesh_path.resolve()), mesh_sha256=sha256(mesh_path),
                source_sha256={key: sha256(record[key]) for key in
                    ('pts_instance_mask_path', 'pts_semantic_mask_path')},
                points_sha256=sha256(points_path), superpoints_sha256=sha256(temporary / name)))
            record['super_pts_path'] = str(output / name)
            records.append(record)
            print(f'{scene}: {len(points)} points, {len(np.unique(groups))} genuine mesh superpoints', flush=True)
        with (temporary / 'infos.pkl').open('wb') as stream:
            pickle.dump(dict(metainfo=dict(classes=protocol.names, ldmr_protocol_id=protocol.identity),
                             data_list=records), stream, protocol=pickle.HIGHEST_PROTOCOL)
        provenance = dict(input=str(args.input.resolve()), input_sha256=sha256(args.input),
            protocol_id=protocol.identity, box_label_space=args.box_label_space,
            segmentator_module=str(segmentator.__file__),
            point_correspondence='exact XYZ row equality; no sampling or nearest-neighbor join', scenes=evidence)
        (temporary / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(output / 'infos.pkl')


if __name__ == '__main__':
    main()
