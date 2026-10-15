#!/usr/bin/env python3
"""Attach existing ScanNet superpoints after exact XYZRGB row verification.

Uses detection boxes and point semantics/instances only from the legacy input.
The reference dataset contributes point-row evidence and superpoint IDs only.
All requested scenes must pass; missing scenes are never silently skipped.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.data import convert_info, verify_point_files


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified_groups(points_path, reference_points, superpoints):
    """Return IDs only when every XYZRGB value has identical row correspondence."""
    paths = list(map(Path, (points_path, reference_points, superpoints)))
    size = paths[0].stat().st_size
    if not size or size % 24 or paths[1].stat().st_size != size:
        raise ValueError('point files must have equal complete XYZRGB row counts')
    points = np.fromfile(paths[0], np.float32).reshape(-1, 6)
    reference = np.fromfile(paths[1], np.float32).reshape(-1, 6)
    if not np.isfinite(points).all() or not np.array_equal(points, reference):
        raise ValueError('reference XYZRGB rows do not match exactly')
    if paths[2].stat().st_size != len(points) * 8:
        raise ValueError('superpoints must contain one int64 ID per point')
    groups = np.fromfile(paths[2], np.int64)
    if np.any(groups < 0):
        raise ValueError('superpoint IDs must be nonnegative')
    return groups


def import_split(*, input_path, data_root, reference_root, output_dir,
                 box_label_space, reference_provenance, stage_setting='scannet35_s5_freqorder'):
    """Publish a complete new split; leave existing sources and outputs untouched."""
    input_path, data_root, reference_root, output_dir, reference_provenance = map(
        lambda p: Path(p).resolve(),
        (input_path, data_root, reference_root, output_dir, reference_provenance))
    for source in (data_root, reference_root, input_path.parent):
        if output_dir == source or source in output_dir.parents or output_dir in source.parents:
            raise ValueError('output must be separate from source dataset directories')
    if output_dir.exists():
        raise FileExistsError(output_dir)
    provenance = json.loads(reference_provenance.read_text())
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('reference provenance must be a nonempty JSON object')
    with input_path.open('rb') as stream:
        infos = pickle.load(stream)  # trusted local legacy metadata
    scenes = [str(info['point_cloud']['lidar_idx']) for info in infos]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError('source scenes must be nonempty and unique')
    if any(not re.fullmatch(r'scene[0-9]{4}_[0-9]{2}', scene) for scene in scenes):
        raise ValueError('invalid ScanNet scene ID')
    protocol = ScanNet35Protocol(stage_setting)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.import-superpoints-', dir=output_dir.parent))
    # Reserve the destination exclusively, so another invocation cannot replace it.
    reserved = False
    try:
        output_dir.mkdir(exist_ok=False)
        reserved = True
        records, evidence = [], []
        for scene, info in zip(scenes, infos):
            points = data_root / info['pts_path']
            reference = reference_root / 'points' / (scene + '.bin')
            source_groups = reference_root / 'super_points' / (scene + '.bin')
            groups = verified_groups(points, reference, source_groups)
            name = scene + '.superpoints.bin'
            groups.tofile(temporary / name)
            record = convert_info(info, protocol, data_root=data_root,
                superpoint_path=temporary / name, box_label_space=box_label_space)
            verify_point_files(record)
            evidence.append(dict(scene=scene, points=len(groups),
                superpoints=len(np.unique(groups)), points_sha256=sha256(points),
                reference_points_sha256=sha256(reference),
                superpoints_sha256=sha256(source_groups),
                semantic_sha256=sha256(record['pts_semantic_mask_path']),
                instance_sha256=sha256(record['pts_instance_mask_path'])))
            record['super_pts_path'] = str(output_dir / name)
            records.append(record)
            print(f'{scene}: {len(groups)} exactly matched points', flush=True)
        with (temporary / 'infos.pkl').open('wb') as stream:
            pickle.dump(dict(metainfo=dict(classes=protocol.names,
                ldmr_protocol_id=protocol.identity), data_list=records), stream,
                protocol=pickle.HIGHEST_PROTOCOL)
        manifest = dict(protocol_id=protocol.identity, scenes=evidence,
            source_info=str(input_path), source_info_sha256=sha256(input_path),
            reference_root=str(reference_root), reference_provenance=provenance,
            reference_provenance_sha256=sha256(reference_provenance),
            box_label_space=box_label_space, source_labels='legacy input only',
            infos_sha256=sha256(temporary / 'infos.pkl'))
        (temporary / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # Replace only the empty directory this invocation reserved.
        os.rename(temporary, output_dir)
        reserved = False
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if reserved:
            output_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--reference-root', required=True, type=Path,
                        help='read-only root containing points/ and super_points/')
    parser.add_argument('--reference-provenance', required=True, type=Path,
                        help='JSON recording reference archive/source identity')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--box-label-space', required=True,
                        choices=['nyu40', 'nyu40_zero_based'])
    parser.add_argument('--stage-setting', default='scannet35_s5_freqorder')
    args = parser.parse_args()
    import_split(input_path=args.input, data_root=args.data_root,
        reference_root=args.reference_root, output_dir=args.output_dir,
        reference_provenance=args.reference_provenance,
        box_label_space=args.box_label_space, stage_setting=args.stage_setting)


if __name__ == '__main__':
    main()
