"""Check external-group import without accepting changed point correspondence."""
import json
import pickle

import numpy as np
import pytest

from adapters.tools.import_superpoints import import_split, verified_groups


def sources(tmp_path):
    data = tmp_path / 'legacy'
    reference = tmp_path / 'reference'
    data.mkdir()
    (reference / 'points').mkdir(parents=True)
    (reference / 'super_points').mkdir()
    scene = 'scene0001_00'
    points = np.arange(24, dtype=np.float32).reshape(4, 6)
    points.tofile(data / 'points.bin')
    points.tofile(reference / 'points' / (scene + '.bin'))
    np.array([9, 9, 12, 12], np.int64).tofile(reference / 'super_points' / (scene + '.bin'))
    np.array([5, 5, 18, 18], np.int64).tofile(data / 'semantic.bin')
    np.array([0, 0, 1, 1], np.int64).tofile(data / 'instance.bin')
    info = dict(point_cloud=dict(lidar_idx=scene, num_features=6),
        pts_path='points.bin', pts_semantic_mask_path='semantic.bin',
        pts_instance_mask_path='instance.bin', annos=dict(
            **{'class': np.array([4, 17])}, axis_align_matrix=np.eye(4),
            gt_boxes_upright_depth=np.tile([0, 0, 1, 1, 1, 1], (2, 1))))
    with (data / 'infos.pkl').open('wb') as stream:
        pickle.dump([info], stream)
    provenance = tmp_path / 'source.json'
    provenance.write_text(json.dumps(dict(source='synthetic fixture')))
    return dict(input_path=data / 'infos.pkl', data_root=data,
        reference_root=reference, output_dir=tmp_path / 'output',
        box_label_space='nyu40_zero_based', reference_provenance=provenance)


def test_import_preserves_legacy_labels_and_exact_groups(tmp_path):
    args = sources(tmp_path)
    original = {p: p.read_bytes() for d in ('legacy', 'reference')
                for p in (tmp_path / d).rglob('*') if p.is_file()}
    report = import_split(**args)
    with (args['output_dir'] / 'infos.pkl').open('rb') as stream:
        result = pickle.load(stream)
    record = result['data_list'][0]
    assert len(result['metainfo']['classes']) == 35
    assert [x['bbox_label_3d'] for x in record['instances']] == [0, 7]
    np.testing.assert_array_equal(np.fromfile(record['super_pts_path'], np.int64), [9, 9, 12, 12])
    assert report['scenes'][0]['points_sha256'] == report['scenes'][0]['reference_points_sha256']
    assert all(p.read_bytes() == content for p, content in original.items())
    with pytest.raises(FileExistsError):
        import_split(**args)


@pytest.mark.parametrize('damage', ['row_order', 'rgb', 'trailing_byte', 'missing', 'short_mask', 'negative_mask'])
def test_failed_import_never_publishes_partial_split(tmp_path, damage):
    args = sources(tmp_path)
    point = args['reference_root'] / 'points/scene0001_00.bin'
    mask = args['reference_root'] / 'super_points/scene0001_00.bin'
    if damage in ('row_order', 'rgb'):
        values = np.fromfile(point, np.float32).reshape(-1, 6)
        if damage == 'row_order':
            values = values[::-1]
        else:
            values[0, 5] += 1
        values.tofile(point)
    elif damage == 'trailing_byte':
        with point.open('ab') as stream:
            stream.write(b'0')
    elif damage == 'missing':
        mask.unlink()
    elif damage == 'short_mask':
        np.array([1, 2], np.int64).tofile(mask)
    else:
        np.array([1, 2, 3, -1], np.int64).tofile(mask)
    with pytest.raises((ValueError, FileNotFoundError)):
        import_split(**args)
    assert not args['output_dir'].exists()
    assert not list(tmp_path.glob('.import-superpoints-*'))


def test_import_rejects_source_output_and_duplicate_scenes(tmp_path):
    args = sources(tmp_path)
    with pytest.raises(ValueError, match='separate'):
        import_split(**dict(args, output_dir=args['data_root'] / 'output'))
    with args['input_path'].open('rb') as stream:
        infos = pickle.load(stream)
    with args['input_path'].open('wb') as stream:
        pickle.dump(infos * 2, stream)
    with pytest.raises(ValueError, match='unique'):
        import_split(**args)
    assert not args['output_dir'].exists()
