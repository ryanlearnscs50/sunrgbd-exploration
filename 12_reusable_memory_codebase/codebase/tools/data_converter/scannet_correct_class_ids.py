#!/usr/bin/env python
"""Convert ScanNet 40-class info files to the 1-based NYU40 class ids the
incremental configs expect.

``tools/create_data.py scannet --use-40-classes`` writes ``annos['class']`` as
0-based indices into the NYU40 table, while the rest of this codebase — the class
mappings in ``configs/_base_/class_mappings/`` and ``valid_cat_ids`` in
``mmdet3d/datasets/scannet/label_maps.py`` — treats a class id as the NYU40 id
itself, which is 1-based. This script shifts ``annos['class']`` by +1 and leaves
every other field untouched, producing the ``*_40class_corrected.pkl`` files the
configs reference.

Usage:

    python tools/data_converter/scannet_correct_class_ids.py --data-root data/scannet

Run ``tools/validate_scannet_alignment_contract.py`` afterwards to check the
result against the aligned bounding boxes on disk.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

SPLITS = ('train', 'val', 'test')
NYU40_MAX = 40


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=Path('data/scannet'),
                   help='Directory holding scannet_infos_<split>_40class.pkl')
    p.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS)
    p.add_argument('--overwrite', action='store_true',
                   help='Rewrite an existing *_corrected.pkl')
    return p.parse_args()


def correct(infos):
    """Shift annos['class'] from 0-based to 1-based NYU40 ids."""
    n_scenes = n_boxes = 0
    for info in infos:
        annos = info.get('annos')
        if not annos or 'class' not in annos:
            continue
        cls = np.asarray(annos['class'])
        if cls.size == 0:
            continue
        if cls.min() < 0:
            raise ValueError(f'negative class id {cls.min()} - unexpected input')
        if cls.max() >= NYU40_MAX:
            raise ValueError(
                f'class id {cls.max()} >= {NYU40_MAX}; these infos look like they '
                'already use 1-based NYU40 ids. Nothing to correct.')
        annos['class'] = cls + 1
        n_scenes += 1
        n_boxes += cls.size
    return n_scenes, n_boxes


def main():
    args = parse_args()
    for split in args.splits:
        src = args.data_root / f'scannet_infos_{split}_40class.pkl'
        dst = args.data_root / f'scannet_infos_{split}_40class_corrected.pkl'
        if not src.exists():
            print(f'{split:<6} skip: {src} not found')
            continue
        if dst.exists() and not args.overwrite:
            print(f'{split:<6} skip: {dst.name} exists (use --overwrite)')
            continue

        with open(src, 'rb') as f:
            infos = pickle.load(f)
        n_scenes, n_boxes = correct(infos)
        with open(dst, 'wb') as f:
            pickle.dump(infos, f)
        print(f'{split:<6} {len(infos):>5} scenes -> {dst.name}  '
              f'({n_scenes} annotated, {n_boxes} boxes shifted)')


if __name__ == '__main__':
    main()
