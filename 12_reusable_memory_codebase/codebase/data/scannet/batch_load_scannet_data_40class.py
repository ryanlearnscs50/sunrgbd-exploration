# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batch mode in loading Scannet scenes with all 40 classes for vertices and ground truth labels
for semantic and instance segmentations.

Usage example: python ./batch_load_scannet_data_40class.py
"""
import argparse
import datetime
import os
from os import path as osp

import numpy as np
from load_scannet_data import export

# No "don't care" classes - use all 40 classes
DONOTCARE_CLASS_IDS = np.array([])
# All 40 object class IDs in ScanNet (1-40)
OBJ_CLASS_IDS = np.arange(1, 41)

MAX_NUM_POINT = 50000


def export_one_scan(scan_name,
                    output_filename_prefix,
                    max_num_point,
                    label_map_file,
                    scannet_dir,
                    test_mode=False):
    mesh_file = osp.join(scannet_dir, scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '.aggregation.json')
    seg_file = osp.join(scannet_dir, scan_name,
                        scan_name + '_vh_clean_2.0.010000.segs.json')
    # includes axisAlignment info for the train set scans.
    meta_file = osp.join(scannet_dir, scan_name, f'{scan_name}.txt')
    
    mesh_vertices, semantic_labels, instance_labels, unaligned_bboxes, \
        aligned_bboxes, instance2semantic, axis_align_matrix = export(
            mesh_file, agg_file, seg_file, meta_file, label_map_file, None,
            test_mode)

    if not test_mode:
        # No need to mask out any classes since we're using all 40
        mask = np.ones(semantic_labels.shape[0], dtype=bool)
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        print(f'Num of instances: {num_instances}')

        # No need to filter bboxes since we're using all classes
        # But we still filter out invalid bboxes (empty or malformed)
        bbox_mask = np.ones(unaligned_bboxes.shape[0], dtype=bool)
        for i in range(unaligned_bboxes.shape[0]):
            bbox = unaligned_bboxes[i]
            if bbox[3] <= 0 or bbox[4] <= 0 or bbox[5] <= 0:  # invalid dimensions
                bbox_mask[i] = False
        
        unaligned_bboxes = unaligned_bboxes[bbox_mask, :]
        aligned_bboxes = aligned_bboxes[bbox_mask, :]
        assert unaligned_bboxes.shape[0] == aligned_bboxes.shape[0]
        print(f'Num of valid instances: {unaligned_bboxes.shape[0]}')

    if max_num_point is not None:
        max_num_point = int(max_num_point)
        N = mesh_vertices.shape[0]
        if N > max_num_point:
            choices = np.random.choice(N, max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            if not test_mode:
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]

    np.save(f'{output_filename_prefix}_vert.npy', mesh_vertices)
    if not test_mode:
        np.save(f'{output_filename_prefix}_sem_label.npy', semantic_labels)
        np.save(f'{output_filename_prefix}_ins_label.npy', instance_labels)
        np.save(f'{output_filename_prefix}_unaligned_bbox.npy',
                unaligned_bboxes)
        np.save(f'{output_filename_prefix}_aligned_bbox.npy', aligned_bboxes)
        np.save(f'{output_filename_prefix}_axis_align_matrix.npy',
                axis_align_matrix)


def batch_export(max_num_point,
                 output_folder,
                 scan_names_file,
                 label_map_file,
                 scannet_dir,
                 test_mode=False):
    if test_mode and not os.path.exists(scannet_dir):
        # test data preparation is optional
        return
    if not os.path.exists(output_folder):
        print(f'Creating new data folder: {output_folder}')
        os.mkdir(output_folder)

    scan_names = [line.rstrip() for line in open(scan_names_file)]
    for scan_name in scan_names:
        print('-' * 20 + 'begin')
        print(datetime.datetime.now())
        print(scan_name)
        output_filename_prefix = osp.join(output_folder, scan_name)
        if osp.isfile(f'{output_filename_prefix}_vert.npy'):
            print('File already exists. skipping.')
            print('-' * 20 + 'done')
            continue
        try:
            export_one_scan(scan_name, output_filename_prefix, max_num_point,
                            label_map_file, scannet_dir, test_mode)
        except Exception as e:
            print(f'Failed export scan: {scan_name}')
            print(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
        print('-' * 20 + 'done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_num_point',
        default=None,
        help='The maximum number of the points.')
    parser.add_argument(
        '--output_folder',
        default='./scannet_instance_data_40class',
        help='output folder of the result.')
    parser.add_argument(
        '--train_scannet_dir', default='scans', help='scannet data directory.')
    parser.add_argument(
        '--test_scannet_dir',
        default='scans_test',
        help='scannet data directory.')
    parser.add_argument(
        '--label_map_file',
        default='meta_data/scannetv2-labels.combined.tsv',
        help='The path of the file that stores the object names and ids.')
    parser.add_argument(
        '--train_scan_names_file',
        default='meta_data/scannet_train.txt',
        help='The path of the file that stores the scan names.')
    parser.add_argument(
        '--test_scan_names_file',
        default='meta_data/scannetv2_test.txt',
        help='The path of the file that stores the scan names.')
    args = parser.parse_args()
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.train_scan_names_file,
        args.label_map_file,
        args.train_scannet_dir,
        test_mode=False)
    batch_export(
        args.max_num_point,
        args.output_folder,
        args.test_scan_names_file,
        args.label_map_file,
        args.test_scannet_dir,
        test_mode=True)


if __name__ == '__main__':
    main()
