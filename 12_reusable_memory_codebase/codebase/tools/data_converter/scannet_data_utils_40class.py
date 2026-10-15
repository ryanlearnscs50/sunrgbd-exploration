# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from concurrent import futures as futures
from os import path as osp

import mmcv
import numpy as np

# Import class mapping configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
mapping_dir = os.path.join(project_root, 'configs', '_base_', 'class_mappings')
if mapping_dir not in sys.path:
    sys.path.append(mapping_dir)
from scannet_nyu40_mapping import (
    NYU40_CLASSES,
    NYU40_IDS,
    NYU40_ID_TO_NAME,
    NYU40_NAME_TO_ID
)


class ScanNetData40Class(object):
    """ScanNet data with all 40 classes.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path)
        
        # Use class mappings from external config
        self.classes = NYU40_CLASSES.copy()
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        
        # Use NYU40 IDs from config (1-40)
        self.cat_ids = np.array(NYU40_IDS)
        # Note: We store class IDs as original NYU40 IDs (1-40), not 0-based indices
        
        # Add validation using config mappings
        assert self.classes == NYU40_CLASSES, "Class list mismatch with config"
        assert len(self.cat_ids) == 40, "Expected 40 NYU40 classes"
        assert set(self.cat_ids) == set(NYU40_IDS), "NYU40 ID mismatch with config"
        
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'scannetv2_{split}.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = mmcv.list_from_file(split_file)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'scannet_instance_data_40class',
                            f'{idx}_aligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, 'scannet_instance_data_40class',
                            f'{idx}_unaligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, 'scannet_instance_data_40class',
                               f'{idx}_axis_align_matrix.npy')
        mmcv.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_images(self, idx):
        paths = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.jpg'):
                paths.append(osp.join('posed_images', idx, file))
        return paths

    def get_extrinsics(self, idx):
        extrinsics = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.txt') and not file == 'intrinsic.txt':
                extrinsics.append(np.loadtxt(osp.join(path, file)))
        return extrinsics

    def get_intrinsics(self, idx):
        matrix_file = osp.join(self.root_dir, 'posed_images', idx,
                               'intrinsic.txt')
        mmcv.check_file_exist(matrix_file)
        return np.loadtxt(matrix_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 'scannet_instance_data_40class',
                                    f'{sample_idx}_vert.npy')
            points = np.load(pts_filename)
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            points.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            # update with RGB image paths if exist
            if os.path.exists(osp.join(self.root_dir, 'posed_images')):
                info['intrinsics'] = self.get_intrinsics(sample_idx)
                all_extrinsics = self.get_extrinsics(sample_idx)
                all_img_paths = self.get_images(sample_idx)
                # some poses in ScanNet are invalid
                extrinsics, img_paths = [], []
                for extrinsic, img_path in zip(all_extrinsics, all_img_paths):
                    if np.all(np.isfinite(extrinsic)):
                        img_paths.append(img_path)
                        extrinsics.append(extrinsic)
                info['extrinsics'] = extrinsics
                info['img_paths'] = img_paths

            if not self.test_mode:
                pts_instance_mask_path = osp.join(
                    self.root_dir, 'scannet_instance_data_40class',
                    f'{sample_idx}_ins_label.npy')
                pts_semantic_mask_path = osp.join(
                    self.root_dir, 'scannet_instance_data_40class',
                    f'{sample_idx}_sem_label.npy')

                pts_instance_mask = np.load(pts_instance_mask_path).astype(
                    np.int64)
                pts_semantic_mask = np.load(pts_semantic_mask_path).astype(
                    np.int64)

                mmcv.mkdir_or_exist(osp.join(self.root_dir, 'instance_mask_40class'))
                mmcv.mkdir_or_exist(osp.join(self.root_dir, 'semantic_mask_40class'))

                pts_instance_mask.tofile(
                    osp.join(self.root_dir, 'instance_mask_40class',
                             f'{sample_idx}.bin'))
                pts_semantic_mask.tofile(
                    osp.join(self.root_dir, 'semantic_mask_40class',
                             f'{sample_idx}.bin'))

                info['pts_instance_mask_path'] = osp.join(
                    'instance_mask_40class', f'{sample_idx}.bin')
                info['pts_semantic_mask_path'] = osp.join(
                    'semantic_mask_40class', f'{sample_idx}.bin')

            if has_label:
                annotations = {}
                # box is of shape [k, 6 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 6
                    unaligned_box = unaligned_box_label[:, :-1]
                    classes = aligned_box_label[:, -1]  # k
                    annotations['name'] = np.array([
                        NYU40_ID_TO_NAME[int(classes[i])]
                        for i in range(annotations['gt_num'])
                    ])
                    # default names are given to aligned bbox for compatibility
                    # we also save unaligned bbox info with marked names
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['unaligned_location'] = unaligned_box[:, :3]
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
                    annotations[
                        'unaligned_gt_boxes_upright_depth'] = unaligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    # Store original NYU40 IDs (1-40) directly - do not convert to indices
                    annotations['class'] = classes.astype(np.int64)
                axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


class ScanNetSegData40Class(object):
    """ScanNet dataset for segmentation task with all 40 classes.

    Args:
        data_root (str): Root path of the data.
        ann_file (str): The generated scannet infos.
        split (str, optional): Set split type of the data. Default: 'train'.
        num_points (int, optional): Number of points in each data input.
            Default: 8192.
        label_weight_func (function, optional): Function to compute the
            label weight. Default: None.
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split='train',
                 num_points=8192,
                 label_weight_func=None):
        self.data_root = data_root
        self.data_infos = mmcv.load(ann_file)
        self.split = split
        assert split in ['train', 'val', 'test']
        self.num_points = num_points

        self.all_ids = np.arange(41)  # all possible ids (0-40)
        # Use all 40 classes (1-40)
        self.cat_ids = np.arange(1, 41)
        self.ignore_index = len(self.cat_ids)

        self.cat_id2class = np.ones((self.all_ids.shape[0],), dtype=np.int) * \
            self.ignore_index
        for i, cat_id in enumerate(self.cat_ids):
            self.cat_id2class[cat_id] = i

        # label weighting function is taken from
        # https://github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py#L24
        self.label_weight_func = (lambda x: 1.0 / np.log(1.2 + x)) if \
            label_weight_func is None else label_weight_func

    def get_seg_infos(self):
        if self.split == 'test':
            return
        scene_idxs, label_weight = self.get_scene_idxs_and_label_weight()
        save_folder = osp.join(self.data_root, 'seg_info')
        mmcv.mkdir_or_exist(save_folder)
        np.save(
            osp.join(save_folder, f'{self.split}_resampled_scene_idxs_40class.npy'),
            scene_idxs)
        np.save(
            osp.join(save_folder, f'{self.split}_label_weight_40class.npy'),
            label_weight)
        print(f'{self.split} resampled scene index and label weight saved (40 classes)')

    def _convert_to_label(self, mask):
        """Convert class_id in loaded segmentation mask to label."""
        if isinstance(mask, str):
            if mask.endswith('npy'):
                mask = np.load(mask)
            else:
                mask = np.fromfile(mask, dtype=np.int64)
        label = self.cat_id2class[mask]
        return label

    def get_scene_idxs_and_label_weight(self):
        """Compute scene_idxs for data re-sampling and label weight for loss \
        calculation.

        We sample more times for scenes with more points. Label_weight is
        inversely proportional to number of class points.
        """
        num_classes = len(self.cat_ids)
        num_point_all = []
        label_weight = np.zeros((num_classes + 1, ))  # +1 for ignore
        for data_info in self.data_infos:
            pts_semantic_mask_path = osp.join(self.data_root, 
                                              data_info['pts_semantic_mask_path'])
            label = self._convert_to_label(pts_semantic_mask_path)
            num_point_all.append(label.shape[0])
            class_count, _ = np.histogram(label, bins=np.arange(num_classes + 2))
            label_weight += class_count

        # repeat scene_idx for num_scene_point // num_sample_point times
        sample_prob = np.array(num_point_all) / float(np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) / float(self.num_points))
        scene_idxs = []
        for idx in range(len(self.data_infos)):
            scene_idxs.extend([idx] * int(round(sample_prob[idx] * num_iter)))
        scene_idxs = np.array(scene_idxs).astype(np.int32)

        # calculate label weight, adopted from PointNet++
        label_weight = label_weight[:-1].astype(np.float32)
        label_weight = label_weight / label_weight.sum()
        label_weight = self.label_weight_func(label_weight).astype(np.float32)

        return scene_idxs, label_weight
