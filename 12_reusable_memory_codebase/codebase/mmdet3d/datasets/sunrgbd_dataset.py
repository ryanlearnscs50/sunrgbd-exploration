# Copyright (c) OpenMMLab. All rights reserved.
from collections import Counter
from collections import OrderedDict
from os import path as osp

import numpy as np

from mmdet3d.core import show_multi_modality_result, show_result_v2
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet.core import eval_map
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class SUNRGBDDataset(Custom3DDataset):
    r"""SUNRGBD Dataset.

    This class serves as the API for experiments on the SUNRGBD Dataset.

    See the `download page <http://rgbd.cs.princeton.edu/challenge.html>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=True, use_lidar=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
            'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']
        self._validate_label_space_contract()

    @staticmethod
    def _extract_scene_id_for_contract_check(info, fallback_idx):
        if isinstance(info, dict):
            pc = info.get('point_cloud', None)
            if isinstance(pc, dict) and 'lidar_idx' in pc:
                return pc['lidar_idx']
            img = info.get('image', None)
            if isinstance(img, dict) and 'image_idx' in img:
                return img['image_idx']
        return fallback_idx

    def _validate_label_space_contract(self):
        """Fail fast when GT labels violate the configured class contract."""
        num_classes = int(len(self.CLASSES))
        if num_classes <= 0:
            return

        out_of_range = Counter()
        mismatch_total = 0
        mismatch_examples = []

        for idx, info in enumerate(self.data_infos):
            if not isinstance(info, dict):
                continue
            annos = info.get('annos', None)
            if not isinstance(annos, dict):
                continue

            labels = np.asarray(annos.get('class', []), dtype=np.int64).reshape(-1)
            if labels.size == 0:
                continue

            bad = labels[(labels < 0) | (labels >= num_classes)]
            for raw in bad.tolist():
                out_of_range[int(raw)] += 1

            names_raw = annos.get('name', None)
            if names_raw is None:
                continue

            names = np.asarray(names_raw, dtype=object).reshape(-1)
            n = min(int(labels.shape[0]), int(names.shape[0]))
            if n <= 0:
                continue

            labels_n = labels[:n]
            valid = (labels_n >= 0) & (labels_n < num_classes)
            if not bool(valid.any()):
                continue

            names_norm = np.asarray(
                [str(x).strip().lower() for x in names[:n].tolist()],
                dtype=object)
            valid_indices = np.where(valid)[0]
            expected_names = np.asarray(
                [str(self.CLASSES[int(v)]) for v in labels_n[valid_indices].tolist()],
                dtype=object)
            observed_names = names_norm[valid_indices]
            mismatch_mask = observed_names != expected_names
            mismatch_count = int(np.sum(mismatch_mask))
            if mismatch_count <= 0:
                continue

            mismatch_total += mismatch_count
            if len(mismatch_examples) >= 8:
                continue

            scene_id = self._extract_scene_id_for_contract_check(info, idx)
            mismatch_positions = valid_indices[np.where(mismatch_mask)[0]]
            for pos in mismatch_positions.tolist():
                mismatch_examples.append(
                    dict(
                        scene_id=scene_id,
                        name=str(names_norm[pos]),
                        label=int(labels_n[pos]),
                        mapped_name=str(self.CLASSES[int(labels_n[pos])]),
                    ))
                if len(mismatch_examples) >= 8:
                    break

        if not out_of_range and mismatch_total == 0:
            return

        msg = [
            "SUNRGBD label-space contract violation detected.",
            f"ann_file={self.ann_file}",
            f"num_classes={num_classes}",
        ]
        if out_of_range:
            shown = [f"{k}:{v}" for k, v in sorted(out_of_range.items())[:16]]
            msg.append(
                "out_of_range_labels(count): " + ", ".join(shown))
        if mismatch_total > 0:
            msg.append(f"name_index_mismatches={mismatch_total}")
            for item in mismatch_examples:
                msg.append(
                    "mismatch_example: "
                    f"scene_id={item['scene_id']}, "
                    f"name={item['name']}, "
                    f"label={item['label']}, "
                    f"mapped_name={item['mapped_name']}")

        if num_classes == 40:
            msg.append(
                "For SUNRGBD top-40 runs, regenerate canonical infos with "
                "`python tools/create_data.py sunrgbd --use-40-classes` and "
                "use PKLs whose `annos['class']` are strictly in 0..39.")

        raise ValueError("\n".join(msg))

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str, optional): Filename of point clouds.
                - file_name (str, optional): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - calib (dict, optional): Camera calibration info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        assert info['point_cloud']['lidar_idx'] == info['image']['image_idx']
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_lidar']:
            pts_filename = osp.join(self.data_root, info['pts_path'])
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_filename = osp.join(
                osp.join(self.data_root, 'sunrgbd_trainval'),
                info['image']['image_path'])
            input_dict['img_prefix'] = None
            input_dict['img_info'] = dict(filename=img_filename)
            calib = info['calib']
            rt_mat = calib['Rt']
            # follow Coord3DMode.convert_point
            rt_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]
                               ]) @ rt_mat.transpose(1, 0)
            depth2img = calib['K'] @ rt_mat
            input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_bboxes_3d']) == 0:
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

        if self.modality['use_camera']:
            if info['annos']['gt_num'] != 0:
                gt_bboxes_2d = info['annos']['bbox'].astype(np.float32)
            else:
                gt_bboxes_2d = np.zeros((0, 4), dtype=np.float32)
            anns_results['bboxes'] = gt_bboxes_2d
            anns_results['labels'] = gt_labels_3d

        return anns_results

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        if self.modality['use_camera']:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            # scale colors to [0, 255]
            points = points.numpy()
            points[:, 3:] *= 255

            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_corners = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            pred_corners = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result_v2(points, gt_corners, gt_labels,
                           pred_corners, pred_labels, out_dir, file_name)

            continue  # todo: REMOVE THIS LINE

            # multi-modality visualization
            if self.modality['use_camera']:
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_multi_modality_result(
                    img,
                    gt_bboxes.tensor.numpy(),
                    pred_bboxes.tensor.numpy(),
                    None,
                    out_dir,
                    file_name,
                    box_mode='depth',
                    img_metas=img_metas,
                    show=show)

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 iou_thr_2d=(0.5, ),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 classwise=False):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            iou_thr (list[float], optional): AP IoU thresholds for 3D
                evaluation. Default: (0.25, 0.5).
            iou_thr_2d (list[float], optional): AP IoU thresholds for 2D
                evaluation. Default: (0.5, ).
            classwise (bool, optional): Whether to return classwise results.
                Default: False.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        # evaluate 3D detection performance
        if isinstance(results[0], dict):
            seen = getattr(self, 'seen_classes_for_eval', None)
            if seen is not None:
                from numbers import Integral
                if (not len(seen) or any(isinstance(c, bool) or not isinstance(c, Integral)
                        or not 0 <= c < len(self.CLASSES) for c in seen)
                        or len(set(seen)) != len(seen)):
                    raise ValueError('seen_classes_for_eval must contain unique valid class indices')
                if set(seen) != set(range(len(self.CLASSES))):
                    from mmdet3d.core.evaluation import indoor_eval
                    from mmdet3d.core.evaluation.incremental_indoor_eval import (
                        _filter_gt_anno_by_seen_classes, _filter_det_anno_by_seen_classes)
                    if len(results) != len(self.data_infos):
                        raise ValueError('one prediction result is required per scene')
                    gt = [_filter_gt_anno_by_seen_classes(info['annos'], seen)
                          for info in self.data_infos]
                    predictions = [_filter_det_anno_by_seen_classes(result, seen)
                                   for result in results]
                    metrics = indoor_eval(gt, predictions, iou_thr,
                        {int(c): self.CLASSES[c] for c in sorted(seen)}, logger=logger,
                        box_type_3d=self.box_type_3d, box_mode_3d=self.box_mode_3d)
                    if show:
                        self.show(results, out_dir, pipeline=pipeline)
                    return metrics
            return super().evaluate(results, metric, iou_thr, logger, show,
                                    out_dir, pipeline)
        # evaluate 2D detection performance
        else:
            eval_results = OrderedDict()
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            iou_thr_2d = (iou_thr_2d) if isinstance(iou_thr_2d,
                                                    float) else iou_thr_2d
            for iou_thr_2d_single in iou_thr_2d:
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=None,
                    iou_thr=iou_thr_2d_single,
                    dataset=self.CLASSES,
                    logger=logger)
                eval_results['mAP_' + str(iou_thr_2d_single)] = mean_ap
            return eval_results
