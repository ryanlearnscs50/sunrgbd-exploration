"""Explicit host registration for the ScanNet-35 adapter.

Import in a UniDet3D/MMDetection3D 1.4 runtime only.
"""
from copy import deepcopy

from mmdet3d.registry import DATASETS, MODELS, TASK_UTILS
import torch
from unidet3d.scannet_dataset import ScanNetDetDataset
from unidet3d.unidet3d import UniDet3D
from unidet3d.criterion import UniMatcher

from .protocol import ScanNet35Protocol
from .transforms import LDMRScanNetTargets, LDMRInsertObjects

_PROTOCOL = ScanNet35Protocol()


@TASK_UTILS.register_module()
class LDMRUniMatcher(UniMatcher):
    """Select up to topk supported queries, including small scenes and ties."""
    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, topk, **kwargs):
        from numbers import Integral
        if isinstance(topk, bool) or not isinstance(topk, Integral) or topk < 1:
            raise ValueError('matcher topk must be a positive integer')
        labels = gt_instances.labels
        masks = gt_instances.query_masks
        if masks.ndim != 2 or masks.shape[0] != len(labels) or masks.dtype != torch.bool:
            raise ValueError('matcher query masks must be a boolean object/query matrix')
        if not len(labels) or not masks.shape[1]:
            return labels.new_empty((0,)), labels.new_empty((0,))
        cost = torch.stack([fn(pred_instances, gt_instances) for fn in self.costs]).sum(0)
        if cost.shape != masks.T.shape or not torch.isfinite(cost).all():
            raise ValueError('matcher costs must be finite and query/object aligned')
        # The upstream strict kth-value threshold drops tied positive matches
        # and requests topk+1 even when the scene has fewer queries. Stable
        # sorting breaks ties by query index, with no unsupported assignments.
        with torch.no_grad():
            order = cost.masked_fill(~masks.T, float('inf')).argsort(dim=0, stable=True)
            chosen = order[:min(int(topk), len(cost))]
            keep = torch.zeros_like(masks.T)
            keep.scatter_(0, chosen, True)
            ids = torch.argwhere(keep & masks.T)
        return ids[:, 0], ids[:, 1]


@DATASETS.register_module()
class LDMRScanNetDataset(ScanNetDetDataset):
    """Converted metadata carries stage-filtered boxes and raw point semantics."""
    METAINFO = dict(classes=_PROTOCOL.names,
                    seg_valid_class_ids=_PROTOCOL.raw_ids,
                    seg_all_class_ids=tuple(range(41)),
                    palette=[((37 * raw) % 256, (97 * raw) % 256, (173 * raw) % 256)
                             for raw in _PROTOCOL.raw_ids])

    def parse_data_info(self, info):
        if info.get('ldmr_semantic_space') != 'nyu40':
            raise ValueError('LDMRScanNetDataset requires converted raw-NYU40 metadata')
        # MMEngine replaces sample_idx with the dataset row index on retrieval.
        # Preserve the manifest scene identity separately for teacher joins.
        info['ldmr_scene_id'] = str(info['sample_idx'])
        return super().parse_data_info(info)


@MODELS.register_module()
class LDMRUniDet3D(UniDet3D):
    """Make single-dataset identity explicit instead of deriving it from paths."""
    def get_dataset(self, lidar_path):
        if self.decoder.datasets != ['scannet']:
            raise ValueError('LDMRUniDet3D adapter supports single-dataset ScanNet only')
        return 'scannet'

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        # Upstream offsets superpoint IDs and replaces box/query targets in
        # place. Own those scratch samples so repeated loss calls, subsequent
        # evaluation and failures cannot corrupt caller-owned supervision.
        # MMEngine BaseDataElement.clone() shares its nested data fields.
        samples = deepcopy(batch_data_samples)
        return super().loss(batch_inputs_dict, samples, **kwargs)

    def predict_by_feat(self, out, sp_pts_masks, points, datasets_names):
        """Use upstream NMS/trimming with bounded top-k for every batch item.

        The pinned upstream postprocessor indexes only item zero and requests
        a fixed top-k even when early stages have fewer query/class pairs.
        Keep its scores and geometry while covering those incremental cases.
        """
        import torch
        from mmdet3d.structures import DepthInstance3DBoxes
        count = len(out['cls_preds'])
        if not all(len(values) == count for values in
                   (out['bboxes'], sp_pts_masks, points, datasets_names)):
            raise ValueError('prediction batch fields have different lengths')
        results = []
        for logits, boxes, groups, xyz, dataset in zip(
                out['cls_preds'], out['bboxes'], sp_pts_masks, points, datasets_names):
            if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
                raise ValueError('nonfinite decoder output; cannot produce valid detections')
            probabilities = logits.softmax(-1)[:, :-1]
            classes = probabilities.shape[1]
            k = min(int(self.test_cfg.topk_insts), probabilities.numel())
            if k < 1 or classes < 1:
                raise ValueError('prediction requires queries, classes and positive top-k')
            scores, indices = probabilities.flatten().topk(k, sorted=True)
            labels = indices.remainder(classes)
            selected = boxes[torch.div(indices, classes, rounding_mode='floor')]
            slot = self.decoder.datasets.index(dataset)
            selected, scores, labels = self._single_scene_multiclass_nms(
                selected, scores, labels, self.fast_nms[slot], self.test_cfg.iou_thr[slot])
            if self.use_superpoints[slot]:
                trimmed = self.trim_bboxes_by_superpoints(groups, xyz, selected, labels, scores)
                for trimmed_boxes, trimmed_labels, trimmed_scores in trimmed:
                    # Upstream min/max trimming emits inf/NaN when no points
                    # support a query, and zero-volume boxes for planar or
                    # single-point support. Neither is a valid 3D detection.
                    tensor = trimmed_boxes.tensor
                    valid = (torch.isfinite(tensor).all(1)
                             & (tensor[:, 3:6] > 0).all(1)
                             & torch.isfinite(trimmed_scores))
                    results.append((trimmed_boxes[valid], trimmed_labels[valid],
                                    trimmed_scores[valid]))
            else:
                results.append((DepthInstance3DBoxes(selected, box_dim=selected.shape[1],
                    with_yaw=selected.shape[1] == 7, origin=(.5, .5, .5)), labels, scores))
        return results

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        # Upstream adds batch offsets to masks in-place. Keep caller-owned
        # local IDs intact so a repeated prediction uses the same input.
        original_groups = [sample.gt_pts_seg.sp_pts_mask for sample in batch_data_samples]
        for sample, groups in zip(batch_data_samples, original_groups):
            sample.gt_pts_seg.sp_pts_mask = groups.clone()
        try:
            results = super().predict(batch_inputs_dict, batch_data_samples, **kwargs)
        finally:
            for sample, groups in zip(batch_data_samples, original_groups):
                sample.gt_pts_seg.sp_pts_mask = groups
        for sample, points in zip(results, batch_inputs_dict['points']):
            sample.pred_instances_3d.points = points
        return results


from mmdet3d.registry import METRICS, HOOKS
from mmengine.hooks import Hook
from unidet3d.indoor_metric import IndoorMetric_


@METRICS.register_module()
class LDMRIndoorMetric(IndoorMetric_):
    def get_dataset(self, lidar_path):
        if self.datasets != ['scannet']:
            raise ValueError('LDMRIndoorMetric supports single-dataset ScanNet only')
        return 'scannet'


@HOOKS.register_module()
class LDMRProtocolCheckpointHook(Hook):
    def __init__(self, stage, stage_setting='scannet35_s5_freqorder'):
        self.protocol = ScanNet35Protocol(stage_setting)
        self.protocol.seen_count(stage)
        self.stage = stage

    def before_train(self, runner):
        if self.stage == 1:
            return
        import json
        from pathlib import Path
        from mmengine.model import is_model_wrapper
        from .checkpoints import assert_loaded_checkpoint
        from .sequence import sha256
        path = Path(runner.work_dir) / f'initialized_stage_{self.stage}.pth'
        checkpoint = torch.load(path, map_location='cpu')
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        count = assert_loaded_checkpoint(model, checkpoint, self.protocol, self.stage)
        report = dict(stage=self.stage, protocol_id=self.protocol.identity,
                      checkpoint_sha256=sha256(path), matched_tensors=count,
                      loaded_state_exact=True)
        with (Path(runner.work_dir) / 'loaded_initialization.json').open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint.setdefault('meta', {}).update(
            ldmr_protocol_id=self.protocol.identity, ldmr_stage=self.stage)


@HOOKS.register_module()
class LDMRTrajectoryCheckpointHook(Hook):
    """Capture actual running weights at explicit boundaries for LD inference.

    Independent of CheckpointHook retention. These are measurement checkpoints,
    not resumable optimizer snapshots. Online reviewing is a separate feature.
    """
    priority = 'LOW'

    def __init__(self, epochs, stage, total_epochs):
        from .collection import training_schedule
        self.epochs = training_schedule(epochs, total_epochs)
        self.stage = stage

    def _save(self, runner, epoch):
        from pathlib import Path
        path = Path(runner.work_dir) / 'ld_checkpoints'
        if (path / f'epoch_{epoch}.pth').exists():
            raise ValueError('refusing to overwrite LD measurement checkpoint')
        meta = dict(epoch=epoch)
        if epoch == 0 and self.stage > 1:
            meta['ldmr_initialized_from_stage'] = self.stage - 1
        runner.save_checkpoint(str(path), f'epoch_{epoch}.pth', meta=meta,
                               save_optimizer=False, save_param_scheduler=False)

    def before_train(self, runner):
        from mmengine.dist import get_world_size
        if get_world_size() != 1 or runner.epoch != 0 or runner.iter != 0:
            raise ValueError('LD capture currently requires fresh single-process training')
        self._save(runner, 0)

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch in self.epochs:
            self._save(runner, epoch)
