# Copyright (c) OpenMMLab. All rights reserved.
# Step C - VARIANT 3 (optional): static-teacher logit DISTILLATION for
# class-incremental TR3D. Subclass of MinkSingleStage3DDetector that holds a
# frozen stage_{t-1} teacher and adds an old-class logit-distillation term to
# forward_train (ported from SDCoT's get_distillation_loss).
#
# Design (see /data3/ryan/incremental_learning/STEP_B_DESIGN.md sec.4):
#   - Teacher = previous-stage model (k_old classes), loaded from teacher_ckpt,
#     frozen + eval. Voxel-coordinate alignment is FREE: the teacher and student
#     voxelize the SAME points, and Minkowski sparse-conv coordinate maps are a
#     deterministic function of input coords (NOT weights), so per (level, scene)
#     the two cls_preds tensors share row order one-to-one. No proposal matching.
#   - L_distill = sqrt(MSE(center(student_cls[:, :k_old]), center(teacher_cls)))
#     where center() subtracts the per-voxel mean over the class dim, exactly as
#     SDCoT mean-centers base-class logits before the sqrt-MSE.
#   - The teacher is stored list-wrapped (self._teacher = [model]) so it is NOT
#     registered as an nn.Module child: its params are excluded from the student
#     state_dict (checkpoints stay loadable by the plain detector at eval time
#     and by expand_head for the next stage) and are never optimized.
import copy

import torch
from mmcv.runner import load_checkpoint

from mmdet3d.models import DETECTORS
from .mink_single_stage import MinkSingleStage3DDetector


@DETECTORS.register_module()
class MinkSingleStage3DDetectorInc(MinkSingleStage3DDetector):
    r"""Incremental TR3D detector with static-teacher logit distillation.

    Extra Args (on top of MinkSingleStage3DDetector):
        teacher_ckpt (str, optional): path to the stage_{t-1} checkpoint. If
            None (e.g. stage 1, or at test time), the detector behaves exactly
            like the plain MinkSingleStage3DDetector (no teacher, no distill).
        n_old_classes (int): number of OLD classes (= teacher head channels =
            cumulative class count of stage t-1). The first n_old_classes
            student logit columns are distilled against the teacher.
        loss_distill_weight (float): weight on L_distill (SDCoT w_distill=1).
    """

    def __init__(self,
                 backbone,
                 head,
                 voxel_size,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 teacher_ckpt=None,
                 n_old_classes=0,
                 loss_distill_weight=1.0):
        # Capture clean copies of the sub-configs BEFORE the parent mutates
        # `head` (it does head.update(train_cfg=...)/update(test_cfg=...)).
        teacher_backbone = copy.deepcopy(backbone)
        teacher_neck = copy.deepcopy(neck)
        teacher_head = copy.deepcopy(head)

        super(MinkSingleStage3DDetectorInc, self).__init__(
            backbone=backbone,
            head=head,
            voxel_size=voxel_size,
            neck=neck,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)

        self.n_old_classes = n_old_classes
        self.loss_distill_weight = loss_distill_weight
        self._teacher = None  # list-wrapped frozen teacher (set below)

        if teacher_ckpt is not None and n_old_classes > 0:
            teacher_head['n_classes'] = n_old_classes
            teacher = MinkSingleStage3DDetector(
                backbone=teacher_backbone,
                head=teacher_head,
                voxel_size=voxel_size,
                neck=teacher_neck,
                train_cfg=train_cfg,
                test_cfg=test_cfg)
            load_checkpoint(teacher, teacher_ckpt, map_location='cpu')
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            # Hide from the module tree: excluded from state_dict / .cuda() /
            # optimizer. Moved to the data device lazily in forward_train.
            self._teacher = [teacher]

    def _distill_loss(self, student_cls_preds, teacher_cls_preds):
        """Old-class logit distillation, ported from SDCoT.

        Args:
            student_cls_preds (list[list[Tensor]]): per level, per scene,
                (N_vox, n_classes_new) student class logits.
            teacher_cls_preds (list[list[Tensor]]): per level, per scene,
                (N_vox, n_old_classes) teacher class logits.
        Returns:
            Tensor: scalar distillation loss (graph-connected zero if no voxels).
        """
        k = self.n_old_classes
        diffs = []
        for s_level, t_level in zip(student_cls_preds, teacher_cls_preds):
            for s_i, t_i in zip(s_level, t_level):
                # Teacher & student voxelize identical points -> identical sparse
                # coordinate maps -> equal row counts in matching order. Guard
                # anyway; skip a scene/level if (unexpectedly) misaligned.
                if s_i.shape[0] != t_i.shape[0] or s_i.shape[0] == 0:
                    continue
                s_old = s_i[:, :k]
                s_c = s_old - s_old.mean(dim=1, keepdim=True)
                t_c = t_i - t_i.mean(dim=1, keepdim=True)
                diffs.append((s_c - t_c).flatten())
        if len(diffs) == 0:
            # graph-connected zero so the head still gets a (zero) gradient
            zero = sum(t.sum() for level in student_cls_preds for t in level)
            return zero * 0.0
        diff = torch.cat(diffs)
        return self.loss_distill_weight * torch.sqrt((diff**2).mean() + 1e-12)

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas):
        x = self.extract_feats(points)
        bbox_preds, cls_preds, point_locs = self.head(x)
        losses = self.head._loss(bbox_preds, cls_preds, point_locs,
                                 gt_bboxes_3d, gt_labels_3d, img_metas)
        if self._teacher is not None:
            teacher = self._teacher[0]
            dev = points[0].device
            if next(teacher.parameters()).device != dev:
                teacher.to(dev)
            with torch.no_grad():
                tx = teacher.extract_feats(points)
                _, teacher_cls_preds, _ = teacher.head(tx)
            losses['distill_loss'] = self._distill_loss(cls_preds,
                                                        teacher_cls_preds)
        return losses
