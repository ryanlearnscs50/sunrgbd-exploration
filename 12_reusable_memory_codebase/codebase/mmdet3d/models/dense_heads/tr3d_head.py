try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.cnn import bias_init_with_prob
from mmcv.ops import nms3d, nms3d_normal
from mmcv.runner import BaseModule
from mmcv.utils import get_logger
from torch import nn
import numpy as np
from collections import defaultdict
import time

from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, build_assigner


@HEADS.register_module()
class TR3DHead(BaseModule):
    def __init__(self,
                 n_classes,
                 in_channels,
                 n_reg_outs,
                 voxel_size,
                 assigner,
                 bbox_loss=dict(type='AxisAlignedIoULoss', reduction='none'),
                 cls_loss=dict(type='FocalLoss', reduction='none'),
                 ignored_classes=None,
                 train_cfg=None,
                 test_cfg=None):
        super(TR3DHead, self).__init__()
        self.voxel_size = voxel_size
        self.assigner = build_assigner(assigner)
        self.bbox_loss = build_loss(bbox_loss)
        self.cls_loss = build_loss(cls_loss)
        self.ignored_classes = ignored_classes or []
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # INCREMENTAL LEARNING: Support for seen classes masking
        self.seen_classes_mask = None  # Will be set by training script
        self.training_classes_mask = None  # For loss computation during training
        self.evaluation_classes_mask = None  # For result filtering during evaluation
        
        # Configuration-based control for class masking
        # Defaults to True for backward compatibility with incremental learning
        # Standard training (18-class, 35-class) should explicitly set this to False
        self.enable_class_masking = train_cfg.get('enable_class_masking', True) if train_cfg else True
        
        # ASSIGNER DIAGNOSTICS: Track assignment statistics for analysis
        self.enable_assigner_diagnostics = getattr(train_cfg, 'enable_assigner_diagnostics', False)
        self.assigner_stats = defaultdict(list) if self.enable_assigner_diagnostics else None
        self.diagnostic_log_interval = getattr(train_cfg, 'diagnostic_log_interval', 100)
        
        # SCENE DISCOVERY: Optional per-scene metrics collection
        # This is disabled by default and only enabled for discovery experiments
        # Can be enabled via config or programmatically
        self.collect_scene_metrics = getattr(train_cfg, 'collect_scene_metrics', False)
        self.scene_losses = {} if self.collect_scene_metrics else None
        self.scene_gradient_norms = {} if self.collect_scene_metrics else None

        # Optional classwise classification-loss collection used by
        # post-hoc analysis scripts (disabled by default).
        self.collect_classwise_cls_loss = False
        self.classwise_cls_loss_sums = {}
        self.classwise_cls_loss_counts = {}
        
        self._init_layers(n_classes, in_channels, n_reg_outs)
    
    def set_evaluation_classes_mask(self, seen_classes):
        """Set evaluation mask for incremental learning.
        
        Args:
            seen_classes (list): List of class indices that should be evaluated
        """
        if seen_classes is not None:
            self.evaluation_classes_mask = [False] * self.num_classes
            for cls_idx in seen_classes:
                if 0 <= cls_idx < self.num_classes:
                    self.evaluation_classes_mask[cls_idx] = True
        else:
            self.evaluation_classes_mask = None

    def _init_layers(self, n_classes, in_channels, n_reg_outs):
        self.bbox_conv = ME.MinkowskiConvolution(
            in_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            in_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.n_classes = n_classes  # Track current number of classes

    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))
    
    def expand_classification_head(self, new_n_classes, logger=None):
        """Expand classification head for incremental learning.
        
        Args:
            new_n_classes (int): New total number of classes
            logger (logging.Logger, optional): If provided, log to this logger
                (so expansion info is captured in the experiment log file).
            
        Returns:
            bool: True if expansion was successful
        """
        def _log_info(msg: str) -> None:
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

        def _log_warning(msg: str) -> None:
            if logger is not None:
                logger.warning(msg)
            else:
                print(msg)

        if new_n_classes <= self.n_classes:
            _log_warning(
                f"TR3DHead: skip expansion (new_n_classes={new_n_classes} <= "
                f"current={self.n_classes})"
            )
            return False

        old_n_classes = int(self.n_classes)
        
        # Save old weights and biases with device info
        old_weight = self.cls_conv.kernel.data.clone()
        old_bias = self.cls_conv.bias.data.clone()
        old_device = old_weight.device

        _log_info(
            f"TR3DHead: expanding cls_conv {old_n_classes}->{new_n_classes} "
            f"(kernel {tuple(old_weight.shape)}; bias {tuple(old_bias.shape)}; "
            f"device={old_device})"
        )
        
        # Create new larger classification layer
        self.cls_conv = ME.MinkowskiConvolution(
            self.cls_conv.in_channels,
            new_n_classes,
            kernel_size=1,
            bias=True,
            dimension=3
        )
        
        # Move new layer to same device as old weights
        self.cls_conv = self.cls_conv.to(old_device)
        
        # Initialize new layer
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

        _log_info(
            f"    New cls_conv: kernel {tuple(self.cls_conv.kernel.shape)}; "
            f"bias {tuple(self.cls_conv.bias.shape)}"
        )
        
        # Copy old weights (handle MinkowskiEngine weight format)
        # MinkowskiEngine weights are shaped [in_channels, out_channels]
        # MinkowskiEngine bias is shaped [1, out_channels] (2D tensor)
        try:
            self.cls_conv.kernel.data[:, :old_n_classes] = old_weight
            self.cls_conv.bias.data[0, :old_n_classes] = old_bias.flatten()  # Flatten old bias to handle shape mismatch
            
            # Validate weight preservation
            weight_preserved = torch.allclose(
                self.cls_conv.kernel.data[:, :old_n_classes], 
                old_weight, 
                atol=1e-6
            )
            bias_preserved = torch.allclose(
                self.cls_conv.bias.data[0, :old_n_classes], 
                old_bias.flatten(), 
                atol=1e-6
            )
            
            if not weight_preserved or not bias_preserved:
                _log_warning(
                    "TR3DHead: weight preservation failed "
                    f"(kernel_ok={weight_preserved}, bias_ok={bias_preserved})"
                )
                return False
                
        except Exception as e:
            _log_warning(f"TR3DHead: failed to copy old weights: {e}")
            _log_warning(
                f"    Old kernel {tuple(old_weight.shape)}, old bias {tuple(old_bias.shape)}; "
                f"new kernel {tuple(self.cls_conv.kernel.shape)}, new bias {tuple(self.cls_conv.bias.shape)}"
            )
            return False
        
        # Update class count
        self.n_classes = new_n_classes

        _log_info(
            f"TR3DHead: expanded cls_conv to {new_n_classes} classes "
            f"(preserved 0–{old_n_classes - 1}, new {old_n_classes}–{new_n_classes - 1})"
        )
        
        return True

    # per level
    def _forward_single(self, x):
        reg_final = self.bbox_conv(x).features
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)
        cls_pred = self.cls_conv(x).features

        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:] * self.voxel_size)

        return bbox_preds, cls_preds, points

    def forward(self, x):
        bbox_preds, cls_preds, points = [], [], []
        for i in range(len(x)):
            bbox_pred, cls_pred, point = self._forward_single(x[i])
            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)
            points.append(point)
        return bbox_preds, cls_preds, points

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)

    # per scene
    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        
        # Perform assignment with optional diagnostics
        if self.enable_assigner_diagnostics:
            assignment_start_time = time.time()
            assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
            assignment_time = time.time() - assignment_start_time
            
            # Collect assignment diagnostics
            self._collect_assignment_diagnostics(
                assigned_ids, points, gt_bboxes, gt_labels, 
                bbox_preds, cls_preds, img_meta, assignment_time
            )
        else:
            assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
        bbox_preds = torch.cat(bbox_preds)
        cls_preds = torch.cat(cls_preds)
        points = torch.cat(points)

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
            # SAFETY: Any label outside [0, n_classes-1] is treated as background
            if cls_targets.dtype.is_floating_point:
                cls_targets = cls_targets.long()
            invalid_mask = (cls_targets < 0) | (cls_targets >= n_classes + 1)
            if invalid_mask.any():
                cls_targets = torch.where(invalid_mask, n_classes, cls_targets)
            
            # INCREMENTAL LEARNING: Apply training classes mask for loss computation
            # Only apply masking when explicitly enabled via configuration
            # Standard training (18-class, 35-class) sets enable_class_masking=False
            # Incremental learning keeps the default True and sets training_classes_mask
            if self.training_classes_mask is not None and self.enable_class_masking:
                # Filter out unseen classes from targets
                unseen_mask = torch.zeros_like(cls_targets, dtype=torch.bool)
                for i in range(min(n_classes, len(self.training_classes_mask))):  # Dynamic class count
                    if not self.training_classes_mask[i]:
                        unseen_mask |= (cls_targets == i)
                
                # Set unseen classes to background (n_classes) so they don't contribute to loss
                cls_targets = torch.where(unseen_mask, n_classes, cls_targets)
                
                # Also mask out predictions for unseen classes
                masked_cls_preds = cls_preds.clone()
                for i in range(min(n_classes, len(self.training_classes_mask))):  # Dynamic class count
                    if not self.training_classes_mask[i]:
                        # Set unseen class predictions to very negative values to suppress them
                        masked_cls_preds[:, i] = -1e6
                        
                cls_preds_for_loss = masked_cls_preds
            else:
                cls_preds_for_loss = cls_preds
            
            # ROBUST: Filter out ignored classes for loss calculation
            if self.ignored_classes and len(self.ignored_classes) > 0:
                ignored_mask = torch.zeros_like(cls_targets, dtype=torch.bool)
                # Extensible ignored class handling - works with any ignored class list
                for ignored_cls in self.ignored_classes:
                    if isinstance(ignored_cls, (int, float)) and 0 <= ignored_cls < n_classes:
                        ignored_mask |= (cls_targets == ignored_cls)
                
                # Set ignored classes to background (n_classes)
                cls_targets = torch.where(ignored_mask, n_classes, cls_targets)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)
            cls_preds_for_loss = cls_preds
        
        if pos_mask.sum() > 0:
            cls_loss = self.cls_loss(cls_preds_for_loss, cls_targets)
        else:
            cls_loss = cls_preds.new_zeros(1)
        self._accumulate_classwise_cls_loss(
            cls_loss=cls_loss,
            cls_targets=cls_targets,
            pos_mask=pos_mask,
            n_classes=n_classes,
        )

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            
            # ROBUST: Filter out ignored classes from bbox loss  
            if self.ignored_classes and len(self.ignored_classes) > 0 and len(gt_labels) > 0:
                pos_gt_labels = gt_labels[assigned_ids][pos_mask]
                valid_bbox_mask = torch.ones_like(pos_gt_labels, dtype=torch.bool)
                
                # Extensible ignored class handling - works with any ignored class list
                for ignored_cls in self.ignored_classes:
                    if isinstance(ignored_cls, (int, float)) and 0 <= ignored_cls < n_classes:
                        valid_bbox_mask &= (pos_gt_labels != ignored_cls)
                
                # CRITICAL FIX: Always ensure we have valid bbox samples
                valid_count = valid_bbox_mask.sum()
                if valid_count > 0:
                    pos_points = pos_points[valid_bbox_mask]
                    pos_bbox_preds = pos_bbox_preds[valid_bbox_mask]
                    pos_bbox_targets = pos_bbox_targets[valid_bbox_mask]
                else:
                    # Return zero bbox loss instead of None to prevent NaN in aggregation
                    bbox_loss = torch.tensor(0.0, device=points.device, dtype=torch.float32, requires_grad=True)
                    return bbox_loss, cls_loss, pos_mask
            
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
                
            # ROBUST: Calculate bbox loss with validation
            try:
                predicted_boxes = self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)
                bbox_loss = self.bbox_loss(
                    self._bbox_to_loss(predicted_boxes),
                    self._bbox_to_loss(pos_bbox_targets))
                
                # Validate bbox loss result
                if torch.isnan(bbox_loss).any() or torch.isinf(bbox_loss).any():
                    bbox_loss = torch.tensor(0.0, device=points.device, dtype=torch.float32, requires_grad=True)
                    
            except Exception as e:
                # Fallback to zero loss if bbox calculation fails
                bbox_loss = torch.tensor(0.0, device=points.device, dtype=torch.float32, requires_grad=True)
                
        else:
            # Return zero bbox loss instead of None
            bbox_loss = torch.tensor(0.0, device=points.device, dtype=torch.float32, requires_grad=True)
        return bbox_loss, cls_loss, pos_mask

    def enable_scene_metrics_collection(self):
        """Enable per-scene metrics collection for discovery experiments."""
        self.collect_scene_metrics = True
        if self.scene_losses is None:
            self.scene_losses = {}
        if self.scene_gradient_norms is None:
            self.scene_gradient_norms = {}
    
    def disable_scene_metrics_collection(self):
        """Disable per-scene metrics collection (return to normal training)."""
        self.collect_scene_metrics = False
        self.scene_losses = None
        self.scene_gradient_norms = None
    
    def get_scene_losses(self) -> dict:
        """Get collected per-scene losses."""
        return self.scene_losses if self.scene_losses is not None else {}
    
    def clear_scene_metrics(self):
        """Clear collected scene metrics."""
        if self.scene_losses is not None:
            self.scene_losses.clear()
        if self.scene_gradient_norms is not None:
            self.scene_gradient_norms.clear()

    def enable_classwise_cls_loss_collection(self):
        """Enable classwise cls-loss collection for analysis."""
        self.collect_classwise_cls_loss = True
        self.clear_classwise_cls_loss_collection()

    def disable_classwise_cls_loss_collection(self):
        """Disable classwise cls-loss collection."""
        self.collect_classwise_cls_loss = False

    def clear_classwise_cls_loss_collection(self):
        """Reset collected classwise cls-loss accumulators."""
        self.classwise_cls_loss_sums = {}
        self.classwise_cls_loss_counts = {}

    def get_classwise_cls_loss_collection(self):
        """Return accumulated per-class cls-loss statistics."""
        out = {}
        for cls_id in sorted(self.classwise_cls_loss_sums.keys()):
            cls_sum = float(self.classwise_cls_loss_sums.get(cls_id, 0.0))
            cls_count = int(self.classwise_cls_loss_counts.get(cls_id, 0))
            cls_mean = cls_sum / float(cls_count) if cls_count > 0 else None
            out[int(cls_id)] = {
                'sum': cls_sum,
                'count': cls_count,
                'mean': cls_mean,
            }
        return out

    def _accumulate_classwise_cls_loss(self, cls_loss, cls_targets, pos_mask, n_classes):
        """Accumulate per-class positive classification loss.

        The collector is analysis-only and does not affect optimization.
        """
        if not self.collect_classwise_cls_loss:
            return
        if cls_loss is None or cls_targets is None or pos_mask is None:
            return
        if not torch.is_tensor(cls_loss) or not torch.is_tensor(cls_targets) or not torch.is_tensor(pos_mask):
            return
        if int(pos_mask.sum().item()) <= 0:
            return
        if cls_loss.dim() == 0:
            # Scalar loss does not preserve per-point attribution.
            return

        # Convert raw cls loss to one scalar per point.
        if cls_loss.shape[0] == pos_mask.shape[0]:
            if cls_loss.dim() == 1:
                per_point_loss = cls_loss
            else:
                per_point_loss = cls_loss.reshape(cls_loss.shape[0], -1).sum(dim=1)
        elif cls_loss.numel() == pos_mask.shape[0]:
            per_point_loss = cls_loss.reshape(pos_mask.shape[0])
        else:
            return

        if per_point_loss.shape[0] != pos_mask.shape[0]:
            return

        pos_targets = cls_targets[pos_mask]
        pos_losses = per_point_loss[pos_mask]
        if pos_targets.numel() == 0 or pos_losses.numel() == 0:
            return

        valid_mask = (pos_targets >= 0) & (pos_targets < int(n_classes))
        if self.ignored_classes and len(self.ignored_classes) > 0:
            for ignored_cls in self.ignored_classes:
                if isinstance(ignored_cls, (int, float)):
                    valid_mask &= (pos_targets != int(ignored_cls))
        if int(valid_mask.sum().item()) <= 0:
            return

        pos_targets = pos_targets[valid_mask].detach().to(torch.long).cpu()
        pos_losses = pos_losses[valid_mask].detach().to(torch.float32).cpu()

        unique_classes = torch.unique(pos_targets).tolist()
        for cls_id_raw in unique_classes:
            cls_id = int(cls_id_raw)
            cls_mask = pos_targets == cls_id
            cls_count = int(cls_mask.sum().item())
            if cls_count <= 0:
                continue
            cls_sum = float(pos_losses[cls_mask].sum().item())
            self.classwise_cls_loss_sums[cls_id] = (
                float(self.classwise_cls_loss_sums.get(cls_id, 0.0)) + cls_sum
            )
            self.classwise_cls_loss_counts[cls_id] = (
                int(self.classwise_cls_loss_counts.get(cls_id, 0)) + cls_count
            )

    def _loss(self, bbox_preds, cls_preds, points,
              gt_bboxes, gt_labels, img_metas):
        bbox_losses, cls_losses, pos_masks = [], [], []
        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i])
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            pos_masks.append(pos_mask)
            
            # SCENE DISCOVERY: Optional per-scene loss storage
            if self.collect_scene_metrics and 'scene_id' in img_metas[i]:
                scene_id = img_metas[i]['scene_id']
                scene_loss_data = {
                    'bbox_loss': bbox_loss.item() if bbox_loss is not None else 0.0,
                    'cls_loss': cls_loss.item(),
                    'pos_points': pos_mask.sum().item() if pos_mask is not None else 0,
                    'total_points': len(points[i]) if i < len(points) else 0
                }
                self.scene_losses[scene_id] = scene_loss_data
            
        # ROBUST LOSS CALCULATION - Handle edge cases to prevent NaN
        logger = get_logger(__name__, log_level='INFO')
        
        # Handle bbox loss: use zero loss if no valid bbox losses exist
        if len(bbox_losses) > 0:
            # Ensure all bbox losses are at least 1-dimensional for concatenation
            bbox_losses_1d = []
            for bbox_loss in bbox_losses:
                if bbox_loss.dim() == 0:  # 0-dimensional (scalar)
                    bbox_losses_1d.append(bbox_loss.unsqueeze(0))
                else:
                    bbox_losses_1d.append(bbox_loss)
            
            bbox_loss_final = torch.mean(torch.cat(bbox_losses_1d))
            # Validate bbox loss is not NaN/inf
            if torch.isnan(bbox_loss_final) or torch.isinf(bbox_loss_final):
                logger.warning(f"TR3D: Invalid bbox_loss detected (NaN/inf), setting to 0.0. Valid bbox losses: {len(bbox_losses)}")
                bbox_loss_final = torch.tensor(0.0, device=bbox_loss_final.device, dtype=bbox_loss_final.dtype)
        else:
            # No valid bbox losses - use zero loss instead of NaN
            device = cls_losses[0].device if cls_losses else torch.device('cuda')
            bbox_loss_final = torch.tensor(0.0, device=device, dtype=torch.float32)
            logger.debug(f"TR3D: No valid bbox losses, using zero loss")
            
        # Handle cls loss: add epsilon to prevent division by zero
        # Ensure all cls losses are at least 1-dimensional for concatenation
        cls_losses_1d = []
        for cls_loss in cls_losses:
            if cls_loss.dim() == 0:  # 0-dimensional (scalar)
                cls_losses_1d.append(cls_loss.unsqueeze(0))
            else:
                # Handle higher dimensional tensors - flatten to 1D
                cls_losses_1d.append(cls_loss.flatten())
        
        # Ensure all pos masks are at least 1-dimensional for concatenation  
        pos_masks_1d = []
        for pos_mask in pos_masks:
            if pos_mask.dim() == 0:  # 0-dimensional (scalar)
                pos_masks_1d.append(pos_mask.unsqueeze(0))
            else:
                # Handle higher dimensional tensors - flatten to 1D  
                pos_masks_1d.append(pos_mask.flatten())
        
        cls_loss_sum = torch.sum(torch.cat(cls_losses_1d))
        pos_mask_sum = torch.sum(torch.cat(pos_masks_1d))
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        cls_loss_final = cls_loss_sum / (pos_mask_sum + epsilon)
        
        # Enhanced monitoring: log key metrics occasionally
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        # Log every 100 steps to monitor for issues
        if self._step_count % 100 == 0:
            logger.debug(f"TR3D Step {self._step_count}: bbox_losses={len(bbox_losses)}, "
                        f"pos_masks_total={pos_mask_sum.item():.1f}, "
                        f"bbox_loss={bbox_loss_final.item():.6f}, cls_loss={cls_loss_final.item():.6f}")
        
        # Validate cls loss is not NaN/inf
        if torch.isnan(cls_loss_final) or torch.isinf(cls_loss_final):
            logger.warning(f"TR3D: Invalid cls_loss detected (NaN/inf), setting to 0.0. "
                          f"cls_sum={cls_loss_sum.item():.6f}, pos_sum={pos_mask_sum.item():.1f}")
            cls_loss_final = torch.tensor(0.0, device=cls_loss_final.device, dtype=cls_loss_final.dtype)
            
        return dict(
            bbox_loss=bbox_loss_final,
            cls_loss=cls_loss_final)

    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        bbox_preds, cls_preds, points = self(x)
        return self._loss(bbox_preds, cls_preds, points,
                          gt_bboxes, gt_labels, img_metas)

    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            # Skip ignored classes (if any)
            if self.ignored_classes and i in self.ignored_classes:
                continue
            
            # INCREMENTAL LEARNING: Apply evaluation mask to filter unseen classes
            if (hasattr(self, 'evaluation_classes_mask') and 
                self.evaluation_classes_mask is not None and 
                len(self.evaluation_classes_mask) > i and 
                not self.evaluation_classes_mask[i]):
                # Skip unseen classes during evaluation
                continue
            
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        
        # INCREMENTAL LEARNING: Apply evaluation mask to zero out unseen classes
        if (hasattr(self, 'evaluation_classes_mask') and 
            self.evaluation_classes_mask is not None):
            # Convert mask to tensor and apply to scores
            mask = torch.tensor(self.evaluation_classes_mask, dtype=torch.bool, device=scores.device)
            
            # Handle dynamic head case where model has fewer classes than mask
            n_model_classes = scores.shape[1]
            if len(mask) > n_model_classes:
                # Truncate mask to match model output size
                mask = mask[:n_model_classes]
            elif len(mask) < n_model_classes:
                # Pad mask with False for additional classes (shouldn't happen in practice)
                padding = torch.zeros(n_model_classes - len(mask), dtype=torch.bool, device=mask.device)
                mask = torch.cat([mask, padding])
            
            # Zero out scores for unseen classes
            scores[:, ~mask] = 0.0
        
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg.nms_pre > 0:
            _, ids = max_scores.topk(self.test_cfg.nms_pre)
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        boxes, scores, labels = self._nms(boxes, scores, img_meta)
        return boxes, scores, labels

    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        # Handle DataContainer objects - extract data if needed
        if hasattr(img_metas, 'data'):
            # img_metas is a DataContainer, extract the actual data
            actual_img_metas = img_metas.data
        else:
            actual_img_metas = img_metas
            
        results = []
        for i in range(len(actual_img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=actual_img_metas[i])
            results.append(result)
        return results

    def forward_test(self, x, img_metas):
        bbox_preds, cls_preds, points = self(x)
        return self._get_bboxes(bbox_preds, cls_preds, points, img_metas)


@BBOX_ASSIGNERS.register_module() 
class TR3DAssigner:
    def __init__(self, top_pts_threshold, label2level):
        # top_pts_threshold: per box
        # label2level: list of len n_classes
        #     scannet: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        #     sunrgbd: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        #       s3dis: [1, 0, 1, 1, 0]
        self.top_pts_threshold = top_pts_threshold
        self.label2level = label2level

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, img_meta):
        # -> object id or -1 for each point
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)

        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        label2level = gt_labels.new_tensor(self.label2level)
        # Ensure gt_labels is long tensor for indexing
        gt_labels_long = gt_labels.long() if gt_labels.dtype != torch.long else gt_labels
        label_levels = label2level[gt_labels_long].unsqueeze(0).expand(n_points, n_boxes)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        
        # Skip ignored classes (label2level = -1) - handle case with no ignored classes
        if torch.any(label_levels == -1):
            valid_labels = label_levels != -1
            level_condition = (label_levels == point_levels) & valid_labels
        else:
            level_condition = label_levels == point_levels

        # condition 2: keep topk location per box by center distance
        center = boxes[..., :3]
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        center_distances = torch.where(level_condition, center_distances, float_max)
        topk_result = torch.topk(center_distances,
                                min(self.top_pts_threshold + 1, len(center_distances)),
                                largest=False, dim=0).values
        topk_distances = float_max if len(topk_result) == 0 else topk_result[-1]
        topk_condition = center_distances < topk_distances.unsqueeze(0)

        # condition 3.0: only closest object to point
        center_distances = torch.sum(torch.pow(center - points, 2), dim=-1)
        _, min_inds_ = center_distances.min(dim=1)

        # condition 3: min center distance to box per point
        center_distances = torch.where(topk_condition, center_distances, float_max)
        min_values, min_ids = center_distances.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)
        min_inds = torch.where(min_inds == min_inds_, min_ids, -1)

        return min_inds
    
    def _collect_assignment_diagnostics(self, assigned_ids, points, gt_bboxes, gt_labels,
                                       bbox_preds, cls_preds, img_meta, assignment_time):
        """Collect assignment statistics for diagnostic analysis.
        
        This method tracks key metrics about the assignment process that can help
        diagnose whether classification or localization is degrading during
        incremental learning.
        """
        if not self.enable_assigner_diagnostics or self.assigner_stats is None:
            return
            
        # Extract scene information
        scene_id = img_meta.get('sample_idx', 'unknown')
        
        # Concatenate predictions and points
        all_points = torch.cat(points) if isinstance(points[0], torch.Tensor) else torch.cat([p for p in points])
        all_bbox_preds = torch.cat(bbox_preds) if isinstance(bbox_preds[0], torch.Tensor) else torch.cat([p for p in bbox_preds])
        all_cls_preds = torch.cat(cls_preds) if isinstance(cls_preds[0], torch.Tensor) else torch.cat([p for p in cls_preds])
        
        # Basic assignment statistics
        n_points = len(assigned_ids)
        n_gts = len(gt_bboxes)
        pos_mask = assigned_ids >= 0
        n_positives = pos_mask.sum().item()
        pos_ratio = n_positives / n_points if n_points > 0 else 0.0
        
        # Per-class assignment statistics
        class_stats = defaultdict(int)
        class_positive_counts = defaultdict(int)
        
        if n_positives > 0 and len(gt_labels) > 0:
            positive_labels = gt_labels[assigned_ids[pos_mask]]
            for label in positive_labels:
                class_positive_counts[label.item()] += 1
        
        # Count GT objects per class
        for label in gt_labels:
            class_stats[label.item()] += 1
        
        # IoU statistics for positive assignments (simplified)
        iou_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        if n_positives > 0 and len(gt_bboxes) > 0:
            try:
                # Simplified IoU computation using center distances as proxy
                pos_points = all_points[pos_mask]
                pos_assigned_gts = assigned_ids[pos_mask]
                
                # Get centers of assigned GT boxes
                gt_centers = gt_bboxes.gravity_center[pos_assigned_gts]  # Shape: [n_pos, 3]
                
                # Compute center distances as IoU proxy
                center_distances = torch.norm(pos_points - gt_centers, dim=1)
                
                # Convert distances to IoU-like scores (closer = higher score)
                max_dist = center_distances.max() + 1e-6
                iou_proxy = 1.0 - (center_distances / max_dist)
                
                iou_stats = {
                    'mean': iou_proxy.mean().item(),
                    'std': iou_proxy.std().item(),
                    'min': iou_proxy.min().item(),
                    'max': iou_proxy.max().item()
                }
            except Exception as e:
                # If IoU computation fails, keep default values
                pass
        
        # Classification confidence statistics for positive assignments
        cls_conf_stats = {'mean': 0.0, 'std': 0.0}
        if n_positives > 0:
            try:
                pos_cls_preds = all_cls_preds[pos_mask]
                if len(gt_labels) > 0:
                    pos_gt_labels = gt_labels[assigned_ids[pos_mask]]
                    # Get confidence for correct classes
                    pos_confidences = pos_cls_preds[range(len(pos_gt_labels)), pos_gt_labels]
                    cls_conf_stats = {
                        'mean': pos_confidences.mean().item(),
                        'std': pos_confidences.std().item() if len(pos_confidences) > 1 else 0.0
                    }
            except Exception as e:
                pass
        
        # Compile diagnostic entry
        diagnostic_entry = {
            'scene_id': scene_id,
            'timestamp': time.time(),
            'assignment_time': assignment_time,
            'n_points': n_points,
            'n_gts': n_gts,
            'n_positives': n_positives,
            'pos_ratio': pos_ratio,
            'class_gt_counts': dict(class_stats),
            'class_positive_counts': dict(class_positive_counts),
            'iou_proxy_stats': iou_stats,
            'cls_confidence_stats': cls_conf_stats
        }
        
        # Store diagnostic entry
        self.assigner_stats['entries'].append(diagnostic_entry)
        
        # Periodic logging
        if len(self.assigner_stats['entries']) % self.diagnostic_log_interval == 0:
            self._log_assignment_summary()
    
    def _log_assignment_summary(self):
        """Log summary of assignment diagnostics."""
        if not self.assigner_stats or not self.assigner_stats['entries']:
            return
            
        logger = get_logger(__name__)
        entries = self.assigner_stats['entries']
        n_entries = len(entries)
        
        # Compute summary statistics
        avg_positives = np.mean([e['n_positives'] for e in entries[-self.diagnostic_log_interval:]])
        avg_pos_ratio = np.mean([e['pos_ratio'] for e in entries[-self.diagnostic_log_interval:]])
        avg_iou = np.mean([e['iou_proxy_stats']['mean'] for e in entries[-self.diagnostic_log_interval:]])
        avg_cls_conf = np.mean([e['cls_confidence_stats']['mean'] for e in entries[-self.diagnostic_log_interval:]])
        
        # Class-wise positive assignment rates
        class_positive_rates = defaultdict(list)
        for entry in entries[-self.diagnostic_log_interval:]:
            gt_counts = entry['class_gt_counts']
            pos_counts = entry['class_positive_counts']
            
            for class_id, gt_count in gt_counts.items():
                pos_count = pos_counts.get(class_id, 0)
                rate = pos_count / gt_count if gt_count > 0 else 0.0
                class_positive_rates[class_id].append(rate)
        
        # Log summary
        logger.info(
            f"TR3D Assigner Diagnostics (last {self.diagnostic_log_interval} scenes):"
        )
        logger.info(f"   Avg positives per scene: {avg_positives:.1f}")
        logger.info(f"   Avg positive ratio: {avg_pos_ratio:.3f}")
        logger.info(f"   Avg IoU proxy: {avg_iou:.3f}")
        logger.info(f"   Avg cls confidence: {avg_cls_conf:.3f}")
        
        # Log per-class statistics for most frequent classes
        frequent_classes = sorted(class_positive_rates.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        logger.info(f"   Per-class positive rates (top 5):")
        for class_id, rates in frequent_classes:
            if rates:
                avg_rate = np.mean(rates)
                logger.info(f"     Class {class_id}: {avg_rate:.3f} ({len(rates)} scenes)")
    
    def get_assignment_diagnostics(self):
        """Get collected assignment diagnostics for analysis.
        
        Returns:
            dict: Dictionary containing all collected diagnostic data
        """
        if not self.enable_assigner_diagnostics or not self.assigner_stats:
            return None
            
        return dict(self.assigner_stats)
    
    def reset_assignment_diagnostics(self):
        """Reset collected assignment diagnostics."""
        if self.enable_assigner_diagnostics and self.assigner_stats:
            self.assigner_stats.clear()
            self.assigner_stats['entries'] = []
