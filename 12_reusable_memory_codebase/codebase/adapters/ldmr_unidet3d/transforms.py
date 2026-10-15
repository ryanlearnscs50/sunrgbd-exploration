"""Explicit MMDetection3D 1.x registry integration; use only in the host env."""
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

from .protocol import ScanNet35Protocol
from .targets import stage_targets


@TRANSFORMS.register_module()
class LDMRCompactSuperpoints(BaseTransform):
    """Compact inference groups without using semantic or instance labels."""
    def transform(self, results):
        import numpy as np
        groups = np.asarray(results['sp_pts_mask'])
        if (groups.ndim != 1 or groups.dtype.kind not in 'iu'
                or len(groups) != len(results['points']) or np.any(groups < 0)):
            raise ValueError('superpoint IDs must be nonnegative and point-aligned')
        _, compact = np.unique(groups, return_inverse=True)
        results['sp_pts_mask'] = compact.astype(np.int64)
        if 'eval_ann_info' in results:
            results['eval_ann_info']['sp_pts_mask'] = results['sp_pts_mask'].copy()
        return results


@TRANSFORMS.register_module()
class LDMRScanNetTargets(BaseTransform):
    """Replace BOTH upstream semantic mapping and detection target mapping.

    Run after loading/alignment/point augmentation, before ElasticTransfrom and
    Pack3DDetInputs_. A replay data record must carry ldmr_replay_from_stage.
    This is a target transform, not a dataset selector or stage controller.
    """
    def __init__(self, stage, stage_setting='scannet35_s5_freqorder'):
        self.protocol = ScanNet35Protocol(stage_setting)
        self.protocol.seen_count(stage)
        self.stage = stage

    def transform(self, results):
        if results.get('ldmr_protocol_id') != self.protocol.identity:
            raise ValueError('missing or mismatched LDMR protocol identity')
        if results.get('is_replay', False) and 'ldmr_replay_from_stage' not in results:
            raise ValueError('replay record missing snapshot visibility')
        if len(results['points']) != len(results['pts_semantic_mask']):
            raise ValueError('points and masks are not aligned')
        if 'gt_sp_masks' in results:
            raise ValueError('LDMRScanNetTargets must replace upstream target mapping')
        targets = stage_targets(
            results['pts_semantic_mask'], results['pts_instance_mask'],
            results['sp_pts_mask'], self.protocol, self.stage,
            results.get('ldmr_replay_from_stage'),
            results.get('ldmr_inserted_mask'),
            results.get('ldmr_include_current', False))
        targets['gt_sp_masks'] = torch.from_numpy(targets['gt_sp_masks'])
        results.update(targets)
        return results


@TRANSFORMS.register_module()
class LDMRInsertObjects(BaseTransform):
    """Insert prior-stage rich crops before geometric/color augmentation.

    Load boxes/labels as well as masks and apply GlobalAlignment first. This
    adapter supports ScanNet AABBs, source-height preservation and strict
    non-overlap with visible boxes. It does not supply pseudo labels.
    """
    def __init__(self, bank_path, stage, stage_setting='scannet35_s5_freqorder',
                 max_exemplars=1, probability=1.0, max_attempts=20,
                 floor_percentile=1.0, sampling_policy="weighted_without_replacement",
                 preserve_source_height=True, collision_iou=0.0, placement_jitter=0.0):
        import numpy as np
        from .objects import load_crops, _integer
        self.protocol = ScanNet35Protocol(stage_setting)
        self.protocol.seen_count(stage)
        if stage == 1:
            raise ValueError('object replay starts at stage 2')
        self.stage = stage
        self.crops = load_crops(bank_path, self.protocol, stage - 1)
        self.max_exemplars = _integer(max_exemplars, 'max_exemplars', 1)
        self.max_attempts = _integer(max_attempts, 'max_attempts', 1)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('probability must be in [0, 1]')
        if not np.isfinite(floor_percentile) or not 0 <= floor_percentile <= 100:
            raise ValueError('floor_percentile must be in [0, 100]')
        if sampling_policy not in ("weighted_without_replacement", "class_uniform"):
            raise ValueError("unknown object sampling policy")
        if not np.isfinite(collision_iou) or not 0 <= collision_iou <= 1:
            raise ValueError("collision IoU must be in [0, 1]")
        if not np.isfinite(placement_jitter) or placement_jitter < 0:
            raise ValueError("placement jitter must be nonnegative")
        self.sampling_policy = sampling_policy
        self.preserve_source_height = preserve_source_height
        self.collision_iou = collision_iou
        self.placement_jitter = placement_jitter
        self.probability = probability
        self.floor_percentile = floor_percentile
        self.review_state = None

    def enable_reviewing(self):
        """Called before workers start; shared storage survives persistent workers."""
        if self.sampling_policy != "weighted_without_replacement":
            raise ValueError("class-uniform reference replay does not use Design-2 review weights")
        if self.review_state is not None:
            raise ValueError('object reviewing already enabled')
        self.review_state = torch.tensor(
            [0.] + [c.get('replay_weight', 1.) for c in self.crops],
            dtype=torch.float64).share_memory_()

    def apply_review(self, weights, *, review_k):
        from .reviewing import object_review_weights
        if (self.review_state is None or type(review_k) is not int
                or review_k != int(self.review_state[0]) + 1):
            raise ValueError('object review updates must be consecutive and enabled')
        values = object_review_weights(self.crops, weights)
        # The hook calls this only after the complete epoch iterator drains,
        # before the next epoch queues work. No concurrent worker reads occur.
        self.review_state[1:] = torch.tensor(values, dtype=torch.float64)
        self.review_state[0] = review_k
        return dict(review_k=review_k, crop_weights=values,
                    crop_ids=[[c['scene_id'], c['stage_id'], c['object_idx'], c['class_id']]
                              for c in self.crops])

    def transform(self, results):
        import numpy as np
        from .objects import paste_crop
        if results.get('ldmr_protocol_id') != self.protocol.identity:
            raise ValueError('missing or mismatched LDMR protocol identity')
        if results.get('is_replay', False) or 'ldmr_inserted_mask' in results:
            raise ValueError('object insertion requires an unmodified natural scene')
        results['ldmr_inserted_count'] = 0
        review = self.review_state.tolist() if self.review_state is not None else None
        if review is not None:
            results['ldmr_object_review_k'] = int(review[0])
            results['ldmr_object_draws'] = []
        if not self.crops or np.random.random() >= self.probability:
            return results
        points = results['points'].tensor.detach().cpu().numpy()
        if not len(points):
            return results
        boxes = results['gt_bboxes_3d']
        original_boxes = boxes.tensor.detach().cpu().numpy()
        occupied = np.column_stack([
            boxes.gravity_center.detach().cpu().numpy(), original_boxes[:, 3:6]])
        if boxes.with_yaw:
            raise ValueError('UniDet3D object replay currently supports ScanNet AABBs only')
        minimum, maximum = points[:, :3].min(0), points[:, :3].max(0)
        floor = float(np.percentile(points[:, 2], self.floor_percentile))
        arrays = dict(points=points, pts_semantic_mask=results['pts_semantic_mask'],
                      pts_instance_mask=results['pts_instance_mask'],
                      sp_pts_mask=results['sp_pts_mask'])
        pool = list(range(len(self.crops)))
        accepted = []
        # Worker RNG is seeded by MMEngine. Weighted draws without replacement
        # follow the maintained LDMR replay policy, rather than the reference's
        # class-uniform sampling with replacement.
        buckets = {}
        for i, crop in enumerate(self.crops):
            buckets.setdefault(crop["class_id"], []).append(i)
        classes = list(buckets)
        draws = self.max_exemplars if self.sampling_policy == "class_uniform" else min(self.max_exemplars, len(pool))
        for _ in range(draws):
            weights = np.array([review[i + 1] if review is not None else
                                self.crops[i].get('replay_weight', 1.0) for i in pool], dtype=float)
            weights /= max(float(weights.max()), 1.0)
            probs = weights / weights.sum() if weights.sum() > 0 else None
            if self.sampling_policy == "class_uniform":
                cid = classes[int(np.random.choice(len(classes)))]
                crop_index = int(np.random.choice(buckets[cid]))
            else:
                crop_index = pool.pop(int(np.random.choice(len(pool), p=probs)))
            crop = self.crops[crop_index]
            if review is not None:
                draw = dict(crop_index=crop_index, weight=review[crop_index + 1], accepted=False)
                results['ldmr_object_draws'].append(draw)
            size = np.asarray(crop['bbox'][3:6])
            low, high = minimum[:2] + size[:2] / 2, maximum[:2] - size[:2] / 2
            if np.any(low > high):
                continue
            for _ in range(self.max_attempts):
                xy = np.random.uniform(low, high)
                if self.placement_jitter:
                    xy += np.random.uniform(-self.placement_jitter, self.placement_jitter, size=2)
                    xy = np.clip(xy, low, high)
                height = crop['source_floor_offset'] if self.preserve_source_height else 0.0
                center = [xy[0], xy[1], floor + height + size[2] / 2]
                pasted = paste_crop(
                    arrays['points'], arrays['pts_semantic_mask'],
                    arrays['pts_instance_mask'], arrays['sp_pts_mask'], crop,
                    center=center, occupied_boxes=occupied, protocol=self.protocol,
                    stage=self.stage, inserted_mask=arrays.get('ldmr_inserted_mask'),
                    collision_iou=self.collision_iou)
                if pasted is not None:
                    placed = pasted.pop('placed_box')
                    occupied = np.concatenate([occupied, placed[None]])
                    arrays = pasted
                    accepted.append((placed, crop['class_id']))
                    if review is not None:
                        draw['accepted'] = True
                    break
        if accepted:
            results['points'] = results['points'].new_point(arrays.pop('points'))
            results.update(arrays)
            appended = np.zeros((len(accepted), original_boxes.shape[1]), dtype=original_boxes.dtype)
            for i, (gravity, _) in enumerate(accepted):
                appended[i, :6] = gravity
                appended[i, 2] -= gravity[5] / 2  # host storage uses bottom center
            results['gt_bboxes_3d'] = boxes.new_box(np.concatenate([original_boxes, appended]))
            labels = results['gt_labels_3d']
            if hasattr(labels, 'detach'):
                labels = labels.detach().cpu().numpy()
            results['gt_labels_3d'] = np.concatenate([
                labels, np.array([cid for _, cid in accepted], dtype=np.int64)])
            results['ldmr_inserted_count'] = len(accepted)
        return results


@TRANSFORMS.register_module()
class LDMRMergePseudoBoxes(BaseTransform):
    """Merge aligned teacher boxes after alignment, before insertion/augmentation."""
    def __init__(self, path, stage, teacher_sha256, train_info_sha256,
                 artifact_sha256, geometry_sha256, stage_setting='scannet35_s5_freqorder', policy_name='ldmr'):
        from .pseudo import load_pseudo, scannet_policy
        self.protocol = ScanNet35Protocol(stage_setting)
        self.stage = stage
        self.records = load_pseudo(path, self.protocol, stage,
            teacher_sha256=teacher_sha256, train_info_sha256=train_info_sha256,
            artifact_sha256=artifact_sha256, geometry_sha256=geometry_sha256, policy_name=policy_name)
        self.policy = scannet_policy(stage, stage_setting, policy_name)

    def transform(self, results):
        import numpy as np
        from .pseudo import select_pseudo
        if results.get('ldmr_protocol_id') != self.protocol.identity:
            raise ValueError('pseudo target protocol mismatch')
        if 'ldmr_pseudo_count' in results:
            raise ValueError('pseudo labels have already been merged')
        scene = results.get('ldmr_scene_id')
        if scene not in self.records:
            raise ValueError('teacher artifact lacks this scene, including empty predictions')
        boxes = results['gt_bboxes_3d']
        if boxes.with_yaw:
            raise ValueError('pseudo merge requires ScanNet axis-aligned boxes')
        visible = np.column_stack((boxes.gravity_center.cpu().numpy(), boxes.tensor[:, 3:6].cpu().numpy()))
        selected = select_pseudo(self.records[scene], visible,
                                 self.protocol.seen_count(self.stage - 1), self.policy)
        count = len(selected['boxes'])
        results['ldmr_pseudo_count'] = count
        if count:
            appended = boxes.tensor.new_zeros((count, boxes.tensor.shape[1]))
            appended[:, :6] = torch.as_tensor(selected['boxes'], dtype=appended.dtype, device=appended.device)
            appended[:, 2] -= appended[:, 5] / 2
            results['gt_bboxes_3d'] = boxes.new_box(torch.cat([boxes.tensor, appended]))
            results['gt_labels_3d'] = np.concatenate([results['gt_labels_3d'], selected['labels']])
        return results


@TRANSFORMS.register_module()
class LDMRBoxTargets(BaseTransform):
    """Preserve loaded/augmented box-label pairs without reading point GT masks."""
    def __init__(self, stage, stage_setting='scannet35_s5_freqorder'):
        self.protocol = ScanNet35Protocol(stage_setting)
        self.protocol.seen_count(stage)
        self.stage = stage

    def transform(self, results):
        import numpy as np
        if results.get('ldmr_protocol_id') != self.protocol.identity:
            raise ValueError('box target protocol mismatch')
        if 'elastic_coords' in results:
            raise ValueError('explicit boxes require a verified deformation policy; elastic coordinates are unsupported')
        boxes = results['gt_bboxes_3d']
        labels = np.asarray(results['gt_labels_3d'])
        if (boxes.with_yaw or labels.shape != (len(boxes),) or labels.dtype.kind not in 'iu'
                or np.any((labels < 0) | (labels >= self.protocol.seen_count(self.stage)))
                or not torch.isfinite(boxes.tensor).all() or torch.any(boxes.tensor[:, 3:6] <= 0)):
            raise ValueError('box targets require finite positive AABBs and matching seen-class labels')
        counts = [results.get(key, 0) for key in ('ldmr_pseudo_count', 'ldmr_inserted_count')]
        if any(type(count) is not int or count < 0 for count in counts) or sum(counts) > len(labels):
            raise ValueError('invalid appended supervision counts')
        replay = results.get('is_replay', False)
        if replay and 'ldmr_replay_from_stage' not in results:
            raise ValueError('replay record missing snapshot visibility')
        visible = set(self.protocol.visible_classes(
            self.stage, results.get('ldmr_replay_from_stage') if replay else None))
        if results.get('ldmr_include_current', False):
            if not replay:
                raise ValueError('merged visibility requires a replay snapshot')
            visible.update(self.protocol.visible_classes(self.stage))
        original_count = len(labels) - sum(counts)
        if any(int(label) not in visible for label in labels[:original_count]):
            raise ValueError('box targets expose GT outside saved/current visibility')
        if sum(counts) and (self.stage == 1 or np.any(labels[original_count:] >= self.protocol.seen_count(self.stage - 1))):
            raise ValueError('appended targets must belong to previous classes')
        # The loss generates distance-based query support; do not recompute
        # labels or box geometry from semantic/instance annotations.
        results = LDMRCompactSuperpoints().transform(results)
        results.pop('gt_sp_masks', None)
        return results
