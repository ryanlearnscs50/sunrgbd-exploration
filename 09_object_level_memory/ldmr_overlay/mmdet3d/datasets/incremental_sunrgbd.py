"""
Incremental SUN RGB-D Dataset

This dataset extends SUNRGBDDataset to support class-incremental learning by
filtering annotations to the current stage classes (training) or cumulative
seen classes (evaluation).

Design notes:
- SUN RGB-D labels in the generated info pkls are already model indices (0..C-1)
  according to the configured class order (no NYU40 remapping).
- Stage splits are defined in config files and passed via `stage_definition`
  and `all_stage_definitions`.
"""

from __future__ import annotations

import copy
import heapq
import os.path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .builder import DATASETS
from .incremental_pseudo_policy import (
    is_memory_or_merged_scene,
    resolve_replay_pseudo_policy,
)
from .pseudo_label_utils import filter_pseudo_by_iou_against_gt, nms_indices_iou
from .sunrgbd_dataset import SUNRGBDDataset


def _compute_all_seen_classes(stage_definitions: List[Dict[str, Any]],
                              current_stage_id: int) -> List[int]:
    seen: List[int] = []
    for stage_def in stage_definitions:
        if int(stage_def.get('stage_id', 0)) <= int(current_stage_id):
            seen.extend([int(x) for x in stage_def.get('class_indices', [])])
    return sorted(set(seen))


def _filter_annos_by_mask(annos: Dict[str, Any],
                          keep_mask: np.ndarray) -> Dict[str, Any]:
    """Filter an `annos` dict using a boolean mask aligned with `annos['class']`."""
    if 'class' not in annos:
        return annos

    cls = np.asarray(annos['class'])
    if cls.ndim != 1:
        return annos

    if keep_mask.dtype != bool:
        keep_mask = keep_mask.astype(bool)

    if keep_mask.shape[0] != cls.shape[0]:
        return annos

    new_annos: Dict[str, Any] = {}
    k = int(cls.shape[0])
    for key, value in annos.items():
        if key == 'gt_num':
            continue
        if isinstance(value, np.ndarray) and value.shape[0] == k:
            new_annos[key] = value[keep_mask]
        elif isinstance(value, list) and len(value) == k:
            idxs = np.where(keep_mask)[0].tolist()
            new_annos[key] = [value[i] for i in idxs]
        else:
            new_annos[key] = value

    new_annos['gt_num'] = int(keep_mask.sum())
    new_annos['index'] = np.arange(new_annos['gt_num'], dtype=np.int32)
    return new_annos


def _aabb6_from_centered_depth_boxes(boxes: np.ndarray) -> np.ndarray:
    """Convert (center+dims[+yaw]) Depth boxes to axis-aligned proxy boxes.

    This helper assumes the input boxes are gravity-centred (SUNRGBD GT convention).
    If yaw is present, it converts (dx, dy) to the enclosing AABB extents.

    Args:
        boxes: (N, 6|7[+]) array with [x, y, z_center, dx, dy, dz, (yaw)]

    Returns:
        (N, 6) float32 array [x, y, z_center, dx_aabb, dy_aabb, dz]
    """
    if boxes.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Boxes must be 2D with >=6 dims; got {arr.shape}")

    if arr.shape[1] < 7:
        return arr[:, :6].astype(np.float32, copy=False)

    yaw = arr[:, 6].astype(np.float32)
    dx = arr[:, 3].astype(np.float32)
    dy = arr[:, 4].astype(np.float32)
    c = np.abs(np.cos(yaw))
    s = np.abs(np.sin(yaw))
    aabb_dx = c * dx + s * dy
    aabb_dy = s * dx + c * dy
    return np.stack([arr[:, 0], arr[:, 1], arr[:, 2], aabb_dx, aabb_dy, arr[:, 5]], axis=1).astype(
        np.float32, copy=False
    )


@DATASETS.register_module()
class IncrementalSUNRGBDDataset(SUNRGBDDataset):
    """SUNRGBD dataset wrapper for stage-wise incremental learning."""

    def __init__(self,
                 stage_definition: Optional[Dict[str, Any]] = None,
                 mappings: Optional[Dict[str, Any]] = None,
                 evaluation_mode: bool = False,
                 all_stage_definitions: Optional[List[Dict[str, Any]]] = None,
                 **kwargs):
        # Accept incremental knobs used by active pipeline.
        self.object_memory_bank = kwargs.pop('object_memory_bank', None)
        self.scene_memory_bank = kwargs.pop('scene_memory_bank', None)
        self.scene_dedup_strategy = kwargs.pop('scene_dedup_strategy', 'keep_both')
        self.use_pseudo_labels = bool(kwargs.pop('use_pseudo_labels', False))
        self.pseudo_label_config = kwargs.pop('pseudo_label_config', None)
        self.pseudo_label_dir = kwargs.pop('pseudo_label_dir', None)
        legacy_use_memory_pseudo = kwargs.pop('use_memory_pseudo_labels', None)
        legacy_memory_pseudo_cfg = kwargs.pop('memory_pseudo_label_config', None)
        self.target_memory_ratio = kwargs.pop('target_memory_ratio', None)
        self.memory_sampling_seed = kwargs.pop('memory_sampling_seed', 0)
        self.enable_gt_merge_iou = kwargs.pop('enable_gt_merge_iou', False)
        self.gt_merge_iou_thr = kwargs.pop('gt_merge_iou_thr', 0.7)
        self.replay_focus = kwargs.pop('replay_focus', None)
        self.work_dir = kwargs.pop('work_dir', None)
        self.experiment_dir = kwargs.pop('experiment_dir', None)
        self.reviewing_sampling = kwargs.pop('reviewing_sampling', None)

        self.stage_definition = stage_definition or {}
        self.mappings = mappings or {}
        self.evaluation_mode = bool(evaluation_mode)
        self.all_stage_definitions = all_stage_definitions or []

        self.stage_id = int(self.stage_definition.get('stage_id', 1))
        self.stage_classes = [int(x) for x in self.stage_definition.get('class_indices', [])]

        if legacy_use_memory_pseudo is not None or legacy_memory_pseudo_cfg is not None:
            raise ValueError(
                "Legacy SUNRGBD memory-pseudo keys are no longer supported. "
                "Use unified pseudo_label_config.apply_to_memory_scenes "
                "(with use_pseudo_labels=True)."
            )

        # target_memory_ratio is optional for SUNRGBD memory replay (downsample replay set).

        self.pseudo_label_config = self.pseudo_label_config or {}
        self.use_pseudo_labels, self.apply_pseudo_to_memory_scenes = resolve_replay_pseudo_policy(
            use_pseudo_labels=self.use_pseudo_labels,
            pseudo_label_config=self.pseudo_label_config,
            evaluation_mode=bool(self.evaluation_mode),
            stage_id=int(self.stage_id),
            dataset_name='IncrementalSUNRGBDDataset',
        )
        self.pseudo_vs_gt_iou_thr = float(self.pseudo_label_config.get('pseudo_vs_gt_iou_thr', 0.25))
        self.pseudo_nms_iou_thr = float(
            self.pseudo_label_config.get(
                'pseudo_nms_iou_thr',
                self.pseudo_label_config.get('nms_threshold', 0.3),
            )
        )
        self.max_pseudo_per_scene = int(self.pseudo_label_config.get('max_pseudo_per_scene', 100))

        if self.evaluation_mode and self.all_stage_definitions:
            self.seen_classes = _compute_all_seen_classes(self.all_stage_definitions,
                                                          self.stage_id)
        else:
            self.seen_classes = list(self.stage_classes)

        # Unified path manager (optional but recommended).
        self.paths = None
        if self.experiment_dir:
            try:
                from mmdet3d.utils.incremental_paths import IncrementalPaths
                self.paths = IncrementalPaths(self.experiment_dir)
            except Exception:
                self.paths = None

        super().__init__(**kwargs)
        self._validate_stage_definition_contract()

        # Keep the unfiltered annotations as the object-memory candidate pool.
        # This is metadata only; shared point-cloud files remain read-only.
        self.original_data_infos = copy.deepcopy(self.data_infos)
        self._object_scene_info_by_id = {
            self._extract_scene_id(info): info
            for info in self.original_data_infos
            if self._extract_scene_id(info) is not None
        }

        # Filter loaded infos in-place so that evaluation uses the correct GT subset.
        self._apply_class_filter()

        if (not self.evaluation_mode and self.stage_id > 1 and
                self.object_memory_bank is not None):
            previous_classes = _compute_all_seen_classes(
                self.all_stage_definitions, self.stage_id - 1)
            self.object_memory_bank.previous_classes = previous_classes
            self.object_memory_bank.dataset_ref = self

        # Scene replay (training only, stages 2+)
        if (not self.evaluation_mode and self.stage_id > 1 and
                self.scene_memory_bank is not None):
            self._add_scene_replay()
            self._apply_reviewing_sampling_if_enabled()

        # Inject pseudo labels pre-pipeline using unified natural/replay policy.
        if self.use_pseudo_labels and not self.evaluation_mode and self.stage_id > 1:
            self.pseudo_labels = self._load_pregenerated_pseudo_labels()
            if self.pseudo_labels:
                self._inject_pseudo_labels_pre_pipeline()

    def _validate_stage_definition_contract(self) -> None:
        """Ensure config-provided class indices are valid for current label space."""
        num_classes = int(len(self.CLASSES))
        if num_classes <= 0:
            return

        def _check_indices(indices: List[int], source: str) -> None:
            bad = sorted({int(x) for x in indices if int(x) < 0 or int(x) >= num_classes})
            if bad:
                raise ValueError(
                    f"SUNRGBD stage-definition contract violation: {source} contains "
                    f"out-of-range class indices {bad} for num_classes={num_classes}."
                )

        _check_indices(list(self.stage_classes), 'stage_definition.class_indices')
        _check_indices(list(self.seen_classes), 'seen_classes')

        for sd in self.all_stage_definitions:
            sid = int(sd.get('stage_id', -1))
            cls = [int(x) for x in sd.get('class_indices', [])]
            _check_indices(cls, f'all_stage_definitions[stage_id={sid}].class_indices')

    def _resolve_pseudo_label_file(self) -> Optional[str]:
        # 1) Explicit path from config (training script typically sets this).
        if isinstance(self.pseudo_label_config, dict):
            pregen = self.pseudo_label_config.get('pregenerated_file', None)
            if pregen:
                return str(pregen)

        # 2) Explicit directory override.
        if self.pseudo_label_dir:
            candidate = Path(str(self.pseudo_label_dir)) / f"stage_{int(self.stage_id)}_pseudo_labels.pkl"
            return str(candidate)

        # 3) Unified experiment path.
        if self.paths is not None:
            try:
                resolved = self.paths.resolve_legacy_pseudo_labels(int(self.stage_id))
                if resolved is not None:
                    return str(resolved)
                return str(self.paths.pseudo_label_file(int(self.stage_id)))
            except Exception:
                return None
        return None

    def _load_pregenerated_pseudo_labels(self) -> Dict[str, Any]:
        import pickle

        pseudo_file = self._resolve_pseudo_label_file()
        if not pseudo_file:
            raise FileNotFoundError(
                "Pseudo labels are enabled for SUNRGBD, but no pseudo label file could be resolved. "
                "Set `pseudo_label_config.pregenerated_file` or provide `experiment_dir`."
            )

        path = Path(pseudo_file)
        if not path.exists():
            raise FileNotFoundError(f"Pseudo label file not found: {pseudo_file}")

        with path.open('rb') as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Pseudo label file must contain a dict, got {type(payload)}.")
        return payload

    def _extract_pseudo_record(self, record: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(record, dict):
            raise ValueError(f"Pseudo record must be a dict, got {type(record)}")

        boxes = record.get('gt_boxes_upright_depth', None)
        labels = record.get('class', None)
        if boxes is None:
            boxes = record.get('boxes', None)
        if labels is None:
            labels = record.get('labels', None)

        scores = record.get('scores', None)
        if scores is None:
            scores = record.get('scores_3d', None)

        boxes = np.asarray(boxes if boxes is not None else np.zeros((0, 7), dtype=np.float32), dtype=np.float32)
        labels = np.asarray(labels if labels is not None else np.zeros((0,), dtype=np.int64), dtype=np.int64).reshape(-1)
        scores = np.asarray(scores if scores is not None else np.ones((labels.shape[0],), dtype=np.float32), dtype=np.float32).reshape(-1)

        if boxes.ndim != 2 or boxes.shape[1] < 6:
            raise ValueError(f"Pseudo boxes must be (N,6/7[+]), got {boxes.shape}")
        if boxes.shape[0] != labels.shape[0] or labels.shape[0] != scores.shape[0]:
            raise ValueError(
                f"Pseudo boxes/labels/scores length mismatch: boxes={boxes.shape}, labels={labels.shape}, scores={scores.shape}"
            )
        num_classes = int(len(self.CLASSES))
        invalid = labels[(labels < 0) | (labels >= num_classes)]
        if invalid.size > 0:
            uniq_invalid = sorted({int(x) for x in invalid.tolist()})
            raise ValueError(
                "Pseudo labels contain out-of-range class ids "
                f"{uniq_invalid} for num_classes={num_classes}."
            )

        # Canonicalize to SUNRGBD GT convention: gravity-centred 7D boxes.
        center_type = record.get('center_type', None)
        is_gravity = (center_type is None) or (str(center_type).lower() == 'gravity')

        if boxes.shape[1] >= 7:
            boxes7 = boxes[:, :7].copy()
        else:
            boxes7 = np.concatenate([boxes[:, :6], np.zeros((boxes.shape[0], 1), dtype=np.float32)], axis=1)

        if not is_gravity:
            # Treat input as bottom-centred: z_center = z_bottom + dz/2
            boxes7[:, 2] = boxes7[:, 2] + boxes7[:, 5] * 0.5

        return (
            boxes7.astype(np.float32, copy=False),
            labels.astype(np.int64, copy=False),
            scores.astype(np.float32, copy=False),
        )

    def _inject_pseudo_labels_pre_pipeline(self) -> None:
        pseudo = getattr(self, 'pseudo_labels', None)
        if not isinstance(pseudo, dict) or not pseudo:
            return

        injected_scenes = 0
        injected_boxes = 0

        for info in self.data_infos:
            if not isinstance(info, dict):
                continue
            if (not bool(getattr(self, 'apply_pseudo_to_memory_scenes', False))
                    and is_memory_or_merged_scene(info)):
                continue

            scene_id = self._extract_scene_id(info)
            if scene_id is None:
                continue
            if str(scene_id) not in pseudo:
                continue

            record = pseudo.get(str(scene_id))
            if record is None or str(scene_id) == '__meta__':
                continue

            boxes_p, labels_p, scores_p = self._extract_pseudo_record(record)
            if boxes_p.size == 0:
                continue

            annos = info.get('annos', {}) or {}
            if not isinstance(annos, dict):
                continue

            gt_boxes = np.asarray(
                annos.get('gt_boxes_upright_depth', np.zeros((0, 7), dtype=np.float32)),
                dtype=np.float32,
            )
            gt_labels = np.asarray(annos.get('class', np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)

            # 1) Suppress pseudo boxes that overlap current-stage GT (class-agnostic).
            gt_aabb6 = _aabb6_from_centered_depth_boxes(gt_boxes)
            pseudo_aabb6 = _aabb6_from_centered_depth_boxes(boxes_p)
            keep_mask = filter_pseudo_by_iou_against_gt(
                gt_aabb6,
                pseudo_aabb6,
                iou_thr=float(self.pseudo_vs_gt_iou_thr),
            )
            if keep_mask.size == 0 or not bool(keep_mask.any()):
                continue

            boxes_p = boxes_p[keep_mask]
            labels_p = labels_p[keep_mask]
            scores_p = scores_p[keep_mask]
            pseudo_aabb6 = pseudo_aabb6[keep_mask]

            # 2) Per-scene NMS among pseudo boxes.
            keep_idx = nms_indices_iou(pseudo_aabb6, scores_p, iou_thr=float(self.pseudo_nms_iou_thr))
            if keep_idx.size == 0:
                continue
            boxes_p = boxes_p[keep_idx]
            labels_p = labels_p[keep_idx]
            scores_p = scores_p[keep_idx]

            # 3) Cap per-scene pseudo count.
            if self.max_pseudo_per_scene is not None and self.max_pseudo_per_scene > 0:
                order = np.argsort(scores_p)[::-1]
                order = order[:int(self.max_pseudo_per_scene)]
                boxes_p = boxes_p[order]
                labels_p = labels_p[order]

            if gt_boxes.size and gt_boxes.shape[1] < boxes_p.shape[1]:
                gt_boxes = np.concatenate(
                    [gt_boxes[:, :6], np.zeros((gt_boxes.shape[0], 1), dtype=np.float32)], axis=1
                )
            new_boxes = np.concatenate([gt_boxes[:, :boxes_p.shape[1]], boxes_p], axis=0) if gt_boxes.size else boxes_p
            new_labels = np.concatenate([gt_labels, labels_p], axis=0) if gt_labels.size else labels_p

            annos['gt_boxes_upright_depth'] = new_boxes.astype(np.float32, copy=False)
            annos['class'] = new_labels.astype(np.int64, copy=False)
            annos['gt_num'] = int(new_boxes.shape[0])
            annos['index'] = np.arange(int(new_boxes.shape[0]), dtype=np.int32)
            info['annos'] = annos
            info['has_pseudo_labels'] = True
            injected_scenes += 1
            injected_boxes += int(boxes_p.shape[0])

        # Optional debug markers.
        self.pseudo_injected_pre_pipeline = bool(injected_scenes > 0)
        self.pseudo_injected_scene_count = int(injected_scenes)
        self.pseudo_injected_box_count = int(injected_boxes)

    def _apply_class_filter(self) -> None:
        allowed = set(self.seen_classes)
        if not allowed:
            return

        for info in self.data_infos:
            annos = info.get('annos', None)
            if not isinstance(annos, dict):
                continue
            gt_num = int(annos.get('gt_num', 0) or 0)
            if gt_num <= 0 or 'class' not in annos:
                continue

            cls = np.asarray(annos['class'])
            if cls.ndim != 1 or cls.shape[0] != gt_num:
                # Be tolerant to older info formats.
                cls = np.asarray(annos['class']).reshape(-1)

            keep_mask = np.isin(cls.astype(np.int64), np.array(sorted(allowed), dtype=np.int64))
            info['annos'] = _filter_annos_by_mask(annos, keep_mask)

        # Align with filter_empty_gt semantics for training:
        # after stage filtering, drop scenes that have no remaining GT objects.
        if getattr(self, 'filter_empty_gt', False) and not self.evaluation_mode:
            filtered = []
            for info in self.data_infos:
                annos = info.get('annos', None)
                if not isinstance(annos, dict):
                    continue
                if int(annos.get('gt_num', 0) or 0) > 0:
                    filtered.append(info)
            self.data_infos = filtered
            if hasattr(self, 'flag'):
                self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

    def _extract_scene_id(self, info: Dict[str, Any]) -> Optional[str]:
        if 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
            return str(info['point_cloud']['lidar_idx'])
        if 'sample_idx' in info:
            return str(info['sample_idx'])
        if 'scene_id' in info:
            return str(info['scene_id'])
        return None

    def _remove_duplicate_boxes(self,
                               boxes: np.ndarray,
                               labels: np.ndarray,
                               distance_threshold: float = 0.1) -> tuple:
        """Remove duplicate boxes by same-class center distance (simple, deterministic).

        Returns:
            (filtered_boxes, filtered_labels, keep_mask)
        """
        if boxes.size == 0:
            keep_mask = np.zeros((0,), dtype=bool)
            return boxes, labels, keep_mask
        boxes = np.asarray(boxes)
        labels = np.asarray(labels)
        keep = np.ones((boxes.shape[0],), dtype=bool)
        for i in range(boxes.shape[0]):
            if not keep[i]:
                continue
            for j in range(i + 1, boxes.shape[0]):
                if not keep[j]:
                    continue
                if int(labels[i]) != int(labels[j]):
                    continue
                dist = float(np.linalg.norm(boxes[i, :3] - boxes[j, :3]))
                if dist < float(distance_threshold):
                    keep[j] = False
        return boxes[keep], labels[keep], keep

    def _merge_scene_labels(self,
                            natural_scene: Dict[str, Any],
                            replay_scene: Dict[str, Any],
                            scene_id: str) -> Dict[str, Any]:
        """Merge annotations from natural + replay versions of the same scene."""
        merged = dict(natural_scene)
        nat_annos = natural_scene.get('annos', {}) or {}
        rep_annos = replay_scene.get('annos', {}) or {}

        nat_boxes = np.asarray(
            nat_annos.get('gt_boxes_upright_depth', np.zeros((0, 7), dtype=np.float32))
        )
        nat_labels = np.asarray(
            nat_annos.get('class', np.zeros((0,), dtype=np.int64))
        ).astype(np.int64)
        rep_boxes = np.asarray(
            rep_annos.get('gt_boxes_upright_depth', np.zeros((0, 7), dtype=np.float32))
        )
        rep_labels = np.asarray(
            rep_annos.get('class', np.zeros((0,), dtype=np.int64))
        ).astype(np.int64)

        nat_count = int(nat_labels.shape[0])
        rep_count = int(rep_labels.shape[0])

        all_boxes = np.concatenate([nat_boxes, rep_boxes], axis=0) if rep_count else nat_boxes
        all_labels = np.concatenate([nat_labels, rep_labels], axis=0) if rep_count else nat_labels

        if all_boxes.size:
            assert not self.enable_gt_merge_iou, (
                'SUNRGBD merge_labels currently supports distance-based dedup only; '
                'set enable_gt_merge_iou=False.'
            )
            merged_boxes, merged_labels, keep = self._remove_duplicate_boxes(
                all_boxes, all_labels, distance_threshold=0.1
            )
        else:
            merged_boxes = np.zeros((0, 7), dtype=np.float32)
            merged_labels = np.zeros((0,), dtype=np.int64)
            keep = np.zeros((0,), dtype=bool)

        merged_annos: Dict[str, Any] = {}
        keys = set()
        if isinstance(nat_annos, dict):
            keys.update(nat_annos.keys())
        if isinstance(rep_annos, dict):
            keys.update(rep_annos.keys())

        for key in keys:
            if key in ('gt_num', 'index'):
                continue
            if key == 'gt_boxes_upright_depth':
                merged_annos[key] = merged_boxes.astype(np.float32)
                continue
            if key == 'class':
                merged_annos[key] = merged_labels.astype(np.int64)
                continue

            nat_val = nat_annos.get(key, None) if isinstance(nat_annos, dict) else None
            rep_val = rep_annos.get(key, None) if isinstance(rep_annos, dict) else None

            nat_is_obj = isinstance(nat_val, np.ndarray) and nat_val.shape[0] == nat_count
            rep_is_obj = isinstance(rep_val, np.ndarray) and rep_val.shape[0] == rep_count

            if nat_is_obj or rep_is_obj:
                if not nat_is_obj and rep_is_obj:
                    nat_val = np.zeros((nat_count,) + rep_val.shape[1:], dtype=rep_val.dtype)
                    nat_is_obj = True
                if not rep_is_obj and nat_is_obj:
                    rep_val = np.zeros((rep_count,) + nat_val.shape[1:], dtype=nat_val.dtype)
                    rep_is_obj = True

                if nat_is_obj and rep_is_obj:
                    combined = np.concatenate([nat_val, rep_val], axis=0) if rep_count else nat_val
                    merged_annos[key] = combined[keep]
                    continue

            # Non per-object metadata: prefer natural value when available.
            merged_annos[key] = nat_val if nat_val is not None else rep_val

        merged_annos['gt_num'] = int(merged_boxes.shape[0])
        merged_annos['index'] = np.arange(int(merged_boxes.shape[0]), dtype=np.int32)
        merged['annos'] = merged_annos

        merged['is_merged'] = True
        merged['merged_scene_id'] = str(scene_id)
        # Preserve replay provenance for reviewing-aware resampling.
        # A merged scene may include multiple replay seats (multi-seat bank).
        def _as_list(x):
            if x is None:
                return []
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x]

        merged_ids = []
        merged_ids.extend(_as_list(natural_scene.get('replay_unique_ids', None)))
        merged_ids.extend(_as_list(replay_scene.get('replay_unique_ids', None)))
        rid = replay_scene.get('replay_unique_id', None)
        if rid is not None:
            merged_ids.append(str(rid))
        # Keep order deterministic but allow duplicates to be collapsed later if desired.
        merged['replay_unique_ids'] = [str(x) for x in merged_ids if str(x)]

        merged_stages = []
        merged_stages.extend(_as_list(natural_scene.get('replay_from_stages', None)))
        merged_stages.extend(_as_list(replay_scene.get('replay_from_stages', None)))
        rs = replay_scene.get('replay_from_stage', None)
        if rs is not None:
            merged_stages.append(int(rs))
        merged['replay_from_stages'] = [int(x) for x in merged_stages]
        # Keep the legacy single field for backward compatibility (best-effort).
        merged['replay_from_stage'] = (
            int(merged['replay_from_stages'][-1])
            if merged.get('replay_from_stages') else
            replay_scene.get('replay_from_stage', None)
        )
        return merged

    def _add_scene_replay(self) -> None:
        """Append replay scenes from SceneMemoryBank and apply dedup strategy."""
        assert self.scene_memory_bank is not None

        previous_classes: List[int] = []
        for sd in self.all_stage_definitions:
            if int(sd.get('stage_id', 0)) < int(self.stage_id):
                previous_classes.extend([int(x) for x in sd.get('class_indices', [])])
        previous_classes = sorted(set(previous_classes))
        if not previous_classes:
            return

        natural_scene_ids = []
        id_to_scene = {}
        for info in self.data_infos:
            sid = self._extract_scene_id(info)
            if sid is None:
                continue
            natural_scene_ids.append(sid)
            id_to_scene[sid] = info
        natural_count = int(len(self.data_infos))

        replay_scenes, replay_ids = self.scene_memory_bank.get_replay_scenes(
            previous_classes, int(self.stage_id)
        )
        if not replay_scenes:
            return

        # Optional: downsample replay scenes to hit a target ratio vs natural scenes.
        if self.target_memory_ratio is not None:
            try:
                r = float(self.target_memory_ratio)
                r = max(0.0, min(0.95, r))
                target_mem = int(round((r / max(1e-6, (1.0 - r))) * natural_count))
                if target_mem < len(replay_scenes):
                    rng = np.random.RandomState(int(self.memory_sampling_seed) + 1000 * int(self.stage_id))
                    idx = rng.choice(len(replay_scenes), size=target_mem, replace=False)
                    replay_scenes = [replay_scenes[i] for i in sorted(idx.tolist())]
            except Exception:
                pass

        if str(self.scene_dedup_strategy) == 'keep_both':
            self.data_infos.extend(replay_scenes)
            if hasattr(self, 'flag'):
                self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
            return

        if str(self.scene_dedup_strategy) != 'merge_labels':
            # Default to keep_both for unknown strategies (explicit is better than silent).
            self.data_infos.extend(replay_scenes)
            if hasattr(self, 'flag'):
                self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
            return

        # merge_labels: replace natural duplicates with merged scenes; keep replay-only scenes.
        merged_by_id: Dict[str, Dict[str, Any]] = {}
        replay_only: List[Dict[str, Any]] = []

        for rep in replay_scenes:
            sid = str(rep.get('original_scene_id') or '')
            if not sid:
                replay_only.append(rep)
                continue
            if sid in id_to_scene:
                base = merged_by_id.get(sid, id_to_scene[sid])
                merged_by_id[sid] = self._merge_scene_labels(base, rep, sid)
            else:
                replay_only.append(rep)

        if not merged_by_id and replay_only:
            self.data_infos.extend(replay_only)
            return

        new_infos = []
        for info in self.data_infos:
            sid = self._extract_scene_id(info)
            if sid is not None and sid in merged_by_id:
                continue
            new_infos.append(info)
        new_infos.extend(list(merged_by_id.values()))
        new_infos.extend(replay_only)
        self.data_infos = new_infos
        if hasattr(self, 'flag'):
            self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

    def _apply_reviewing_sampling_if_enabled(self) -> None:
        cfg = self.reviewing_sampling
        if not isinstance(cfg, dict):
            return
        if not bool(cfg.get('enabled', False)):
            return

        target_len = cfg.get('target_length', None)
        if target_len is None:
            raise ValueError('reviewing_sampling.enabled=True requires target_length.')
        target_len = int(target_len)
        if target_len <= 0:
            raise ValueError(f'Invalid reviewing_sampling.target_length={target_len}.')

        weights_by_uid = cfg.get('weights_by_replay_unique_id', {}) or {}
        if not isinstance(weights_by_uid, dict):
            raise ValueError('reviewing_sampling.weights_by_replay_unique_id must be a dict.')

        seed = cfg.get('seed', None)
        if seed is None:
            raise ValueError('reviewing_sampling.enabled=True requires seed.')
        rng = np.random.RandomState(int(seed))

        strict_memory_coverage = bool(cfg.get('strict_memory_coverage', True))

        memory_share_max = cfg.get('memory_share_max', None)
        if memory_share_max is not None:
            memory_share_max = float(memory_share_max)
            assert 0.0 < memory_share_max <= 1.0, memory_share_max

        candidates = list(self.data_infos)
        if not candidates:
            raise RuntimeError('reviewing_sampling has no candidates to sample from.')

        # Compute per-sample weights and a memory-bearing mask.
        sample_weights = np.ones((len(candidates),), dtype=np.float64)
        is_memory = np.zeros((len(candidates),), dtype=bool)
        memory_indices = []
        natural_indices = []

        for i, info in enumerate(candidates):
            if not isinstance(info, dict):
                natural_indices.append(int(i))
                continue

            if bool(info.get('is_replay', False)):
                uid = str(info.get('replay_unique_id', ''))
                if uid:
                    is_memory[i] = True
                    w = float(weights_by_uid.get(uid, 1.0))
                    sample_weights[i] = max(1.0, w)
                    memory_indices.append(int(i))
                else:
                    natural_indices.append(int(i))
                continue

            if bool(info.get('is_merged', False)):
                uids = info.get('replay_unique_ids', None) or []
                uids = [str(x) for x in uids if str(x)]
                if uids:
                    is_memory[i] = True
                    w = max(float(weights_by_uid.get(uid, 1.0)) for uid in uids)
                    sample_weights[i] = max(1.0, w)
                    memory_indices.append(int(i))
                else:
                    natural_indices.append(int(i))
                continue

            natural_indices.append(int(i))

        # Optional: cap expected memory share by scaling memory weights (loose safety cap).
        if memory_share_max is not None:
            mem_sum = float(sample_weights[is_memory].sum()) if bool(is_memory.any()) else 0.0
            nat_sum = float(sample_weights[~is_memory].sum()) if bool((~is_memory).any()) else 0.0
            denom = mem_sum + nat_sum
            if denom > 0.0 and nat_sum > 0.0:
                share = mem_sum / denom
                if share > memory_share_max and mem_sum > 0.0:
                    scale = (memory_share_max / max(1e-6, (1.0 - memory_share_max))) * (nat_sum / mem_sum)
                    sample_weights[is_memory] *= float(scale)

        # Normalize to probabilities.
        total_w = float(sample_weights.sum())
        if not np.isfinite(total_w) or total_w <= 0.0:
            raise RuntimeError('Invalid reviewing_sampling weights (sum <= 0).')

        n_mem = int(len(memory_indices))
        if n_mem > int(target_len):
            if strict_memory_coverage:
                raise RuntimeError(
                    "reviewing_sampling cannot guarantee memory coverage: "
                    f"target_length={int(target_len)} < num_memory_candidates={n_mem}."
                )
            target_len = int(n_mem)

        mem_sum = float(sample_weights[memory_indices].sum()) if n_mem > 0 else 0.0
        nat_sum = float(sample_weights[natural_indices].sum()) if natural_indices else 0.0
        denom = mem_sum + nat_sum
        if denom > 0.0 and mem_sum > 0.0:
            mem_share = float(mem_sum / denom)
        elif n_mem > 0 and not natural_indices:
            mem_share = 1.0
        else:
            mem_share = 0.0

        k_mem_target = int(round(float(target_len) * float(mem_share)))
        k_mem_target = max(0, min(int(target_len), int(k_mem_target)))
        if n_mem > 0:
            k_mem_target = max(int(k_mem_target), int(n_mem))
        k_nat_target = int(max(0, int(target_len) - int(k_mem_target)))
        k_mem_extra = int(max(0, int(k_mem_target) - int(n_mem)))

        selected_indices = []
        if n_mem > 0:
            selected_indices.extend([int(x) for x in memory_indices])
            if k_mem_extra > 0:
                mem_weights = np.asarray(
                    [float(sample_weights[int(i)]) for i in memory_indices],
                    dtype=np.float64,
                )
                mem_total = float(mem_weights.sum())
                if not np.isfinite(mem_total) or mem_total <= 0.0:
                    mem_probs = np.ones((n_mem,), dtype=np.float64) / float(n_mem)
                else:
                    mem_probs = mem_weights / mem_total
                extra_idx = rng.choice(
                    n_mem,
                    size=int(k_mem_extra),
                    replace=True,
                    p=mem_probs,
                )
                selected_indices.extend(
                    [int(memory_indices[int(j)]) for j in extra_idx.tolist()]
                )

        if k_nat_target > 0:
            if natural_indices:
                if k_nat_target <= len(natural_indices):
                    nat_idx = rng.choice(
                        len(natural_indices),
                        size=int(k_nat_target),
                        replace=False,
                    )
                    selected_indices.extend(
                        [int(natural_indices[int(j)]) for j in nat_idx.tolist()]
                    )
                else:
                    nat_weights = np.asarray(
                        [float(sample_weights[int(i)]) for i in natural_indices],
                        dtype=np.float64,
                    )
                    nat_total = float(nat_weights.sum())
                    if not np.isfinite(nat_total) or nat_total <= 0.0:
                        nat_probs = np.ones(
                            (len(natural_indices),), dtype=np.float64
                        ) / float(len(natural_indices))
                    else:
                        nat_probs = nat_weights / nat_total
                    nat_idx = rng.choice(
                        len(natural_indices),
                        size=int(k_nat_target),
                        replace=True,
                        p=nat_probs,
                    )
                    selected_indices.extend(
                        [int(natural_indices[int(j)]) for j in nat_idx.tolist()]
                    )
            elif n_mem > 0:
                mem_weights = np.asarray(
                    [float(sample_weights[int(i)]) for i in memory_indices],
                    dtype=np.float64,
                )
                mem_total = float(mem_weights.sum())
                if not np.isfinite(mem_total) or mem_total <= 0.0:
                    mem_probs = np.ones((n_mem,), dtype=np.float64) / float(n_mem)
                else:
                    mem_probs = mem_weights / mem_total
                extra_mem_idx = rng.choice(
                    n_mem,
                    size=int(k_nat_target),
                    replace=True,
                    p=mem_probs,
                )
                selected_indices.extend(
                    [int(memory_indices[int(j)]) for j in extra_mem_idx.tolist()]
                )

        if len(selected_indices) != int(target_len):
            raise RuntimeError(
                "reviewing_sampling constructed invalid sample length: "
                f"got={len(selected_indices)}, target={int(target_len)}."
            )

        rng.shuffle(selected_indices)
        self.data_infos = [candidates[int(j)] for j in selected_indices]
        memory_index_set = set(int(x) for x in memory_indices)
        seen_memory_candidates = set(
            int(j) for j in selected_indices if int(j) in memory_index_set
        )
        self.reviewing_sampling_debug = {
            'strict_memory_coverage': bool(strict_memory_coverage),
            'memory_candidate_count': int(n_mem),
            'natural_candidate_count': int(len(natural_indices)),
            'memory_target_count': int(k_mem_target),
            'memory_base_count': int(n_mem),
            'memory_extra_count': int(k_mem_extra),
            'natural_target_count': int(k_nat_target),
            'memory_seen_count': int(len(seen_memory_candidates)),
            'memory_never_seen_count': int(
                max(0, n_mem - len(seen_memory_candidates))
            ),
            'memory_coverage_ratio': (
                float(len(seen_memory_candidates) / float(n_mem))
                if n_mem > 0 else float('nan')
            ),
        }
        if hasattr(self, 'flag'):
            self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

    def update_scene_memory_bank_from_stage(self,
                                           model=None,
                                           device: str = 'cuda',
                                           *,
                                           forgetness_class_drops: Optional[Dict[int, float]] = None,
                                           underlearning_class_ap: Optional[Dict[int, float]] = None,
                                           underlearning_new_classes: Optional[List[int]] = None,
                                           learning_dynamics_forgetness_by_seat: Optional[Dict[str, Dict[int, float]]] = None,
                                           learning_dynamics_replay_priority_by_seat: Optional[Dict[str, Dict[int, float]]] = None,
                                           learning_dynamics_design1_payload: Optional[Dict[str, Any]] = None,
                                           learning_dynamics_design2_payload: Optional[Dict[str, Any]] = None,
                                           dataset_ref=None) -> None:
        """Update SceneMemoryBank after a stage finishes (memory-only baseline)."""
        if self.scene_memory_bank is None:
            return

        if dataset_ref is None:
            dataset_ref = self

        # Cumulative seen classes up to current stage
        seen_classes = _compute_all_seen_classes(self.all_stage_definitions, self.stage_id)

        # Only consider natural (non-replay) scenes for adding to the bank
        natural_scenes = [info for info in self.data_infos
                          if not bool(info.get('is_replay', False))]

        self.scene_memory_bank.add_stage_scenes(
            stage_id=int(self.stage_id),
            scene_infos=natural_scenes,
            seen_classes=seen_classes,
            mappings=self.mappings,
            dataset_ref=dataset_ref,
            scene_metrics=None,
            forgetness_class_drops=forgetness_class_drops,
            underlearning_class_ap=underlearning_class_ap,
            underlearning_new_classes=underlearning_new_classes,
            learning_dynamics_forgetness_by_seat=learning_dynamics_forgetness_by_seat,
            learning_dynamics_replay_priority_by_seat=learning_dynamics_replay_priority_by_seat,
            learning_dynamics_design1_payload=learning_dynamics_design1_payload,
            learning_dynamics_design2_payload=learning_dynamics_design2_payload,
        )

    def _load_scene_points(self, scene_id: str) -> Optional[np.ndarray]:
        """Read one SUN RGB-D point array without modifying dataset storage."""
        scene_id = str(scene_id)
        info = self._object_scene_info_by_id.get(scene_id)
        if info is None:
            return None
        pts_path = info.get('pts_path')
        if not pts_path:
            return None
        filename = osp.join(self.data_root, str(pts_path))
        if not osp.isfile(filename):
            return None
        num_features = int(info.get('point_cloud', {}).get('num_features', 6))
        points = np.fromfile(filename, dtype=np.float32)
        if num_features <= 0 or points.size % num_features != 0:
            raise ValueError(
                f'Invalid SUN RGB-D point file shape: {filename}, '
                f'{points.size} values/{num_features} features.')
        return points.reshape(-1, num_features)

    def update_memory_bank_from_stage(
            self,
            learning_dynamics_design2_payload: Optional[
                Dict[str, Any]] = None) -> None:
        """Crop current-stage SUN RGB-D objects into object memory."""
        bank = self.object_memory_bank
        if bank is None or self.evaluation_mode:
            return
        # Retain metadata access for optional source-scene reviewing. Replay
        # itself continues to use only the self-contained crops.
        bank.dataset_ref = self

        class_objects: Dict[int, List[Dict[str, Any]]] = {
            int(class_id): [] for class_id in self.stage_classes}
        for info in self.original_data_infos:
            annos = info.get('annos', {})
            labels = np.asarray(
                annos.get('class', np.empty((0,), dtype=np.int64)),
                dtype=np.int64).reshape(-1)
            boxes = np.asarray(
                annos.get('gt_boxes_upright_depth', np.empty((0, 7))),
                dtype=np.float32)
            if boxes.ndim != 2 or len(labels) != len(boxes):
                continue
            scene_id = self._extract_scene_id(info)
            if scene_id is None:
                continue
            for object_idx, (class_id, box) in enumerate(zip(labels, boxes)):
                class_id = int(class_id)
                if class_id in class_objects:
                    class_objects[class_id].append({
                        'scene_id': scene_id,
                        'object_idx': int(object_idx),
                        'class_id': class_id,
                        'bbox': box,
                        'confidence': 1.0,
                    })

        debug_dir = None
        if self.paths is not None:
            debug_dir = str(self.paths.debug_dir() / 'object_exemplars')
        elif self.work_dir:
            debug_dir = osp.join(str(self.work_dir), 'object_exemplars')

        for class_id, objects in class_objects.items():
            if bank.selection_strategy_name not in (
                    'random', 'largest_point_count', 'learning_dynamics_design2'):
                raise ValueError(
                    'SUN RGB-D object-memory population currently supports '
                    "selection_strategy in ['random', 'largest_point_count', "
                    "'learning_dynamics_design2'].")
            if bank.selection_strategy_name == 'learning_dynamics_design2':
                # Rank lightweight metadata first. Loading/cropping every object
                # in SUN RGB-D would defeat the purpose of a compact bank; walk
                # the ranking only until the class quota has enough valid crops.
                objects = bank.rank_learning_dynamics_design2(
                    objects, class_id, int(self.stage_id),
                    learning_dynamics_design2_payload)
            elif bank.selection_strategy_name == 'random':
                # Shuffle lightweight metadata, then read one source scene at a
                # time until enough valid crops exist. Failed/sparse crops are
                # replaced by later candidates instead of shrinking the bank.
                rng = np.random.RandomState(
                    bank.random_seed + 1009 * int(class_id))
                objects = [objects[int(index)]
                           for index in rng.permutation(len(objects))]

            prepared = []
            largest = bank.selection_strategy_name == 'largest_point_count'
            top_crops = []
            for candidate_index, candidate in enumerate(objects):
                obj = dict(candidate)
                scene = self._load_scene_points(obj['scene_id'])
                if scene is None:
                    continue
                crop = bank.extract_object_points(
                    scene, obj['bbox'], crop_margin=bank.crop_margin)
                if len(crop) < bank.min_points:
                    continue
                # Inspect the entire candidate pool for point-count selection.
                # Keep only the best K crops in memory; ties retain the earlier
                # metadata entry, matching the bank's stable ranking.
                rank = (len(crop), -candidate_index)
                if largest and len(top_crops) >= bank.exemplars_per_class:
                    if rank <= top_crops[0][:2]:
                        continue
                obj['points'] = crop
                source_floor = float(np.percentile(
                    scene[:, 2], bank.floor_percentile))
                obj['source_floor_offset'] = float(
                    obj['bbox'][2] - obj['bbox'][5] * 0.5 - source_floor)
                if largest:
                    entry = (rank[0], rank[1], obj)
                    if len(top_crops) < bank.exemplars_per_class:
                        heapq.heappush(top_crops, entry)
                    else:
                        heapq.heapreplace(top_crops, entry)
                else:
                    prepared.append(obj)
                if not largest and len(prepared) >= bank.exemplars_per_class:
                    break
            if largest:
                prepared = [entry[2] for entry in sorted(top_crops, reverse=True)]
            objects = prepared

            bank.add_exemplars(
                class_id,
                objects,
                debug_save_dir=debug_dir,
                stage_id=self.stage_id,
                learning_dynamics_design2_payload=(
                    learning_dynamics_design2_payload
                    if bank.selection_strategy_name == 'learning_dynamics_design2'
                    else None),
            )
        bank.print_statistics()

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 classwise=False,
                 by_epoch=True,
                 eval_purpose: Optional[str] = None):
        """Incremental evaluation on cumulative seen classes.

        This overrides the default indoor evaluation to:
        - evaluate only on classes seen so far (exclude unseen from aggregates)
        - attach per-class stage ids for readable reporting
        - support stage-cohort summary reporting in the evaluation output
        """
        from mmdet3d.core.evaluation.incremental_indoor_eval import (
            incremental_indoor_eval,
        )

        assert isinstance(results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(results[0], dict), (
            f'Expect elements in results to be dict, got {type(results[0])}.'
        )

        seen_classes = sorted({int(x) for x in getattr(self, 'seen_classes', [])})
        if not seen_classes:
            # Fallback to parent evaluation if stage info is missing.
            return super().evaluate(results, metric, iou_thr, logger, show, out_dir, pipeline)

        # Human-facing split naming (train/val/test) derived from ann_file.
        def _infer_split(ann_file: Any) -> str:
            if ann_file is None:
                return 'unknown'
            ann = str(ann_file)
            if ann in ('__memory__', ''):
                return 'memory'
            name = Path(ann).name.lower()
            if '_train_' in name or name.endswith('_train.pkl') or name.endswith('_train.pickle'):
                return 'train'
            if '_val_' in name or name.endswith('_val.pkl') or name.endswith('_val.pickle'):
                return 'val'
            if '_test_' in name or name.endswith('_test.pkl') or name.endswith('_test.pickle'):
                return 'test'
            # Common SUNRGBD naming: sunrgbd_infos_val_*.pkl
            if 'infos_train' in name:
                return 'train'
            if 'infos_val' in name:
                return 'val'
            if 'infos_test' in name:
                return 'test'
            return 'unknown'

        # Make eval split explicit in logs (train vs val/test) without spamming.
        try:
            from mmcv.utils import print_log

            if logger is not None and not getattr(self, '_logged_eval_split', False):
                ann_file = getattr(self, 'ann_file', None)
                split = _infer_split(ann_file)
                print_log(
                    "SUNRGBD Eval Context: "
                    f"split={split}, ann_file={ann_file}, stage_id={getattr(self, 'stage_id', None)}",
                    logger=logger,
                )
                self._logged_eval_split = True
        except Exception:
            pass

        gt_annos = [info.get('annos', {'gt_num': 0}) for info in self.data_infos]
        class_names = list(self.CLASSES)

        # Build class meta: class idx -> stage id (from config-provided stage splits).
        class_to_stage: Dict[int, int] = {}
        for sd in self.all_stage_definitions:
            sid = int(sd.get('stage_id', 0) or 0)
            for idx in sd.get('class_indices', []):
                class_to_stage[int(idx)] = sid

        class_meta: Dict[int, Dict[str, Any]] = {}
        for idx in seen_classes:
            if 0 <= idx < len(class_names):
                class_meta[int(idx)] = dict(stage=int(class_to_stage.get(int(idx), -1)),
                                            name=class_names[int(idx)])

        stage_idx = max(0, int(getattr(self, 'stage_id', 1)) - 1)
        eval_context = dict(
            dataset='SUNRGBD',
            split=_infer_split(getattr(self, 'ann_file', None)),
            purpose=str(eval_purpose) if eval_purpose else 'epoch_end',
            stage_id=max(0, int(getattr(self, 'stage_id', 1))),
        )
        ret_dict = incremental_indoor_eval(
            gt_annos,
            results,
            iou_thr,
            seen_classes=seen_classes,
            class_names=class_names,
            stage_idx=stage_idx,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            class_meta=class_meta,
            eval_context=eval_context,
        )

        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict
