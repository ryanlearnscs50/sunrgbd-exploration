"""
Incremental ScanNet Dataset

This dataset extends the standard ScanNet dataset to support incremental learning
by filtering training data to only include specific classes for each stage.
"""

import numpy as np
import copy
import os
import time
from typing import Dict, List, Tuple, Any, Optional
from .scannet_dataset import ScanNetDataset
from .builder import DATASETS
from .pseudo_label_generator import PseudoLabelGenerator
from .pseudo_label_utils import (
    build_canonical_pseudo_record,
    canonicalize_bottom_center_boxes,
    pairwise_aligned_iou,
    filter_pseudo_by_iou_against_gt,
    dedup_same_class_by_iou,
)
from .incremental_pseudo_policy import (
    is_memory_or_merged_scene,
    resolve_replay_pseudo_policy,
)
from mmdet3d.core.evaluation.incremental_indoor_eval import incremental_indoor_eval


@DATASETS.register_module()
class IncrementalScanNetDataset(ScanNetDataset):
    """Incremental ScanNet Dataset for stage-wise training.

    Filters the dataset to only include objects from specified classes for the current stage,
    plus optional exemplars from previous stages stored in a memory bank.
    """

    def __init__(self,
                 stage_definition=None,
                 mappings=None,
                 object_memory_bank=None,
                 scene_memory_bank=None,
                 scene_dedup_strategy='keep_both',
                 min_samples_per_class=1,
                 evaluation_mode=False,
                 all_stage_definitions=None,
                 work_dir=None,
                 experiment_dir=None,
                 use_object_memory_handler=True,
                 use_pseudo_labels=False,
                 pseudo_label_config=None,
                 pseudo_label_dir=None,
                 target_memory_ratio: float = None,
                 memory_sampling_seed: int = 0,
                 enable_gt_merge_iou: bool = False,
                 gt_merge_iou_thr: float = 0.7,
                 replay_focus: dict = None,
                 reviewing_sampling: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Args:
            stage_definition (dict): Explicit stage definition with class_indices, names, etc.
            mappings (dict): Unified mappings from incremental_mappings.py
            object_memory_bank (MemoryBank): Object-based memory bank for exemplar replay
            scene_memory_bank (SceneMemoryBank): Scene-based memory bank for full scene replay
            scene_dedup_strategy (str): How to handle duplicate scenes ('keep_both', 'prefer_replay', 'prefer_natural')
            min_samples_per_class (int): Minimum samples required per class
            evaluation_mode (bool): If True, include ALL seen classes for evaluation
            all_stage_definitions (list): All stage definitions for computing seen classes
            work_dir (str, optional): Work directory for saving debug files
            use_object_memory_handler (bool): Whether to use memory handler for object-based approach
            use_pseudo_labels (bool): Whether to generate and use pseudo labels for replay scenes
            pseudo_label_config (dict): Configuration for pseudo label generation
            **kwargs: Arguments passed to parent ScanNetDataset
        """
        self.stage_definition = stage_definition or {}
        self.mappings = mappings or {}
        self.object_memory_bank = object_memory_bank
        self.scene_memory_bank = scene_memory_bank
        self.scene_dedup_strategy = scene_dedup_strategy
        self.min_samples_per_class = min_samples_per_class
        self.evaluation_mode = evaluation_mode
        self.all_stage_definitions = all_stage_definitions or []
        # Path management: Use experiment_dir if provided, fallback to work_dir for backward compatibility
        if experiment_dir:
            from mmdet3d.utils.incremental_paths import IncrementalPaths
            self.paths = IncrementalPaths(experiment_dir)
            self.work_dir = work_dir  # Keep for backward compatibility
        else:
            self.work_dir = work_dir
            self.paths = None

        self.original_data_infos = None
        self.use_object_memory_handler = use_object_memory_handler

        # Memory replay sampling controls
        self.target_memory_ratio = target_memory_ratio
        self.memory_sampling_seed = int(memory_sampling_seed) if memory_sampling_seed is not None else 0

        # Optional GT↔GT IoU-based dedup (off by default)
        self.enable_gt_merge_iou = bool(enable_gt_merge_iou)
        self.gt_merge_iou_thr = float(gt_merge_iou_thr)

        # Pseudo labeling configuration (shared contract across incremental datasets).
        self.use_pseudo_labels = bool(use_pseudo_labels)
        self.pseudo_label_config = pseudo_label_config or {}
        self.pseudo_label_dir = pseudo_label_dir
        self.debug_mode = self.pseudo_label_config.get('debug_mode', True)
        # IoU thresholds from config (defaults preserve prior behavior)
        self.pseudo_vs_gt_iou_thr = float(self.pseudo_label_config.get('pseudo_vs_gt_iou_thr', 0.25))
        self.pseudo_nms_iou_thr = float(self.pseudo_label_config.get('nms_threshold', 0.3))

        # Dynamic head expansion: Use unified mapping from parent dataset
        self.use_sequential_gci = kwargs.get('use_sequential_gci', True)

        # Extract stage info from definition FIRST (needed for pseudo label logging)
        self.stage_classes = self.stage_definition.get('class_indices', [])
        self.stage_idx = self.stage_definition.get('stage_id', 1) - 1  # Convert to 0-based
        self.stage_name = self.stage_definition.get('stage_name', f'Stage {self.stage_idx + 1}')
        self.use_pseudo_labels, self.apply_pseudo_to_memory_scenes = resolve_replay_pseudo_policy(
            use_pseudo_labels=self.use_pseudo_labels,
            pseudo_label_config=self.pseudo_label_config,
            evaluation_mode=bool(self.evaluation_mode),
            stage_id=int(self.stage_idx + 1),
            dataset_name='IncrementalScanNetDataset',
        )

        # Optional replay focusing controls for Segment B adaptation
        # Expected keys: class_ids: List[int], scene_ids: List[str], focus_share: float in [0,1]
        self.replay_focus_config = replay_focus or None
        self.reviewing_sampling = reviewing_sampling

        # Load pseudo labels from file if available
        if self.use_pseudo_labels and stage_definition and stage_definition.get('stage_id', 1) > 1:
            # Check if using pre-generated pseudo labels (faster for experimentation)
            if self.pseudo_label_config and self.pseudo_label_config.get('pregenerated_file'):
                self.pseudo_labels = self._load_pregenerated_pseudo_labels(
                    self.pseudo_label_config['pregenerated_file']
                )
            else:
                # Use on-the-fly generation (original approach, slower but more flexible)
                self.pseudo_labels = self._load_pseudo_labels()

            # Display detailed pseudo label statistics after loading
            if self.pseudo_labels:
                self._log_pseudo_label_statistics()
        else:
            self.pseudo_labels = {}
            if stage_definition and stage_definition.get('stage_id', 1) == 1:
                print("  Pseudo Labels: N/A (Stage 1 - no previous model)")
            elif not self.use_pseudo_labels:
                if self.evaluation_mode:
                    print("  Pseudo Labels: Disabled (evaluation dataset uses ground truth)")
                else:
                    print("  Pseudo Labels: Disabled")

        # Initialize memory bank handler for edge case management
        self.memory_handler = None
        if self.object_memory_bank is not None:
            # Set dataset reference in memory bank for ScanNet re-extraction
            self.object_memory_bank.dataset_ref = self

            if self.use_object_memory_handler:
                from .object_memory_bank_handler import MemoryBankHandler
                handler_log_dir = os.path.join(work_dir, 'memory_handler_logs') if work_dir else None
                self.memory_handler = MemoryBankHandler(
                    object_memory_bank=self.object_memory_bank,
                    min_exemplars_per_class=2,
                    priority_mode='balanced',
                    log_dir=handler_log_dir
                )

        # Initialize pseudo label generator if enabled
        if self.use_pseudo_labels and self.stage_idx > 0:  # Only for stages after first
            pseudo_cache_dir = os.path.join(work_dir, 'pseudo_labels') if work_dir else None
            self.pseudo_label_generator = PseudoLabelGenerator(
                confidence_threshold=self.pseudo_label_config.get('confidence_threshold', 0.7),
                nms_threshold=self.pseudo_label_config.get('nms_threshold', 0.3),
                cache_dir=pseudo_cache_dir,
                max_pseudo_per_scene=self.pseudo_label_config.get('max_pseudo_per_scene', 50),
                debug_mode=self.pseudo_label_config.get('debug_mode', True),
                class_thresholds=self.pseudo_label_config.get('class_thresholds', None)
            )
            # Always-on detailed counters for debugging (requested)

        # Compute all seen classes if in evaluation mode
        if self.evaluation_mode and self.all_stage_definitions:
            self.all_seen_classes = []
            current_stage_id = self.stage_definition.get('stage_id', 1)
            for stage_def in self.all_stage_definitions:
                if stage_def['stage_id'] <= current_stage_id:
                    self.all_seen_classes.extend(stage_def['class_indices'])
            self.all_seen_classes = sorted(set(self.all_seen_classes))
        else:
            self.all_seen_classes = self.stage_classes.copy()

        # Filter out custom kwargs before passing to parent
        parent_kwargs = {k: v for k, v in kwargs.items()
                        if k not in ['use_sequential_gci']}

        # Set dynamic head variant if using sequential GCI
        if self.use_sequential_gci:
            parent_kwargs['variant'] = 'dynamic_head'

        # Initialize parent dataset first
        super().__init__(**parent_kwargs)

        # Note: Label conversion is now handled by the parent dataset's dynamic_head variant

        # Store original data before filtering
        self.original_data_infos = copy.deepcopy(self.data_infos)

        # Add identity tracking to all natural scenes
        for i, scene in enumerate(self.data_infos):
            self.data_infos[i] = self._add_scene_identity_tracking(scene, 'natural_only')


        # ALWAYS keep 35 classes for fixed model approach - no dynamic CLASSES update
        # The model will always expect 35 classes, but we'll filter training data
        mode_str = "EVALUATION" if self.evaluation_mode else "TRAINING"
        print(f"Incremental Dataset Setup ({mode_str}):")
        print(f"  Stage: {self.stage_name}")
        # Dynamic head: model class count grows by stage (7×stage_id)
        print(f"  Model classes: dynamic head (stage {self.stage_idx + 1})")
        if self.evaluation_mode:
            print(f"  All seen classes: {self.all_seen_classes} ({len(self.all_seen_classes)} classes)")
        else:
            print(f"  Stage classes: {self.stage_classes}")
            print(f"  Class names: {[self.mappings.get('model_idx_to_name', {}).get(i, f'class_{i}') for i in self.stage_classes]}")

        # Display pseudo labeling configuration
        if self.use_pseudo_labels:
            if self.stage_idx > 0:
                print(
                    "  Pseudo labeling: ENABLED "
                    f"(threshold: {self.pseudo_label_config.get('confidence_threshold', 0.7)}, "
                    f"apply_to_memory_scenes={bool(self.apply_pseudo_to_memory_scenes)})"
                )
            else:
                print(f"  Pseudo labeling: SKIPPED (first stage)")
        else:
            if self.evaluation_mode:
                print(f"  Pseudo labeling: DISABLED (evaluation uses ground truth)")
            else:
                print(f"  Pseudo labeling: DISABLED")

        # Filter data for current stage
        self._filter_data_for_stage()

        # Add memory bank exemplars if available and not first stage
        # Support both object-based and scene-based memory banks
        if self.stage_idx > 0 and not self.evaluation_mode:
            if self.scene_memory_bank is not None:
                # Scene-based replay (new approach)
                self._add_scene_replay()
                print(f"  Using SCENE-BASED memory bank for replay")
                # Apply correct filtering after replay scenes are added
                self._filter_mixed_dataset()
                # Optional segmented reviewing/LD resampling for k>0 segments.
                self._apply_reviewing_sampling_if_enabled()
            elif self.object_memory_bank is not None:
                # Object-based exemplars (legacy approach)
                self._add_memory_exemplars()
                print(f"  Using OBJECT-BASED memory bank for replay")

        # Inject pseudo labels PRE-PIPELINE so they undergo the same transforms
        # as GT (alignment/flip/rot/scale). This avoids coordinate mismatches
        # that occur when adding pseudo labels after the pipeline.
        self.pseudo_injected_pre_pipeline = False
        if (self.use_pseudo_labels and self.stage_idx > 0 and not self.evaluation_mode
                and hasattr(self, 'pseudo_labels') and self.pseudo_labels):
            try:
                self._inject_pseudo_labels_pre_pipeline()
                self.pseudo_injected_pre_pipeline = True
                print("  Pseudo labels: injected pre-pipeline into scene annos for consistent transforms")
            except Exception as e:
                print(f"  ⚠️ Failed to inject pseudo labels pre-pipeline: {e}")

        # Persist a compact training composition summary for analysis/debugging
        try:
            self._save_training_composition_summary()
        except Exception as _e:
            # Non-fatal; analysis-only
            print(f"  ⚠️ Failed to save training composition summary: {_e}")

    def _log_pseudo_label_statistics(self):
        """Log detailed statistics about loaded pseudo labels."""
        if not self.pseudo_labels:
            return

        # Gather comprehensive statistics
        total_scenes = len(self.pseudo_labels)
        total_detections = 0
        class_counts = {}
        confidence_stats = []
        scenes_per_class = {}

        for scene_id, labels in self.pseudo_labels.items():
            if isinstance(labels, dict) and 'labels' in labels:
                scene_labels = labels['labels']
                scene_scores = labels.get('scores', [])
                total_detections += len(scene_labels)

                # Track unique classes per scene
                unique_classes = set()
                for label in scene_labels:
                    cls_idx = int(label)
                    class_counts[cls_idx] = class_counts.get(cls_idx, 0) + 1
                    unique_classes.add(cls_idx)

                # Track which scenes have each class
                for cls_idx in unique_classes:
                    if cls_idx not in scenes_per_class:
                        scenes_per_class[cls_idx] = []
                    scenes_per_class[cls_idx].append(scene_id)

                if len(scene_scores) > 0:
                    confidence_stats.extend(scene_scores)

        # Print detailed statistics
        print(f"\n{'='*60}")
        print(f"PSEUDO LABEL STATISTICS (Stage {self.stage_definition.get('stage_id', 'Unknown')})")
        print(f"{'='*60}")
        print(f"Summary:")
        print(f"  Total scenes with labels: {total_scenes}")
        print(f"  Total detections: {total_detections:,}")
        print(f"  Average detections per scene: {total_detections/total_scenes:.1f}")

        if confidence_stats:
            conf_array = np.array(confidence_stats)
            print(f"\nConfidence Distribution:")
            print(f"  Min: {conf_array.min():.3f}")
            print(f"  25%: {np.percentile(conf_array, 25):.3f}")
            print(f"  50%: {np.percentile(conf_array, 50):.3f}")
            print(f"  75%: {np.percentile(conf_array, 75):.3f}")
            print(f"  Max: {conf_array.max():.3f}")
            print(f"  Mean: {conf_array.mean():.3f} ± {conf_array.std():.3f}")

        if class_counts:
            # Clarify label spaces: NYU40 ids in pseudo file, map to GCI + names for readability
            print(f"\nClass Distribution:")
            m = getattr(self, 'mappings', {}) or {}
            nyu2gci = m.get('nyu40_to_model_idx', {})
            gci2name = m.get('model_idx_to_name', {})
            detected = sorted(class_counts.keys())
            label_formatter = None
            if nyu2gci and gci2name:
                def _format_label(ny_val: int) -> str:
                    gci_val = nyu2gci.get(int(ny_val))
                    if gci_val is None:
                        return f"NYU40 {int(ny_val)}"
                    name_val = gci2name.get(int(gci_val), f"class_{int(gci_val)}")
                    return f"NYU40 {int(ny_val)} -> GCI {int(gci_val)} ({name_val})"

                label_formatter = _format_label
                pretty = [_format_label(int(ny)) for ny in detected]
                print(f"  Classes detected (NYU40 -> GCI (name)): {pretty}")
            else:
                print(f"  Classes detected (NYU40 ids): {detected}")
            print(f"  Number of classes: {len(class_counts)}")
            print(f"\n  Per-class statistics:")
            for ny in sorted(class_counts.keys()):
                scene_count = len(scenes_per_class.get(ny, []))
                detection_count = class_counts[ny]
                if nyu2gci and gci2name and int(ny) in nyu2gci and label_formatter:
                    header = label_formatter(int(ny))
                else:
                    header = f"NYU40 {int(ny)}"
                print(f"    {header}: {detection_count:6,} detections in {scene_count:4} scenes "
                      f"(avg {detection_count/max(1,scene_count):.1f} per scene)")

        # Check alignment with current stage (NYU40 label space)
        if self.stage_classes and self.all_stage_definitions:
            # Build NYU40 sets for previous and current stage
            prev_stage_nyu40 = []
            for stage_def in self.all_stage_definitions[:self.stage_idx]:
                prev_stage_nyu40.extend(stage_def.get('nyu40_ids', []))
            prev_stage_nyu40 = sorted(set(int(x) for x in prev_stage_nyu40))

            curr_stage_nyu40 = []
            for stage_def in self.all_stage_definitions:
                if stage_def.get('stage_id') == (self.stage_idx + 1):
                    curr_stage_nyu40 = sorted(set(int(x) for x in stage_def.get('nyu40_ids', [])))
                    break

            print(f"\nStage Alignment (NYU40):")
            formatter = label_formatter or (lambda x: f"NYU40 {int(x)}")
            print(f"  Previous stage NYU40: {[formatter(int(c)) for c in prev_stage_nyu40]}")
            print(f"  Current stage NEW NYU40: {[formatter(int(c)) for c in curr_stage_nyu40]}")

            detected_prev = [c for c in prev_stage_nyu40 if c in class_counts]
            detected_new = [c for c in curr_stage_nyu40 if c in class_counts]
            missing_prev = [c for c in prev_stage_nyu40 if c not in class_counts]
            missing_current = [c for c in curr_stage_nyu40 if c not in class_counts]

            print(f"  Detected from previous: {[formatter(int(c)) for c in detected_prev]} "
                  f"({len(detected_prev)}/{len(prev_stage_nyu40)})")
            if missing_prev:
                print(f"  Missing previous-stage classes (0 detections): "
                      f"{[formatter(int(c)) for c in missing_prev]}")
            else:
                print("  All previous-stage classes present in pseudo labels")

            print(f"  Detected from current (should be 0): {[formatter(int(c)) for c in detected_new]}")
            if missing_current:
                print(f"  Current-stage classes without pseudo detections: "
                      f"{[formatter(int(c)) for c in missing_current]}")

            if detected_new:
                print(f"  WARNING: Pseudo labels contain NEW classes {detected_new} - this should not happen!")

        print(f"{'='*60}\n")

    def _inject_pseudo_labels_pre_pipeline(self):
        """Merge loaded pseudo labels into data_infos annos before pipeline.

        Pseudo labels MUST already be canonical (aligned, gravity-centered, 6D,
        NYU40). No corrective transforms are applied here.
        """
        import numpy as _np

        apply_to_memory = bool(getattr(self, 'apply_pseudo_to_memory_scenes', False))

        # Restrict to train scenes selected by policy:
        # - natural only (default)
        # - natural + replay/merged when apply_to_memory_scenes=True
        target_scene_ids = set()
        for _info in self.data_infos:
            if (not apply_to_memory) and is_memory_or_merged_scene(_info):
                continue
            _sid = self._get_scene_id_from_data_info(_info)
            if _sid:
                target_scene_ids.add(str(_sid))
        if isinstance(self.pseudo_labels, dict) and target_scene_ids:
            _before = len(self.pseudo_labels)
            self.pseudo_labels = {
                k: v for k, v in self.pseudo_labels.items()
                if str(k) in target_scene_ids
            }
            _after = len(self.pseudo_labels)
            policy_name = (
                'natural+replay/merged' if apply_to_memory else 'natural-only'
            )
            print(
                "  Pseudo labels: filtered to training scenes by policy "
                f"{policy_name} ({_after}/{_before})"
            )

        # Collect previous-stage NYU40 IDs to avoid leaking new classes
        prev_stage_nyu40 = []
        if hasattr(self, 'all_stage_definitions') and self.all_stage_definitions:
            curr_stage_id = self.stage_definition.get('stage_id', self.stage_idx + 1)
            for sd in self.all_stage_definitions:
                if sd.get('stage_id', 0) < curr_stage_id:
                    prev_stage_nyu40.extend([int(x) for x in sd.get('nyu40_ids', [])])
        prev_stage_nyu40 = sorted(set(prev_stage_nyu40))

        injected_scenes = 0
        total_added = 0
        per_scene_net_added = []
        # Aggregated pseudo dedup counters for debug
        agg_pseudo_orig = 0
        agg_pseudo_after_gt = 0
        agg_pseudo_after_nms = 0
        # Aggregated per-class pseudo counts after dedup (model class indices)
        pseudo_class_counts_model = {}

        for i, info in enumerate(self.data_infos):
            # Default policy: augment natural scenes only.
            if (not apply_to_memory) and is_memory_or_merged_scene(info):
                continue

            # Extract scene id
            scene_id = None
            if 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
                scene_id = str(info['point_cloud']['lidar_idx'])
            elif 'sample_idx' in info:
                scene_id = str(info['sample_idx'])
            elif 'pts_path' in info:
                try:
                    import os as _os
                    scene_id = _os.path.basename(info['pts_path']).split('.')[0]
                except Exception:
                    scene_id = None

            if not scene_id or scene_id not in self.pseudo_labels:
                continue

            pseudo_data = self.pseudo_labels[scene_id]
            pseudo_boxes = pseudo_data.get('boxes', None)
            pseudo_labels_nyu40 = pseudo_data.get('labels', None)
            if pseudo_boxes is None or pseudo_labels_nyu40 is None:
                continue

            pseudo_boxes = _np.asarray(pseudo_boxes, dtype=_np.float32)
            pseudo_labels_nyu40 = _np.asarray(pseudo_labels_nyu40, dtype=_np.int64)
            if pseudo_boxes.size == 0 or pseudo_labels_nyu40.size == 0:
                continue

            # Enforce canonical 6D shape; fail early if wrong
            if pseudo_boxes.ndim != 2 or pseudo_boxes.shape[1] != 6:
                raise ValueError("Pseudo boxes must be (N,6) in canonical upright_depth_6d format")

            # Filter to previous-stage NYU40 IDs if available
            if prev_stage_nyu40:
                keep_mask = _np.isin(pseudo_labels_nyu40, prev_stage_nyu40)
                if not keep_mask.any():
                    continue
                pseudo_boxes = pseudo_boxes[keep_mask]
                pseudo_labels_nyu40 = pseudo_labels_nyu40[keep_mask]

            annos = info.get('annos', {})
            existing_boxes = _np.asarray(annos.get('gt_boxes_upright_depth', _np.zeros((0, 6), dtype=_np.float32)), dtype=_np.float32)
            existing_labels = _np.asarray(annos.get('class', _np.zeros((0,), dtype=_np.int64)), dtype=_np.int64)

            # Suppress pseudo that overlap any GT by IoU (class-agnostic)
            orig_pseudo = int(pseudo_boxes.shape[0])
            try:
                keep_mask = filter_pseudo_by_iou_against_gt(existing_boxes, pseudo_boxes, iou_thr=self.pseudo_vs_gt_iou_thr)
            except Exception:
                keep_mask = _np.ones((pseudo_boxes.shape[0],), dtype=bool)
            pseudo_boxes_kept = pseudo_boxes[keep_mask]
            pseudo_labels_kept = pseudo_labels_nyu40[keep_mask]
            after_gt = int(pseudo_boxes_kept.shape[0])

            # Per-class NMS on pseudo (use scores if provided)
            pseudo_scores_full = pseudo_data.get('scores', None)
            if pseudo_scores_full is not None:
                pseudo_scores_full = _np.asarray(pseudo_scores_full, dtype=_np.float32)
                if pseudo_scores_full.shape[0] == pseudo_labels_nyu40.shape[0]:
                    pseudo_scores = pseudo_scores_full[keep_mask]
                else:
                    pseudo_scores = _np.ones((pseudo_labels_kept.shape[0],), dtype=_np.float32)
            else:
                pseudo_scores = _np.ones((pseudo_labels_kept.shape[0],), dtype=_np.float32)

            if pseudo_boxes_kept.size:
                kept_idx_global: list[int] = []
                try:
                    from .pseudo_label_utils import nms_indices_iou as _nms_iou
                except Exception:
                    _nms_iou = None
                for cls in _np.unique(pseudo_labels_kept):
                    cls_idx = _np.where(pseudo_labels_kept == cls)[0]
                    if cls_idx.size == 0:
                        continue
                    if _nms_iou is not None:
                        keep_local = _nms_iou(pseudo_boxes_kept[cls_idx], pseudo_scores[cls_idx], iou_thr=self.pseudo_nms_iou_thr)
                        kept_idx_global.extend(cls_idx[_np.asarray(keep_local, dtype=_np.int64)].tolist())
                    else:
                        kept_idx_global.extend(cls_idx.tolist())
                if kept_idx_global:
                    kept_idx_global = sorted(set(kept_idx_global))
                    pseudo_boxes_kept = pseudo_boxes_kept[kept_idx_global]
                    pseudo_labels_kept = pseudo_labels_kept[kept_idx_global]
            after_nms = int(pseudo_boxes_kept.shape[0])

            # Per-scene debug line for pseudo dedup
            if self.debug_mode:
                removed_gt = orig_pseudo - after_gt
                removed_nms = max(0, after_gt - after_nms)
                print(
                    f"    🔍 Pseudo dedup [{scene_id}]: {orig_pseudo}->{after_gt}->{after_nms} "
                    f"(rm GT {removed_gt}, NMS {removed_nms}; thr gt={self.pseudo_vs_gt_iou_thr:.2f}, nms={self.pseudo_nms_iou_thr:.2f})"
                )

            # Aggregate counters
            agg_pseudo_orig += orig_pseudo
            agg_pseudo_after_gt += after_gt
            agg_pseudo_after_nms += after_nms

            # Merge GT + filtered pseudo
            if existing_boxes.size:
                merged_boxes = _np.concatenate([existing_boxes, pseudo_boxes_kept], axis=0)
                merged_labels = _np.concatenate([existing_labels, pseudo_labels_kept], axis=0)
            else:
                merged_boxes, merged_labels = pseudo_boxes_kept, pseudo_labels_kept

            self.data_infos[i]['annos']['gt_boxes_upright_depth'] = merged_boxes
            self.data_infos[i]['annos']['class'] = merged_labels
            self.data_infos[i]['annos']['gt_num'] = int(len(merged_boxes))

            injected_scenes += 1
            # Net boxes attributable to pseudo after de-duplication
            net_added = max(0, int(len(merged_boxes)) - int(len(existing_boxes)))
            total_added += net_added
            per_scene_net_added.append(net_added)

            # Aggregate per-class pseudo counts (model-indexed) for analysis
            try:
                nyu2model = self.mappings.get('nyu40_to_model_idx', {}) if isinstance(self.mappings, dict) else {}
                if nyu2model and pseudo_labels_kept.size:
                    model_labels_pseudo = [_np.int64(nyu2model.get(int(n), -999)) for n in pseudo_labels_kept]
                    for ml in model_labels_pseudo:
                        if int(ml) == -999:
                            continue
                        pseudo_class_counts_model[int(ml)] = pseudo_class_counts_model.get(int(ml), 0) + 1
            except Exception:
                pass

        print(f"  Injected pseudo labels into {injected_scenes} scenes (net added {total_added} boxes post-dedup)")

        # Always-on detailed counters
        if injected_scenes > 0:
            try:
                import numpy as _np
                arr = _np.array(per_scene_net_added, dtype=_np.int32)
                avg = float(arr.mean()) if arr.size else 0.0
                med = float(_np.median(arr)) if arr.size else 0.0
                p90 = float(_np.percentile(arr, 90)) if arr.size else 0.0
                print("  📈 PSEUDO COUNTERS (post-threshold, post-alignment, post-dedup):")
                print(f"    Scenes with pseudo: {injected_scenes}")
                print(f"    Total pseudo boxes used: {total_added}")
                print(f"    Avg per scene: {avg:.1f} | Median: {med:.1f} | P90: {p90:.1f}")
                # Optional detailed summary for pseudo dedup steps
                if self.debug_mode and agg_pseudo_orig > 0:
                    removed_gt_agg = agg_pseudo_orig - agg_pseudo_after_gt
                    removed_nms_agg = max(0, agg_pseudo_after_gt - agg_pseudo_after_nms)
                    print("  🧮 PSEUDO DEDUP SUMMARY:")
                    print(f"    Original: {agg_pseudo_orig} | after GT: {agg_pseudo_after_gt} | after NMS: {agg_pseudo_after_nms}")
                    print(f"    Removed due to GT IoU: {removed_gt_agg} | Removed by NMS: {removed_nms_agg}")
            except Exception:
                # Fallback without numpy
                count = len(per_scene_net_added)
                total = sum(per_scene_net_added)
                avg = total / count if count else 0.0
                print("  📈 PSEUDO COUNTERS (post-threshold, post-alignment, post-dedup):")
                print(f"    Scenes with pseudo: {injected_scenes}")
                print(f"    Total pseudo boxes used: {total}")
                print(f"    Avg per scene: {avg:.1f}")

        # Store a compact summary for later retrieval by training script or analyses
        try:
            self.pseudo_injection_summary = {
                'injected_scenes': int(injected_scenes),
                'total_pseudo_boxes_used': int(total_added),
                'dedup': {
                    'original': int(agg_pseudo_orig),
                    'after_gt': int(agg_pseudo_after_gt),
                    'after_nms': int(agg_pseudo_after_nms)
                },
                'per_class_model_counts': {str(int(k)): int(v) for k, v in pseudo_class_counts_model.items()}
            }
        except Exception:
            self.pseudo_injection_summary = None

    def _filter_data_for_stage(self):
        """Filter dataset based on mode: current stage classes (training) or all seen classes (evaluation)."""
        # Determine which classes to include based on mode
        if self.evaluation_mode:
            target_classes = self.all_seen_classes
            mode_desc = f"ALL SEEN CLASSES (evaluation mode)"
        else:
            target_classes = self.stage_classes
            mode_desc = f"CURRENT STAGE CLASSES (training mode)"

        if not target_classes:
            # No filtering if no target classes specified
            return

        filtered_data_infos = []
        class_sample_counts = {cls: 0 for cls in target_classes}

        # Use explicit mappings if available, fallback to parent mappings
        nyu40_to_model_mapping = self.mappings.get('nyu40_to_model_idx', {})
        if not nyu40_to_model_mapping:
            nyu40_to_model_mapping = getattr(self, 'nyu40_to_model_id', {})

        valid_nyu40_ids = self.mappings.get('valid_nyu40_ids', [])
        if not valid_nyu40_ids:
            valid_nyu40_ids = getattr(self, 'VALID_CLASS_IDS', [])
        mapped_nyu40_ids = np.array(
            sorted(int(x) for x in nyu40_to_model_mapping.keys()),
            dtype=np.int64)

        for data_info in self.original_data_infos:
            if data_info['annos']['gt_num'] == 0:
                continue

            # Get NYU40 class labels
            gt_labels_nyu40 = data_info['annos']['class'].astype(np.int64)

            # Convert to model class indices using explicit mappings.
            # Keep only labels that are both valid ScanNet-35 ids and mappable.
            valid_mask = np.isin(gt_labels_nyu40, valid_nyu40_ids)
            if mapped_nyu40_ids.size > 0:
                valid_mask = np.logical_and(
                    valid_mask, np.isin(gt_labels_nyu40, mapped_nyu40_ids))
            if not valid_mask.any():
                continue

            valid_nyu40_labels = gt_labels_nyu40[valid_mask]
            model_labels = np.array([
                nyu40_to_model_mapping[int(nyu40_id)]
                for nyu40_id in valid_nyu40_labels
            ], dtype=np.int64)

            if len(model_labels) == 0:
                continue

            # Check if scene contains any objects from target classes
            target_objects_mask = np.isin(model_labels, target_classes)
            if not target_objects_mask.any():
                continue

            if self.evaluation_mode:
                # EVALUATION MODE: Keep all mapped ScanNet-35 objects in scenes
                # that contain any seen-class object.
                filtered_data_info = copy.deepcopy(data_info)
                filtered_data_info['annos']['gt_boxes_upright_depth'] = \
                    data_info['annos']['gt_boxes_upright_depth'][valid_mask]
                filtered_data_info['annos']['class'] = \
                    data_info['annos']['class'][valid_mask]
                filtered_data_info['annos']['gt_num'] = int(valid_mask.sum())
                if filtered_data_info['annos']['gt_num'] == 0:
                    continue

                # Count objects for all classes (not just target classes)
                for model_idx in model_labels:
                    if model_idx in class_sample_counts:
                        class_sample_counts[model_idx] += 1
            else:
                # TRAINING MODE: Filter annotations to only include current stage objects
                filtered_data_info = copy.deepcopy(data_info)

                # Keep only target class objects in annotations for training
                all_valid_mask = valid_mask
                target_mask_full = np.zeros_like(
                    gt_labels_nyu40, dtype=bool)

                valid_indices = np.where(all_valid_mask)[0]
                target_indices_in_valid = np.where(target_objects_mask)[0]
                if len(target_indices_in_valid) > 0 and len(valid_indices) >= len(target_indices_in_valid):
                    target_indices_full = valid_indices[target_indices_in_valid]
                    target_mask_full[target_indices_full] = True

                if not target_mask_full.any():
                    continue

                # Update annotations to only include target class objects
                filtered_data_info['annos']['gt_boxes_upright_depth'] = \
                    data_info['annos']['gt_boxes_upright_depth'][target_mask_full]

                # Use the already converted labels from parent dataset
                filtered_data_info['annos']['class'] = \
                    data_info['annos']['class'][target_mask_full]

                filtered_data_info['annos']['gt_num'] = int(
                    target_mask_full.sum())

                # Update class counts for target classes only
                target_model_labels = model_labels[target_objects_mask]
                for cls in target_model_labels:
                    if cls in class_sample_counts:
                        class_sample_counts[cls] += 1

            filtered_data_infos.append(filtered_data_info)

        # Log filtering statistics
        print(f"Stage {self.stage_idx + 1} ({self.stage_name}) filtering results:")
        print(f"  Mode: {mode_desc}")
        print(f"  Original samples: {len(self.original_data_infos)}")
        print(f"  Filtered samples: {len(filtered_data_infos)}")
        print(f"  Target classes: {target_classes}")
        print(f"  Class sample counts: {dict(sorted(class_sample_counts.items()))}")

        # LOGGING: Record scene IDs used in this stage for visualization
        if self.work_dir is not None and not self.evaluation_mode:
            # Extract scene IDs from filtered data
            used_scene_ids = []
            for data_info in filtered_data_infos:
                scene_id = self._get_scene_id_from_data_info(data_info)
                if scene_id:
                    used_scene_ids.append(scene_id)

            # Save scene IDs to JSON file for visualization notebook
            import json
            scene_log_path = os.path.join(
                self.work_dir,
                f"training_scenes_stage_{self.stage_idx + 1}.json"
            )
            # No need to create directory as self.work_dir should already exist

            scene_log_data = {
                'stage_id': self.stage_idx + 1,
                'stage_name': self.stage_name,
                'mode': mode_desc,
                'total_scenes': len(used_scene_ids),
                'scene_ids': used_scene_ids,
                'target_classes': target_classes,
                'class_sample_counts': dict(sorted(class_sample_counts.items())),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            try:
                with open(scene_log_path, 'w') as f:
                    json.dump(scene_log_data, f, indent=2)
                print(f"  Training scenes logged to: {os.path.relpath(scene_log_path)}")
                print(f"  📋 Scene IDs sample: {used_scene_ids[:10]}..." if len(used_scene_ids) > 10 else f"  📋 All scene IDs: {used_scene_ids}")
            except Exception as e:
                print(f"  WARNING: Failed to log training scenes: {e}")

        # Check for classes with too few samples (only warn in training mode)
        if not self.evaluation_mode:
            low_sample_classes = [cls for cls, count in class_sample_counts.items()
                                 if count < self.min_samples_per_class]
            if low_sample_classes:
                print(f"  WARNING: Classes with <{self.min_samples_per_class} samples: {low_sample_classes}")

        self.data_infos = filtered_data_infos

        # Update flag_size to match filtered data length
        if hasattr(self, 'flag'):
            # Reset flag to match new dataset size
            self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

    def _add_memory_exemplars(self):
        """Configure memory bank for in-scene exemplar insertion during training.

        NOTE: This method no longer creates separate exemplar samples. Instead, it
        prepares the memory bank for in-scene insertion via the InsertExemplarObjects
        transform in the data pipeline.
        """
        if not hasattr(self.object_memory_bank, 'get_exemplars'):
            print("Memory bank does not support get_exemplars method")
            return

        # Get all previous stage classes
        previous_classes = []
        current_stage_id = self.stage_definition.get('stage_id', 1)
        for stage_def in self.all_stage_definitions:
            if stage_def['stage_id'] < current_stage_id:
                previous_classes.extend(stage_def['class_indices'])

        if not previous_classes:
            print("No previous stages, skipping memory exemplar configuration")
            return

        # Get exemplars from memory bank for logging/verification
        exemplars = self.object_memory_bank.get_exemplars(previous_classes)

        if not exemplars:
            print(f"No exemplars available for previous classes: {previous_classes}")
            return

        print(f"INCREMENTAL LEARNING: Memory bank configured for in-scene insertion")
        print(f"   Available exemplars: {len(exemplars)} objects from classes {set(e['class_id'] for e in exemplars)}")

        # Log exemplar statistics by class
        from collections import Counter
        class_counts = Counter(e['class_id'] for e in exemplars)
        for class_id, count in sorted(class_counts.items()):
            class_name = self.mappings.get('model_idx_to_name', {}).get(class_id, f"class_{class_id}")
            print(f"   📦 Class {class_id} ({class_name}): {count} exemplars available")

        # Configure memory bank for insertion (store previous classes for sampling)
        self.object_memory_bank.previous_classes = previous_classes
        print(f"Memory bank ready for in-scene exemplar insertion")
        print(f"   Dataset size remains: {len(self.data_infos)} scenes (exemplars will be inserted INTO scenes)")

        # NOTE: No modification to self.data_infos - the dataset size stays the same
        # Exemplars will be inserted into existing scenes via the pipeline transform

    def update_memory_bank_from_stage(
            self, learning_dynamics_design2_payload=None):
        """Update memory bank with exemplars from current stage after training."""
        if learning_dynamics_design2_payload is not None:
            raise NotImplementedError(
                'Object Design-2 population is currently implemented for '
                'SUN RGB-D only.')
        if self.object_memory_bank is None:
            print("No memory bank available for update")
            return

        if self.evaluation_mode:
            print("Skipping memory bank update in evaluation mode")
            return

        print(f"Updating memory bank from stage {self.stage_idx + 1} with classes: {self.stage_classes}")

        # Collect objects from all scenes for current stage classes
        class_objects = {cls: [] for cls in self.stage_classes}
        scene_points_dict = {}  # Cache for scene points

        # Use explicit mappings
        nyu40_to_model_mapping = self.mappings.get('nyu40_to_model_idx', {})
        if not nyu40_to_model_mapping:
            nyu40_to_model_mapping = getattr(self, 'nyu40_to_model_id', {})

        valid_nyu40_ids = self.mappings.get('valid_nyu40_ids', [])
        if not valid_nyu40_ids:
            valid_nyu40_ids = getattr(self, 'VALID_CLASS_IDS', [])

        for data_info in self.original_data_infos:
            if data_info['annos']['gt_num'] == 0:
                continue

            # All data is now regular scene data - no exemplar skipping needed

            scene_id = self._get_scene_id_from_data_info(data_info)
            if scene_id is None:
                print(f"WARNING: Could not extract scene_id from data_info")
                continue

            # Get NYU40 class labels and boxes
            gt_labels_nyu40 = data_info['annos']['class'].astype(np.int64)
            gt_boxes = data_info['annos']['gt_boxes_upright_depth']

            # Convert to model class indices
            valid_mask = np.isin(gt_labels_nyu40, valid_nyu40_ids)
            if not valid_mask.any():
                continue

            valid_nyu40_labels = gt_labels_nyu40[valid_mask]
            valid_boxes = gt_boxes[valid_mask]

            model_labels = np.array([
                nyu40_to_model_mapping[nyu40_id] for nyu40_id in valid_nyu40_labels
                if nyu40_id in nyu40_to_model_mapping
            ])

            if len(model_labels) == 0:
                continue

            # Group objects by class for current stage
            for i, (model_label, box, nyu40_id) in enumerate(zip(model_labels, valid_boxes, valid_nyu40_labels)):
                if model_label in class_objects:
                    obj_info = {
                        'scene_id': scene_id,
                        'object_idx': i,
                        'bbox': box,
                        'class_id': model_label,
                        'nyu40_id': nyu40_id,
                        'confidence': 1.0  # Default confidence, could use model predictions
                    }
                    class_objects[model_label].append(obj_info)

                    # Cache scene points for extraction using original dataset
                    if scene_id not in scene_points_dict:
                        scene_points = self._load_scene_points(scene_id)
                        scene_points_dict[scene_id] = scene_points

        # Add exemplars to memory bank for each class
        debug_dir = None
        if self.paths is not None:
            debug_dir = str(self.paths.debug_dir() / 'exemplars_debug')
        elif self.work_dir is not None:
            debug_dir = os.path.join(self.work_dir, 'exemplars_debug')
        else:
            debug_dir = None

        # Track edge cases during memory bank update
        update_summary = {
            'stage_id': self.stage_idx + 1,
            'total_classes': len(self.stage_classes),
            'classes_with_objects': 0,
            'empty_classes': [],
            'insufficient_classes': [],
            'successful_classes': []
        }

        for class_id in self.stage_classes:
            objects = class_objects.get(class_id, [])
            class_name = self.mappings.get('model_idx_to_name', {}).get(class_id, f"class_{class_id}")

            if not objects:
                # EDGE CASE: Empty class
                print(f"WARNING: No objects found for class {class_id} ({class_name})")
                update_summary['empty_classes'].append(class_id)
                if self.memory_handler:
                    self.memory_handler.handle_insufficient_exemplars(class_id, 0, self.object_memory_bank.exemplars_per_class)
                continue

            update_summary['classes_with_objects'] += 1

            # Check for insufficient exemplars
            if len(objects) < self.object_memory_bank.exemplars_per_class:
                print(f"WARNING: Insufficient objects for class {class_id} ({class_name})")
                print(f"    Found: {len(objects)}, Requested: {self.object_memory_bank.exemplars_per_class}")
                update_summary['insufficient_classes'].append(class_id)
                if self.memory_handler:
                    self.memory_handler.handle_insufficient_exemplars(
                        class_id, len(objects), self.object_memory_bank.exemplars_per_class
                    )
            else:
                update_summary['successful_classes'].append(class_id)

            print(f"Found {len(objects)} objects for class {class_id} ({class_name})")
            self.object_memory_bank.add_exemplars(
                class_id, objects,
                scene_points_dict=scene_points_dict,
                debug_save_dir=debug_dir,
                stage_id=self.stage_idx + 1  # Convert back to 1-based for debug files
            )
            print(f"Added exemplars for class {class_id} ({class_name})")

        # Check memory limits and handle overflow
        current_count = self.object_memory_bank.get_total_exemplar_count()
        max_count = self.object_memory_bank.max_total_exemplars

        if current_count > max_count:
            print(f"🚨 EDGE CASE: Memory bank overflow detected!")
            print(f"   Current: {current_count}, Maximum: {max_count}")

            if self.memory_handler:
                # Use handler for sophisticated reduction
                reduction_report = self.memory_handler.handle_memory_overflow(current_count, max_count)
                print(f"   Applied {reduction_report['strategy_name']} reduction strategy")
                print(f"   Reduced by {reduction_report['actual_reduction']} exemplars")

            # Apply standard reduction as fallback
            self.object_memory_bank._reduce_exemplars()
        elif current_count > max_count * 0.9:
            print(f"WARNING: Memory bank approaching limit ({current_count}/{max_count})")

        # Print detailed memory bank statistics
        print(f"Memory Bank Population Complete for Stage {self.stage_idx + 1}")
        print("="*60)

        # Print update summary
        print(f"📋 Update Summary:")
        print(f"  Total stage classes: {update_summary['total_classes']}")
        print(f"  Classes with objects: {update_summary['classes_with_objects']}")
        print(f"  Successful classes: {len(update_summary['successful_classes'])}")
        print(f"  Empty classes: {len(update_summary['empty_classes'])} {update_summary['empty_classes'] if update_summary['empty_classes'] else ''}")
        print(f"  Insufficient exemplars: {len(update_summary['insufficient_classes'])} {update_summary['insufficient_classes'] if update_summary['insufficient_classes'] else ''}")

        # Print memory bank statistics
        self.object_memory_bank.print_statistics()

        # Log per-class exemplar counts with edge case indicators
        print(f"\n📦 Per-Class Exemplar Counts:")
        total_exemplars = 0
        for class_id in self.stage_classes:
            class_name = self.mappings.get('model_idx_to_name', {}).get(class_id, f"class_{class_id}")
            if class_id in self.object_memory_bank.exemplars:
                count = len(self.object_memory_bank.exemplars[class_id])
                total_exemplars += count

                # Add edge case indicators
                status = "OK"
                if class_id in update_summary['empty_classes']:
                    status = "🔴 EMPTY"
                elif class_id in update_summary['insufficient_classes']:
                    status = "INSUFFICIENT"

                print(f"  Class {class_id:2d} ({class_name:15s}): {count:3d} exemplars {status}")
            else:
                print(f"  Class {class_id:2d} ({class_name:15s}):   0 exemplars 🔴 MISSING")

        print(f"  Total exemplars stored: {total_exemplars}")

        # Generate and save edge case report if handler available
        if self.memory_handler:
            print(f"Generating Edge Case Report...")
            report = self.memory_handler.generate_report(stage_id=self.stage_idx + 1)
            report['update_summary'] = update_summary

            # Save report
            report_filename = f"stage_{self.stage_idx + 1}_memory_report.json"
            self.memory_handler.save_report(report, report_filename)

            # Print handler summary
            self.memory_handler.print_summary()

        # Save active exemplar manifest for tracking
        if hasattr(self.object_memory_bank, 'save_active_manifest'):
            if self.paths:
                manifest_path = str(self.paths.object_memory_manifest(self.stage_idx + 1))
            elif self.work_dir:
                manifest_path = os.path.join(self.work_dir, f'memory_bank_stage_{self.stage_idx + 1}.json')
            else:
                manifest_path = None

            if manifest_path:
                self.object_memory_bank.save_active_manifest(manifest_path, stage_id=self.stage_idx + 1)
                print(f"Memory bank manifest saved: {manifest_path}")

        print("="*60)

    def get_stage_info(self):
        """Get information about current stage."""
        return {
            'stage_idx': self.stage_idx,
            'stage_classes': self.stage_classes,
            'total_samples': len(self.data_infos),
            'memory_exemplars': 0 if self.object_memory_bank is None else len(self.object_memory_bank.get_all_exemplars())
        }

    def __len__(self):
        """Return the length of filtered dataset."""
        return len(self.data_infos)

    def get_class_distribution(self):
        """Get distribution of classes in current dataset."""
        class_counts = {}

        for data_info in self.data_infos:
            if data_info['annos']['gt_num'] == 0:
                continue

            gt_labels_nyu40 = data_info['annos']['class'].astype(np.int64)
            valid_mask = np.isin(gt_labels_nyu40, list(self.VALID_CLASS_IDS))

            if valid_mask.any():
                valid_nyu40_labels = gt_labels_nyu40[valid_mask]
                model_labels = np.array([
                    self.nyu40_to_model_id[nyu40_id] for nyu40_id in valid_nyu40_labels
                ])

                for cls in model_labels:
                    class_counts[cls] = class_counts.get(cls, 0) + 1

        return class_counts

    def _get_scene_id_from_data_info(self, data_info):
        """Extract scene ID from data_info, handling various ScanNet formats.

        ScanNet datasets can store scene IDs in different locations:
        - Nested: data_info['point_cloud']['lidar_idx']
        - Flat: data_info['sample_idx']
        - Fallback: extract from pts_path
        """
        # Try nested format first (standard ScanNet)
        if 'point_cloud' in data_info and 'lidar_idx' in data_info['point_cloud']:
            scene_id = data_info['point_cloud']['lidar_idx']
        # Try flat format (some datasets)
        elif 'sample_idx' in data_info:
            scene_id = data_info['sample_idx']
        # Fallback: extract from pts_path
        elif 'pts_path' in data_info:
            scene_id = data_info['pts_path'].split('/')[-1].replace('.bin', '')
        else:
            return None

        # Ensure scene_id is string and clean format
        scene_id = str(scene_id)
        if '/' in scene_id:
            scene_id = scene_id.split('/')[-1]
        return scene_id

    def _load_scene_points(self, scene_id):
        """Load scene points by finding the corresponding data_info."""
        import os.path as osp
        from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile

        # Find the data_info for this scene_id using robust scene ID extraction
        target_data_info = None
        for data_info in self.original_data_infos:
            data_scene_id = self._get_scene_id_from_data_info(data_info)
            if data_scene_id and data_scene_id == scene_id:
                target_data_info = data_info
                break

        if target_data_info is None:
            print(f"WARNING: Scene {scene_id} not found in dataset")
            return None

        # Construct pts_filename using same logic as parent dataset
        pts_filename = osp.join(self.data_root, target_data_info['pts_path'])

        if not osp.exists(pts_filename):
            print(f"WARNING: Point cloud file not found: {pts_filename}")
            return None

        # Use the dataset's point loading pipeline
        point_loader = LoadPointsFromFile(
            coord_type='DEPTH',
            load_dim=6,
            use_dim=[0, 1, 2, 3, 4, 5],  # Use all dimensions to ensure proper reshaping
            shift_height=False,
            use_color=False,
            file_client_args=dict(backend='disk')
        )

        # Load points using the loader - this returns 1D array from .bin file
        points_1d = point_loader._load_points(pts_filename)

        # CRITICAL FIX: Ensure proper 2D array format
        # The _load_points method returns 1D array from .bin files
        if points_1d.ndim == 1:
            # Reshape to 2D array with 6 columns (x, y, z, r, g, b)
            if points_1d.size % 6 != 0:
                print(f"WARNING: Invalid point cloud format: {points_1d.size} points not divisible by 6")
                return None
            points_2d = points_1d.reshape(-1, 6)
        else:
            points_2d = points_1d

        # CRITICAL FIX: Apply axis alignment transformation to match bbox coordinate system
        # Raw .bin files contain unaligned points, but gt_boxes are in aligned coordinates
        if target_data_info and 'annos' in target_data_info and 'axis_align_matrix' in target_data_info['annos']:
            axis_align_matrix = target_data_info['annos']['axis_align_matrix']
            if axis_align_matrix is not None:
                print(f"Applying axis alignment to scene {scene_id} points for coordinate consistency")

                # Apply transformation to xyz coordinates (keep RGB unchanged)
                xyz = points_2d[:, :3].astype(np.float32)
                rgb = points_2d[:, 3:6]

                # Extract rotation and translation from 4x4 matrix
                rot_mat = axis_align_matrix[:3, :3]
                trans_vec = axis_align_matrix[:3, 3]

                # Apply transformation: xyz_aligned = xyz @ rot_mat.T + trans_vec
                xyz_aligned = xyz @ rot_mat.T + trans_vec

                # Combine aligned xyz with original rgb
                points_2d = np.concatenate([xyz_aligned, rgb], axis=1).astype(np.float32)
                print(f"Applied axis alignment to {len(points_2d)} points")
            else:
                print(f"WARNING: No axis alignment matrix found for scene {scene_id}")
        else:
            print(f"WARNING: No annotation data found for axis alignment in scene {scene_id}")

        return points_2d

    def _add_scene_replay(self):
        """Add replay scenes from scene memory bank.

        This method integrates scenes from previous stages with proper label filtering.
        Unlike object-based approach, this adds complete scenes to the dataset.
        """
        if self.scene_memory_bank is None:
            return

        # Get classes from previous stages
        previous_classes = []
        for stage_def in self.all_stage_definitions[:self.stage_idx]:
            previous_classes.extend(stage_def['class_indices'])
        previous_classes = sorted(set(previous_classes))

        if not previous_classes:
            print("  No previous classes to replay")
            return

        # Extract scene IDs properly from nested structure (needed for debug output and deduplication)
        current_scene_ids = set()
        for info in self.data_infos:
            # Try multiple possible locations for scene ID
            scene_id = None
            if 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
                scene_id = info['point_cloud']['lidar_idx']
            elif 'sample_idx' in info:
                scene_id = info['sample_idx']
            elif 'scene_id' in info:
                scene_id = info['scene_id']

            if scene_id:
                current_scene_ids.add(scene_id)

        # Scene-based memory replay setup
        print(f"  Previous classes to replay: {previous_classes}")
        print(f"  Current stage scenes before replay: {len(self.data_infos)}")

        # Get replay scenes from memory bank
        replay_scenes, replay_scene_ids = self.scene_memory_bank.get_replay_scenes(
            previous_classes, self.stage_idx + 1
        )

        if not replay_scenes:
            print("  No replay scenes available from memory bank")
            return

        # Optional: downsample replay scenes to hit target memory ratio
        if self.target_memory_ratio is not None:
            try:
                natural_count = len(self.data_infos)
                # desired memory = R/(1-R) * natural
                target_mem = int(round((self.target_memory_ratio / max(1e-6, (1.0 - self.target_memory_ratio))) * natural_count))
                if target_mem < len(replay_scenes):
                    import numpy as _np
                    rng = _np.random.RandomState(self.memory_sampling_seed + self.stage_idx + 1)

                    # If we have replay focusing config, allocate a portion from focus candidates
                    if self.replay_focus_config and target_mem > 0:
                        focus_share = float(self.replay_focus_config.get('focus_share', 0.0))
                        focus_classes = set(int(c) for c in self.replay_focus_config.get('class_ids', []) )
                        focus_scene_ids = set(self.replay_focus_config.get('scene_ids', []) )

                        def _is_focus_scene(scene):
                            try:
                                sid = scene.get('original_scene_id')
                                if sid in focus_scene_ids:
                                    return True
                                labels = scene.get('annos', {}).get('class', [])
                                if any(int(l) in focus_classes for l in labels):
                                    return True
                            except Exception:
                                return False
                            return False

                        focus_candidates = [s for s in replay_scenes if _is_focus_scene(s)]
                        other_candidates = [s for s in replay_scenes if not _is_focus_scene(s)]

                        k_focus = int(round(min(1.0, max(0.0, focus_share)) * target_mem))
                        k_focus = min(k_focus, len(focus_candidates))
                        k_other = target_mem - k_focus
                        k_other = min(k_other, len(other_candidates))
                        # If not enough other candidates, fill from focus and vice versa
                        if k_focus + k_other < target_mem:
                            remaining = target_mem - (k_focus + k_other)
                            spare_focus = min(remaining, len(focus_candidates) - k_focus)
                            spare_other = min(remaining - spare_focus, len(other_candidates) - k_other)
                            k_focus += spare_focus
                            k_other += spare_other

                        focus_idx = rng.choice(len(focus_candidates), size=k_focus, replace=False).tolist() if k_focus > 0 else []
                        other_idx = rng.choice(len(other_candidates), size=k_other, replace=False).tolist() if k_other > 0 else []
                        new_replay = [focus_candidates[i] for i in sorted(focus_idx)] + [other_candidates[i] for i in sorted(other_idx)]
                        replay_scenes = new_replay
                        print(f"  Memory sampling (focused): kept {len(replay_scenes)} of {len(replay_scene_ids)}; focus={k_focus}, other={k_other}, ratio={self.target_memory_ratio:.2f}")
                    else:
                        idx = rng.choice(len(replay_scenes), size=target_mem, replace=False)
                        replay_scenes = [replay_scenes[i] for i in sorted(idx.tolist())]
                        print(f"  Memory sampling: kept {len(replay_scenes)}/{len(replay_scene_ids)} to target ratio {self.target_memory_ratio:.2f}")
                else:
                    print(f"  Memory sampling: target {target_mem} >= available {len(replay_scenes)}; keeping all")
            except Exception as e:
                print(f"  ⚠️ Memory sampling failed; keeping all replay scenes. Error: {e}")

        # Handle deduplication based on strategy
        # Recommended strategies:
        #   - 'keep_both': Simple, no deduplication (good for understanding data flow)
        #   - 'merge_labels': Most accurate, combines annotations from both versions
        # Deprecated strategies (complex and potentially confusing):
        #   - 'prefer_replay', 'prefer_natural': Use with caution

        final_replay_scenes = []
        duplicates_found = []

        if self.scene_dedup_strategy == 'keep_both':
            # Keep both natural and replay versions
            final_replay_scenes = replay_scenes
            # Check for duplicates just for logging
            for scene in replay_scenes:
                if scene['original_scene_id'] in current_scene_ids:
                    duplicates_found.append(scene['original_scene_id'])

        elif self.scene_dedup_strategy == 'prefer_replay':
            # Remove natural scenes if they're in replay
            print(f"  WARNING: Using deprecated deduplication strategy '{self.scene_dedup_strategy}'. "
                  f"Consider 'keep_both' or 'merge_labels' for clearer behavior.")
            scenes_to_remove = []
            for scene in replay_scenes:
                if scene['original_scene_id'] in current_scene_ids:
                    duplicates_found.append(scene['original_scene_id'])
                    scenes_to_remove.append(scene['original_scene_id'])
                final_replay_scenes.append(scene)

            # Remove duplicates from current data_infos
            if scenes_to_remove:
                self.data_infos = [info for info in self.data_infos
                                 if info.get('sample_idx', '') not in scenes_to_remove]
                print(f"  🔄 Removed {len(scenes_to_remove)} natural scenes in favor of replay versions")

        elif self.scene_dedup_strategy == 'prefer_natural':
            # Only add replay scenes not in current stage
            print(f"  WARNING: Using deprecated deduplication strategy '{self.scene_dedup_strategy}'. "
                  f"Consider 'keep_both' or 'merge_labels' for clearer behavior.")
            for scene in replay_scenes:
                if scene['original_scene_id'] not in current_scene_ids:
                    final_replay_scenes.append(scene)
                else:
                    duplicates_found.append(scene['original_scene_id'])

        elif self.scene_dedup_strategy == 'merge_labels':
            # Merge labels from replay and natural versions of same scenes
            merged_scenes = {}
            scenes_to_remove = set()  # Use set to prevent duplicate indices

            for scene in replay_scenes:
                scene_id = scene['original_scene_id']

                if scene_id in current_scene_ids:
                    # Find the natural version of this scene
                    natural_scene = None
                    for i, natural_info in enumerate(self.data_infos):
                        # Extract scene ID properly from natural scene
                        natural_scene_id = None
                        if 'point_cloud' in natural_info and 'lidar_idx' in natural_info['point_cloud']:
                            natural_scene_id = natural_info['point_cloud']['lidar_idx']
                        elif 'sample_idx' in natural_info:
                            natural_scene_id = natural_info['sample_idx']
                        elif 'scene_id' in natural_info:
                            natural_scene_id = natural_info['scene_id']

                        if natural_scene_id == scene_id:
                            natural_scene = natural_info
                            scenes_to_remove.add(i)  # Add to set instead of append to list
                            break

                    if natural_scene is not None:
                        # Merge annotations from both versions
                        merged_scene = self._merge_scene_labels(natural_scene, scene, scene_id)
                        merged_scenes[scene_id] = merged_scene
                        duplicates_found.append(scene_id)
                    else:
                        # Replay scene without natural counterpart - add identity tracking
                        scene = self._add_scene_identity_tracking(scene, 'memory_only')
                        final_replay_scenes.append(scene)
                else:
                    # Replay scene not in current stage - add identity tracking
                    scene = self._add_scene_identity_tracking(scene, 'memory_only')
                    final_replay_scenes.append(scene)

            # Remove natural scenes that were merged
            if scenes_to_remove:
                # Remove in reverse order to maintain indices (convert set to sorted list)
                for i in sorted(scenes_to_remove, reverse=True):
                    del self.data_infos[i]

                # Add merged scenes to data_infos (they replace the natural versions)
                for merged_scene in merged_scenes.values():
                    self.data_infos.append(merged_scene)

                print(f"  🔄 Merged {len(merged_scenes)} duplicate scenes with complete labels")

        else:
            # Default to keep_both
            final_replay_scenes = replay_scenes

        # Store deduplication information for debugging
        if self.scene_memory_bank is not None:
            dedup_stats = {
                'strategy_used': self.scene_dedup_strategy,
                'total_replay_candidates': len(replay_scenes),
                'final_replay_scenes': len(final_replay_scenes),
                'duplicates_detected': len(duplicates_found),
                'duplicate_scene_ids': duplicates_found,
                'scenes_removed': getattr(self, '_scenes_removed_count', 0),
                'natural_scenes_before': len(current_scene_ids),
                'final_dataset_size': len(self.data_infos) + len(final_replay_scenes)
            }

            # Add merge_labels specific information
            if self.scene_dedup_strategy == 'merge_labels' and 'merged_scenes' in locals():
                dedup_stats.update({
                    'scenes_merged': len(merged_scenes),
                    'merged_scene_ids': list(merged_scenes.keys()),
                    'merge_details': {
                        scene_id: {
                            'natural_objects': scene.get('natural_object_count', 0),
                            'replay_objects': scene.get('replay_object_count', 0),
                            'final_objects': scene['annos']['gt_num'],
                            'stages_merged': scene.get('merged_from_stages', [])
                        } for scene_id, scene in merged_scenes.items()
                    }
                })

            self.scene_memory_bank.deduplication_stats[self.stage_idx + 1] = dedup_stats

        # Add replay scenes to dataset
        self.data_infos.extend(final_replay_scenes)

        # Log statistics
        print(f"  Scene replay summary: {len(final_replay_scenes)} scenes added, {len(duplicates_found)} duplicates handled")
        print(f"  Total dataset size: {len(self.data_infos)} scenes")

        # Report pseudo label coverage for replay scenes when replay pseudo is enabled.
        if (self.pseudo_labels and final_replay_scenes and
                bool(getattr(self, 'apply_pseudo_to_memory_scenes', False))):
            replay_with_pseudo = 0
            total_pseudo_detections = 0
            class_coverage = {}

            for scene in final_replay_scenes:
                # Extract scene ID from various possible locations
                scene_id = scene.get('original_scene_id')
                if not scene_id:
                    if 'point_cloud' in scene and 'lidar_idx' in scene['point_cloud']:
                        scene_id = scene['point_cloud']['lidar_idx']
                    elif 'sample_idx' in scene:
                        scene_id = scene['sample_idx']
                    elif 'scene_id' in scene:
                        scene_id = scene['scene_id']

                if scene_id and scene_id in self.pseudo_labels:
                    replay_with_pseudo += 1
                    pseudo_data = self.pseudo_labels[scene_id]
                    if 'labels' in pseudo_data:
                        scene_labels = pseudo_data['labels']
                        total_pseudo_detections += len(scene_labels)
                        # Track class coverage
                        for label in scene_labels:
                            cls_idx = int(label)
                            class_coverage[cls_idx] = class_coverage.get(cls_idx, 0) + 1

            print(f"\n  PSEUDO LABEL COVERAGE FOR REPLAY SCENES:")
            print(f"    Scenes with pseudo labels: {replay_with_pseudo}/{len(final_replay_scenes)} "
                  f"({100*replay_with_pseudo/len(final_replay_scenes) if final_replay_scenes else 0:.1f}%)")

            if replay_with_pseudo > 0:
                print(f"    Total pseudo detections: {total_pseudo_detections:,}")
                print(f"    Average per scene: {total_pseudo_detections/replay_with_pseudo:.1f}")
                print(f"    Classes in pseudo labels: {sorted(class_coverage.keys())}")
            else:
                print(f"    WARNING: No pseudo labels found for any replay scenes!")
        elif self.pseudo_labels and final_replay_scenes:
            print(
                "  Pseudo labels on replay scenes: disabled by "
                "pseudo_label_config.apply_to_memory_scenes=False"
            )

        # Aggregate replay class counts by model index for analysis
        try:
            nyu2model = self.mappings.get('nyu40_to_model_idx', {}) if isinstance(self.mappings, dict) else {}
            replay_class_counts_by_model = {}
            for scene in final_replay_scenes:
                if 'annos' in scene and 'class' in scene['annos']:
                    for ny in scene['annos']['class']:
                        if int(ny) in nyu2model:
                            mid = int(nyu2model[int(ny)])
                            replay_class_counts_by_model[mid] = replay_class_counts_by_model.get(mid, 0) + 1
            self.replay_class_counts_by_model = replay_class_counts_by_model
        except Exception:
            self.replay_class_counts_by_model = None

        # CRITICAL CHECK: Dataset size should not exceed total ScanNet scenes
        if len(self.data_infos) > 1201:
            print(f"  WARNING: Dataset size ({len(self.data_infos)}) exceeds total ScanNet scenes (1201)")
        else:
            print(f"  Dataset size within bounds: {len(self.data_infos)}/1201")

        # Log class distribution in replay scenes
        class_counts = {}
        for scene in final_replay_scenes:
            if 'annos' in scene and 'class' in scene['annos']:
                # Convert NYU40 to model indices
                nyu40_to_model = self.mappings.get('nyu40_to_model_idx', {})
                for nyu40_id in scene['annos']['class']:
                    if nyu40_id in nyu40_to_model:
                        model_id = nyu40_to_model[nyu40_id]
                        class_counts[model_id] = class_counts.get(model_id, 0) + 1

        # Class distribution computed for internal tracking

        # CRITICAL FIX: Update flag array to match new dataset length after adding replay scenes
        # The flag array is used by __getitem__ for sampling and must match data_infos length
        if not self.test_mode:
            self._set_group_flag()
            # Flag array updated for sampling consistency

    def _filter_mixed_dataset(self):
        """Apply correct filtering to mixed dataset (natural + replay scenes).

        Natural scenes: Only current stage objects
        Replay scenes: Keep all objects (they already have correct historical labels)
        """
        print(f"Applying mixed dataset filtering for Stage {self.stage_idx + 1}")

        # Get mappings
        nyu40_to_model_mapping = self.mappings.get('nyu40_to_model_idx', {})
        if not nyu40_to_model_mapping:
            nyu40_to_model_mapping = getattr(self, 'nyu40_to_model_id', {})

        valid_nyu40_ids = self.mappings.get('valid_nyu40_ids', [])
        if not valid_nyu40_ids:
            valid_nyu40_ids = getattr(self, 'VALID_CLASS_IDS', [])

        current_stage_classes = self.stage_classes
        filtered_data_infos = []
        natural_scenes_filtered = 0
        replay_scenes_kept = 0
        merged_scenes_kept = 0

        for data_info in self.data_infos:
            if data_info['annos']['gt_num'] == 0:
                continue

            is_replay = bool(data_info.get('is_replay', False))
            is_merged = bool(data_info.get('is_merged', False))

            if is_replay or is_merged:
                # Replay scenes: Keep all objects (already have correct labels from when saved)
                filtered_data_infos.append(data_info)
                if is_replay:
                    replay_scenes_kept += 1
                if is_merged:
                    merged_scenes_kept += 1
            else:
                # Natural scenes: Filter to only current stage objects
                gt_labels_nyu40 = data_info['annos']['class'].astype(np.int64)

                # Convert to model class indices
                valid_mask = np.isin(gt_labels_nyu40, valid_nyu40_ids)
                if not valid_mask.any():
                    continue

                valid_nyu40_labels = gt_labels_nyu40[valid_mask]
                model_labels = np.array([
                    nyu40_to_model_mapping[nyu40_id] for nyu40_id in valid_nyu40_labels
                    if nyu40_id in nyu40_to_model_mapping
                ])

                if len(model_labels) == 0:
                    continue

                # Check if scene contains any current stage objects
                target_objects_mask = np.isin(model_labels, current_stage_classes)
                if not target_objects_mask.any():
                    continue

                # Filter annotations to only include current stage objects
                filtered_data_info = copy.deepcopy(data_info)
                all_valid_mask = np.isin(gt_labels_nyu40, valid_nyu40_ids)
                target_mask_full = np.zeros_like(gt_labels_nyu40, dtype=bool)

                valid_indices = np.where(all_valid_mask)[0]
                target_indices_in_valid = np.where(target_objects_mask)[0]
                if len(target_indices_in_valid) > 0 and len(valid_indices) >= len(target_indices_in_valid):
                    target_indices_full = valid_indices[target_indices_in_valid]
                    target_mask_full[target_indices_full] = True

                if not target_mask_full.any():
                    continue

                # Update annotations to only include current stage objects
                filtered_data_info['annos']['gt_boxes_upright_depth'] = \
                    data_info['annos']['gt_boxes_upright_depth'][target_mask_full]
                filtered_data_info['annos']['class'] = \
                    data_info['annos']['class'][target_mask_full]
                filtered_data_info['annos']['gt_num'] = target_mask_full.sum()

                filtered_data_infos.append(filtered_data_info)
                natural_scenes_filtered += 1

        print(
            f"  Mixed dataset filtering results: {natural_scenes_filtered} natural scenes, "
            f"{replay_scenes_kept} replay scenes, {merged_scenes_kept} merged scenes, "
            f"{len(filtered_data_infos)} total"
        )

        # Update dataset
        self.data_infos = filtered_data_infos

        # Persist composition counters for later summary
        try:
            self._mixed_filtering_counters = {
                'natural_scenes_filtered': int(natural_scenes_filtered),
                'replay_scenes_kept': int(replay_scenes_kept),
                'merged_scenes_kept': int(merged_scenes_kept),
                'total_after_filter': int(len(self.data_infos))
            }
        except Exception:
            self._mixed_filtering_counters = None

    def _save_training_composition_summary(self):
        """Save a concise JSON summary of GT vs pseudo vs replay exposure.

        Written to memory_bank/scores/training_composition_stage_{k}.json when possible.
        Non-fatal if any pieces are missing.
        """
        import json as _json
        import os as _os

        # Only in training mode
        if self.evaluation_mode:
            return

        stage_id = int(self.stage_definition.get('stage_id', self.stage_idx + 1)) if self.stage_definition else int(self.stage_idx + 1)

        # Count GT objects (after filtering) in natural scenes only
        natural_gt_counts = {}
        natural_scene_count = 0
        replay_scene_count = 0
        total_dataset = len(self.data_infos)

        try:
            nyu2model = self.mappings.get('nyu40_to_model_idx', {}) if isinstance(self.mappings, dict) else {}
        except Exception:
            nyu2model = {}

        for di in self.data_infos:
            is_replay = bool(di.get('is_replay', False))
            if is_replay:
                replay_scene_count += 1
                continue
            natural_scene_count += 1
            try:
                if 'annos' in di and 'class' in di['annos']:
                    for ny in di['annos']['class']:
                        if int(ny) in nyu2model:
                            mid = int(nyu2model[int(ny)])
                            natural_gt_counts[mid] = natural_gt_counts.get(mid, 0) + 1
            except Exception:
                pass

        # Pseudo & replay summaries (if populated earlier)
        pseudo_summary = getattr(self, 'pseudo_injection_summary', None)
        replay_counts = getattr(self, 'replay_class_counts_by_model', None)
        mixed_cnt = getattr(self, '_mixed_filtering_counters', None)

        summary = {
            'stage_id': stage_id,
            'stage_name': self.stage_definition.get('stage_name', f'Stage {stage_id}') if self.stage_definition else f'Stage {stage_id}',
            'dataset': {
                'natural_scenes': int(natural_scene_count),
                'replay_scenes': int(replay_scene_count),
                'total_scenes': int(total_dataset)
            },
            'gt_natural_per_class_model': {str(int(k)): int(v) for k, v in natural_gt_counts.items()},
            'pseudo_summary': pseudo_summary,
            'replay_per_class_model': {str(int(k)): int(v) for k, v in replay_counts.items()} if isinstance(replay_counts, dict) else None,
            'mixed_filtering': mixed_cnt,
            'target_memory_ratio': float(self.target_memory_ratio) if self.target_memory_ratio is not None else None
        }

        # Resolve output path
        out_path = None
        try:
            if self.paths is not None:
                out_path = (
                    self.paths.memory_bank_scores_dir()
                    / f'training_composition_stage_{stage_id}.json'
                )
            elif self.work_dir:
                out_dir = _os.path.join(self.work_dir, 'memory_bank', 'scores')
                _os.makedirs(out_dir, exist_ok=True)
                out_path = _os.path.join(out_dir, f'training_composition_stage_{stage_id}.json')
        except Exception:
            out_path = None

        if out_path is None:
            return

        # Write file
        try:
            # pathlib.Path or str
            if hasattr(out_path, 'parent'):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'w') as f:
                    _json.dump(summary, f, indent=2)
            else:
                _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'w') as f:
                    _json.dump(summary, f, indent=2)
            print(f"  💾 Training composition summary saved: {out_path}")
        except Exception as _e:
            print(f"  ⚠️ Failed to write training composition summary: {_e}")

    def update_scene_memory_bank_from_stage(self,
                                            model=None,
                                            device: str = 'cuda',
                                            *,
                                            forgetness_class_drops: Dict[int, float] = None,
                                            underlearning_class_ap: Dict[int, float] = None,
                                            underlearning_new_classes: List[int] = None,
                                            learning_dynamics_forgetness_by_seat: Dict[str, Dict[int, float]] = None,
                                            learning_dynamics_replay_priority_by_seat: Dict[str, Dict[int, float]] = None,
                                            learning_dynamics_design1_payload: Dict[str, Any] = None,
                                            learning_dynamics_design2_payload: Dict[str, Any] = None,
                                            dataset_ref=None):
        """Update scene memory bank with scenes from current stage.

        This is called after training completes to save scene references.
        """
        if self.scene_memory_bank is None:
            return

        print(f"Updating scene memory bank from stage {self.stage_idx + 1}")

        # Get cumulative seen classes up to current stage
        seen_classes = []
        for stage_def in self.all_stage_definitions[:self.stage_idx + 1]:
            seen_classes.extend(stage_def['class_indices'])
        seen_classes = sorted(set(seen_classes))

        # Filter out replay scenes - only save natural scenes
        natural_scenes = [info for info in self.data_infos
                         if not info.get('is_replay', False)]

        print(f"  📋 Natural scenes to consider: {len(natural_scenes)}")
        print(f"  Seen classes at this stage: {seen_classes}")

        # Optional: compute per-scene uncertainty/diversity metrics when a
        # trained model is available and the memory bank is configured to use
        # advanced selection strategies.
        scene_metrics = None
        if (model is not None and
                hasattr(self, 'scene_memory_bank') and
                self.scene_memory_bank is not None and
                not self.evaluation_mode):
            strategy = getattr(self.scene_memory_bank, 'selection_strategy',
                               None)
            if strategy in (
                    'uncertainty_only',
                    'diversity_only',
                    'uncertainty_diversity_combined'):
                try:
                    scene_metrics = self._compute_uncertainty_diversity_metrics(
                        model=model,
                        device=device,
                        natural_scenes=natural_scenes,
                        seen_classes=seen_classes,
                    )
                except Exception as _e:
                    print(f"  ⚠️ Failed to compute uncertainty/diversity "
                          f"metrics for memory bank selection: {_e}")
                    scene_metrics = None

        # Add scenes to memory bank
        self.scene_memory_bank.add_stage_scenes(
            stage_id=self.stage_idx + 1,
            scene_infos=natural_scenes,
            seen_classes=seen_classes,
            mappings=self.mappings,
            dataset_ref=(self if dataset_ref is None else dataset_ref),
            scene_metrics=scene_metrics,
            forgetness_class_drops=forgetness_class_drops,
            underlearning_class_ap=underlearning_class_ap,
            underlearning_new_classes=underlearning_new_classes,
            learning_dynamics_forgetness_by_seat=learning_dynamics_forgetness_by_seat,
            learning_dynamics_replay_priority_by_seat=learning_dynamics_replay_priority_by_seat,
            learning_dynamics_design1_payload=learning_dynamics_design1_payload,
            learning_dynamics_design2_payload=learning_dynamics_design2_payload,
        )

        # Save memory bank state for debugging
        if self.work_dir:
            state_file = os.path.join(
                self.work_dir,
                f"scene_memory_bank_stage_{self.stage_idx + 1}.json"
            )
            self.scene_memory_bank.save_state(state_file)

        # Print summary
        self.scene_memory_bank.print_summary()

    def _compute_uncertainty_diversity_metrics(self,
                                               model,
                                               device: str,
                                               natural_scenes: list,
                                               seen_classes: list) -> dict:
        """Compute per-scene uncertainty and diversity metrics.

        This helper is used at the end of a stage to derive metrics for
        scene-based memory selection. It relies on the standard
        `inference_detector` API to ensure the same test pipeline as
        evaluation/pseudo-label generation.
        """
        from mmdet3d.apis import inference_detector
        from mmdet3d.core.bbox import DepthInstance3DBoxes
        import torch
        import numpy as _np
        import os as _os
        import json as _json

        stage_id = self.stage_definition.get('stage_id', 1) if self.stage_definition else 1
        print(f"  🔍 Computing uncertainty/diversity metrics for stage {stage_id}")

        mb = self.scene_memory_bank
        conf_thr = getattr(mb, 'uncertainty_conf_thresh', 0.15)
        div_conf_thr = getattr(mb, 'diversity_conf_thresh', conf_thr)
        iou_thr = getattr(mb, 'iou_match_thr', 0.25)
        undet_mode = getattr(mb, 'undet_classes', 'old_and_current')
        max_boxes_per_scene = getattr(mb, 'max_boxes_per_scene', None)

        # Determine which model-class indices to use for FN computation
        fn_class_ids = set()
        if self.all_stage_definitions:
            if undet_mode == 'old_only':
                target_stage_ids = [
                    sd['stage_id'] for sd in self.all_stage_definitions
                    if sd.get('stage_id', 0) < stage_id
                ]
            elif undet_mode == 'old_and_current':
                target_stage_ids = [
                    sd['stage_id'] for sd in self.all_stage_definitions
                    if sd.get('stage_id', 0) <= stage_id
                ]
            else:
                target_stage_ids = [
                    sd['stage_id'] for sd in self.all_stage_definitions
                    if sd.get('stage_id', 0) <= stage_id
                ]

            for sd in self.all_stage_definitions:
                if sd.get('stage_id') in target_stage_ids:
                    for cid in sd.get('class_indices', []):
                        fn_class_ids.add(int(cid))

        # Mapping from NYU40 ids to model indices
        try:
            nyu40_to_model = self.mappings.get('nyu40_to_model_idx', {})
        except Exception:
            nyu40_to_model = {}

        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        model.to(device_obj)
        model.eval()

        metrics_by_scene = {}

        def _safe_entropy_from_score(score: float) -> float:
            # Two-class proxy entropy using predicted score as P(correct)
            eps = 1e-6
            p = max(min(score, 1.0 - eps), eps)
            q = 1.0 - p
            return float(-p * _np.log(p) - q * _np.log(q))

        for idx, scene_info in enumerate(natural_scenes):
            if 'point_cloud' in scene_info and 'lidar_idx' in scene_info['point_cloud']:
                scene_id = scene_info['point_cloud']['lidar_idx']
            else:
                scene_id = scene_info.get('scene_id', f'scene_{idx}')

            # Construct points path from dataset root only.
            if hasattr(self, 'data_root') and self.data_root:
                points_file = _os.path.join(
                    str(self.data_root),
                    'points',
                    f'{scene_id}.bin',
                )
            else:
                print("    ⚠️ data_root is missing; skipping scene metric inference.")
                continue

            if not _os.path.exists(points_file):
                # Skip scenes without points (should be rare)
                continue

            if idx % 50 == 0:
                print(f"    ↪ Scoring scene {idx}/{len(natural_scenes)}: {scene_id}")

            try:
                with torch.no_grad():
                    infer_result = inference_detector(model, points_file)
            except Exception as _e:
                print(f"    ⚠️ Inference failed for scene {scene_id}: {_e}")
                continue

            # Unpack predictions from inference_detector
            predictions = None
            if isinstance(infer_result, tuple):
                result_list = infer_result[0]
            else:
                result_list = infer_result

            if isinstance(result_list, list) and len(result_list) > 0:
                if isinstance(result_list[0], dict):
                    predictions = result_list[0]
                elif (isinstance(result_list[0], (list, tuple)) and
                      len(result_list[0]) > 0 and
                      isinstance(result_list[0][0], dict)):
                    predictions = result_list[0][0]
            elif isinstance(result_list, dict):
                predictions = result_list

            if not predictions or 'scores_3d' not in predictions:
                continue

            scores_3d = predictions['scores_3d']
            labels_3d = predictions['labels_3d']
            boxes_3d = predictions['boxes_3d']

            # Move to device and apply confidence threshold
            if hasattr(scores_3d, 'to'):
                scores = scores_3d.to(device_obj)
                labels = labels_3d.to(device_obj)
                boxes = (boxes_3d.to(device_obj) if isinstance(
                    boxes_3d, DepthInstance3DBoxes) else boxes_3d)
            else:
                scores = torch.tensor(scores_3d, dtype=torch.float32,
                                      device=device_obj)
                labels = torch.tensor(labels_3d, dtype=torch.long,
                                      device=device_obj)
                if isinstance(boxes_3d, DepthInstance3DBoxes):
                    boxes = boxes_3d.to(device_obj)
                else:
                    boxes = DepthInstance3DBoxes(
                        torch.as_tensor(boxes_3d, dtype=torch.float32,
                                        device=device_obj),
                        box_dim=boxes_3d.shape[-1])

            keep_mask = scores >= float(conf_thr)
            if keep_mask.sum() == 0:
                # No confident detections → pure FN count
                scores = scores.new_zeros((0, ))
                labels = labels.new_zeros((0, ), dtype=torch.long)
                boxes_filt = DepthInstance3DBoxes(
                    torch.zeros((0, 6), dtype=torch.float32, device=device_obj),
                    box_dim=6)
            else:
                if max_boxes_per_scene is not None:
                    # Keep top-K scores before thresholding to bound compute
                    topk = min(int(max_boxes_per_scene), scores.shape[0])
                    top_vals, top_idx = torch.topk(scores, topk)
                    keep_mask = keep_mask & torch.zeros_like(
                        keep_mask, dtype=torch.bool).scatter(0, top_idx,
                                                             True)
                scores = scores[keep_mask]
                labels = labels[keep_mask]
                if isinstance(boxes, DepthInstance3DBoxes):
                    boxes_filt = boxes[keep_mask]
                else:
                    boxes_filt = DepthInstance3DBoxes(
                        boxes.tensor[keep_mask],
                        box_dim=boxes.tensor.shape[-1])

            # Prepare GT boxes / labels (mapped to model indices)
            annos = scene_info.get('annos', {})
            gt_boxes_np = annos.get('gt_boxes_upright_depth', _np.zeros(
                (0, 6), dtype=_np.float32)).astype(_np.float32)
            gt_labels_nyu40 = annos.get('class', _np.zeros(
                (0, ), dtype=_np.int64)).astype(_np.int64)

            valid_gt_mask = _np.array([
                (int(ny) in nyu40_to_model)
                for ny in gt_labels_nyu40
            ],
                                      dtype=bool)
            gt_boxes_np = gt_boxes_np[valid_gt_mask]
            gt_labels_nyu40 = gt_labels_nyu40[valid_gt_mask]

            if gt_boxes_np.shape[0] > 0:
                gt_labels_model = _np.array(
                    [nyu40_to_model[int(ny)] for ny in gt_labels_nyu40],
                    dtype=_np.int64)
                gt_boxes = DepthInstance3DBoxes(
                    torch.from_numpy(gt_boxes_np).to(device_obj),
                    box_dim=gt_boxes_np.shape[-1],
                    with_yaw=False,
                    origin=(0.5, 0.5, 0.5))
            else:
                gt_labels_model = _np.zeros((0, ), dtype=_np.int64)
                gt_boxes = DepthInstance3DBoxes(
                    torch.zeros((0, 6), dtype=torch.float32, device=device_obj),
                    box_dim=6,
                    with_yaw=False,
                    origin=(0.5, 0.5, 0.5))

            # Compute IoU matrix if we have both GT and predictions
            if (gt_boxes.tensor.shape[0] > 0 and
                    isinstance(boxes_filt, DepthInstance3DBoxes) and
                    boxes_filt.tensor.shape[0] > 0):
                ious = gt_boxes.overlaps(boxes_filt)
            else:
                ious = torch.zeros(
                    (gt_boxes.tensor.shape[0], boxes_filt.tensor.shape[0]),
                    device=device_obj)

            # Determine TP/FP per prediction
            tp_flags = []
            for j in range(scores.shape[0]):
                pred_cls = int(labels[j].item())
                if ious.shape[0] == 0:
                    tp_flags.append(False)
                    continue
                gt_mask = (torch.as_tensor(gt_labels_model,
                                           device=device_obj) == pred_cls)
                if not gt_mask.any():
                    tp_flags.append(False)
                    continue
                ious_for_pred = ious[gt_mask, j]
                tp_flags.append(bool((ious_for_pred >= float(iou_thr)).any()))

            tp_count = sum(1 for flag in tp_flags if flag)
            fp_count = sum(1 for flag in tp_flags if not flag)

            # U_det: reliability-weighted entropy of predictions
            if scores.numel() > 0:
                entropies = _np.array(
                    [_safe_entropy_from_score(float(s)) for s in scores])
                weights = []
                for s, is_tp in zip(scores, tp_flags):
                    s_val = float(s)
                    if is_tp:
                        weights.append(s_val)
                    else:
                        # Down-weight FPs
                        weights.append(0.5 * s_val)
                weights = _np.array(weights, dtype=_np.float32)
                denom = float(weights.sum()) + 1e-6
                u_det = float((weights * entropies).sum() / denom)
            else:
                u_det = 0.0

            # U_undet: normalized FN count over configured classes
            fn_count = 0
            gt_total = 0
            if gt_boxes.tensor.shape[0] > 0:
                gt_labels_model_t = torch.as_tensor(
                    gt_labels_model, device=device_obj)
                for i_gt in range(gt_boxes.tensor.shape[0]):
                    cls_i = int(gt_labels_model_t[i_gt].item())
                    if fn_class_ids and cls_i not in fn_class_ids:
                        continue
                    gt_total += 1
                    if ious.shape[1] == 0:
                        fn_count += 1
                        continue
                    pred_mask = (labels == cls_i)
                    if not pred_mask.any():
                        fn_count += 1
                        continue
                    ious_for_gt = ious[i_gt, pred_mask]
                    if not (ious_for_gt >= float(iou_thr)).any():
                        fn_count += 1

            u_undet_raw = float(fn_count / float(gt_total)) if gt_total > 0 else 0.0

            # Diversity histogram: simple per-class counts from predictions
            diversity_hist = {}
            for cls_id in labels.tolist():
                cls_int = int(cls_id)
                diversity_hist[cls_int] = diversity_hist.get(cls_int, 0) + 1

            metrics_by_scene[scene_id] = {
                'uncertainty': {
                    'U_det': u_det,
                    'U_undet_raw': u_undet_raw,
                    'tp_count': int(tp_count),
                    'fp_count': int(fp_count),
                    'fn_count': int(fn_count),
                },
                'diversity_hist': diversity_hist,
            }

        # Normalize U_undet over candidate scenes and compute final S_unc
        u_undet_values = [
            m['uncertainty']['U_undet_raw']
            for m in metrics_by_scene.values()
        ]
        if len(u_undet_values) > 1:
            mean_u = float(_np.mean(u_undet_values))
            std_u = float(_np.std(u_undet_values)) or 1.0
        else:
            mean_u, std_u = 0.0, 1.0

        ratio = 3.0
        shift = 0.5
        for scene_id, m in metrics_by_scene.items():
            u_undet_raw = m['uncertainty']['U_undet_raw']
            z = (u_undet_raw - mean_u) / (std_u * ratio)
            tilde = max(0.0, min(1.0, shift + z))
            m['uncertainty']['U_undet'] = tilde
            u_det_val = m['uncertainty']['U_det']
            m['uncertainty']['S_unc'] = (1.0 + tilde) * u_det_val

        # Persist metrics for analysis/debugging
        out_path = None
        try:
            metrics_dir = None
            if getattr(self, 'paths', None) is not None:
                metrics_dir = self.paths.memory_bank_scores_dir()
            elif self.work_dir:
                metrics_dir = _os.path.join(self.work_dir, 'memory_bank', 'scores')

            if metrics_dir is not None:
                if hasattr(metrics_dir, 'mkdir'):
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    out_path = metrics_dir / f'uncertainty_diversity_scores_stage_{stage_id}.json'
                else:
                    _os.makedirs(metrics_dir, exist_ok=True)
                    out_path = _os.path.join(
                        metrics_dir,
                        f'uncertainty_diversity_scores_stage_{stage_id}.json')
        except Exception:
            out_path = None

        if out_path is not None:
            try:
                # Convert numpy types to native for JSON
                serializable = {}
                for sid, m in metrics_by_scene.items():
                    serializable[sid] = {
                        'uncertainty': {
                            'U_det': float(m['uncertainty']['U_det']),
                            'U_undet_raw': float(
                                m['uncertainty']['U_undet_raw']),
                            'U_undet': float(m['uncertainty']['U_undet']),
                            'S_unc': float(m['uncertainty']['S_unc']),
                            'tp_count': int(m['uncertainty']['tp_count']),
                            'fp_count': int(m['uncertainty']['fp_count']),
                            'fn_count': int(m['uncertainty']['fn_count']),
                        },
                        'diversity_hist': {
                            str(int(k)): int(v)
                            for k, v in m['diversity_hist'].items()
                        }
                    }

                # pathlib.Path or str
                if hasattr(out_path, 'parent'):
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, 'w') as f:
                        _json.dump(
                            {
                                'stage_id': stage_id,
                                'strategy': getattr(
                                    mb, 'selection_strategy', None),
                                'candidates': serializable,
                            },
                            f,
                            indent=2)
                else:
                    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, 'w') as f:
                        _json.dump(
                            {
                                'stage_id': stage_id,
                                'strategy': getattr(
                                    mb, 'selection_strategy', None),
                                'candidates': serializable,
                            },
                            f,
                            indent=2)
                print(f"  💾 Uncertainty/diversity metrics saved: {out_path}")
            except Exception as _e:
                print(f"  ⚠️ Failed to save uncertainty/diversity metrics: {_e}")

        return metrics_by_scene


    def _remove_duplicate_boxes(self, boxes, labels, distance_threshold=0.1, iou_threshold=None, same_class_only=True):
        """Remove duplicate bounding boxes based on position and class.

        Args:
            boxes: Array of bounding boxes (N, 6) - [x,y,z,w,h,d]
            labels: Array of class labels (N,)
            distance_threshold: Maximum distance to consider boxes as duplicates

        Returns:
            Tuple of (unique_boxes, unique_labels)
        """
        import numpy as np

        if len(boxes) == 0:
            return boxes, labels

        # IoU-based dedup if requested
        if iou_threshold is not None:
            if same_class_only:
                return dedup_same_class_by_iou(np.asarray(boxes), np.asarray(labels), float(iou_threshold))
            # class-agnostic IoU dedup (rare): greedily suppress later overlaps
            arr = np.asarray(boxes, dtype=np.float32)[:, :6]
            L = np.asarray(labels, dtype=np.int64)
            keep = np.ones((arr.shape[0],), dtype=bool)
            for i in range(arr.shape[0]):
                if not keep[i]:
                    continue
                if i + 1 < arr.shape[0]:
                    iou = pairwise_aligned_iou(arr[i:i+1], arr[i+1:])
                    sup = (iou[0] >= float(iou_threshold))
                    keep[i+1:] &= ~sup
            return np.asarray(boxes)[keep], L[keep]

        # Legacy center-distance heuristic (same-class)
        keep_mask = np.ones(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            if not keep_mask[i]:
                continue  # Already marked for removal
            for j in range(i + 1, len(boxes)):
                if not keep_mask[j]:
                    continue
                if labels[i] == labels[j]:
                    center_i = boxes[i][:3]
                    center_j = boxes[j][:3]
                    distance = np.linalg.norm(center_i - center_j)
                    if distance < distance_threshold:
                        keep_mask[j] = False
        return np.asarray(boxes)[keep_mask], np.asarray(labels)[keep_mask]

    def _merge_scene_labels(self, natural_scene: Dict, replay_scene: Dict, scene_id: str) -> Dict:
        """Merge labels from natural and replay versions of the same scene.

        Args:
            natural_scene: Scene data from current stage (more complete labels)
            replay_scene: Scene data from memory bank (filtered labels from previous stage)
            scene_id: Scene identifier for debugging

        Returns:
            Merged scene with combined annotations
        """
        import numpy as np

        # Start with natural scene as base (it has the most up-to-date structure)
        merged_scene = copy.deepcopy(natural_scene)

        # Get annotations from both versions
        natural_annos = natural_scene.get('annos', {})
        replay_annos = replay_scene.get('annos', {})

        # Count original objects for statistics
        natural_count = natural_annos.get('gt_num', 0)
        replay_count = replay_annos.get('gt_num', 0)

        if natural_count == 0 and replay_count == 0:
            return merged_scene

        # Prepare arrays for merging
        all_boxes = []
        all_labels = []

        # Add natural scene annotations
        if natural_count > 0:
            natural_boxes = natural_annos.get('gt_boxes_upright_depth', [])
            natural_labels = natural_annos.get('class', [])

            if len(natural_boxes) > 0 and len(natural_labels) > 0:
                all_boxes.extend(natural_boxes)
                all_labels.extend(natural_labels)

        # Add replay scene annotations
        if replay_count > 0:
            replay_boxes = replay_annos.get('gt_boxes_upright_depth', [])
            replay_labels = replay_annos.get('class', [])

            if len(replay_boxes) > 0 and len(replay_labels) > 0:
                all_boxes.extend(replay_boxes)
                all_labels.extend(replay_labels)

        if not all_boxes or not all_labels:
            # If no valid annotations found, return natural scene
            return merged_scene

        # Convert to numpy arrays
        all_boxes = np.array(all_boxes)
        all_labels = np.array(all_labels)

        # Remove duplicate annotations
        if self.enable_gt_merge_iou:
            merged_boxes, merged_labels = self._remove_duplicate_boxes(
                all_boxes, all_labels, iou_threshold=self.gt_merge_iou_thr, same_class_only=True
            )
        else:
            merged_boxes, merged_labels = self._remove_duplicate_boxes(
                all_boxes, all_labels, distance_threshold=0.1
            )

        # Update merged scene annotations
        merged_scene['annos']['gt_boxes_upright_depth'] = merged_boxes
        merged_scene['annos']['class'] = merged_labels
        merged_scene['annos']['gt_num'] = len(merged_boxes)

        # Add metadata for debugging
        merged_scene['is_merged'] = True
        merged_scene['natural_object_count'] = natural_count
        merged_scene['replay_object_count'] = replay_count
        merged_scene['final_object_count'] = len(merged_boxes)
        merged_scene['merged_from_stages'] = [replay_scene.get('replay_from_stage', 'unknown')]
        merged_scene['merge_scene_id'] = scene_id

        # Preserve replay-seat provenance for reviewing-aware resampling.
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
        merged_scene['replay_unique_ids'] = [str(x) for x in merged_ids if str(x)]

        merged_stages = []
        merged_stages.extend(_as_list(natural_scene.get('replay_from_stages', None)))
        merged_stages.extend(_as_list(replay_scene.get('replay_from_stages', None)))
        rs = replay_scene.get('replay_from_stage', None)
        if rs is not None:
            merged_stages.append(int(rs))
        merged_scene['replay_from_stages'] = [int(x) for x in merged_stages]
        merged_scene['replay_from_stage'] = (
            int(merged_scene['replay_from_stages'][-1])
            if merged_scene.get('replay_from_stages') else
            replay_scene.get('replay_from_stage', None)
        )

        # ENHANCED: Add explicit scene identity tracking
        current_stage = self.stage_definition.get('stage_id', 1) if self.stage_definition else 1
        replay_stage = replay_scene.get('replay_from_stage', 'unknown')

        # Create comprehensive scene identity
        scene_identity = {
            'type': 'multi_identity',
            'sources': {
                'natural': {
                    'stage': current_stage,
                    'object_count': natural_count,
                    'classes_present': list(set(natural_annos.get('class', [])))
                },
                'memory_bank': {
                    'stage': replay_stage,
                    'object_count': replay_count,
                    'classes_present': list(set(replay_annos.get('class', [])))
                }
            },
            'merged_result': {
                'final_object_count': len(merged_boxes),
                'final_classes': list(set(merged_labels.tolist())),
                'duplicates_removed': (natural_count + replay_count) - len(merged_boxes)
            },
            'display_tag': f"[Natural:Stage{current_stage}+Memory:Stage{replay_stage}]"
        }

        merged_scene['scene_identity'] = scene_identity

        return merged_scene

    def _add_scene_identity_tracking(self, scene: Dict, identity_type: str) -> Dict:
        """Add explicit scene identity tracking for non-merged scenes.

        Args:
            scene: Scene data to enhance with identity
            identity_type: 'natural_only' or 'memory_only'

        Returns:
            Scene with identity tracking added
        """
        scene = copy.deepcopy(scene)
        current_stage = self.stage_definition.get('stage_id', 1) if self.stage_definition else 1

        if identity_type == 'memory_only':
            replay_stage = scene.get('replay_from_stage', 'unknown')
            object_count = scene.get('annos', {}).get('gt_num', 0)
            classes_present = list(set(scene.get('annos', {}).get('class', [])))

            scene_identity = {
                'type': 'memory_only',
                'sources': {
                    'memory_bank': {
                        'stage': replay_stage,
                        'object_count': object_count,
                        'classes_present': classes_present
                    }
                },
                'display_tag': f"[Memory:Stage{replay_stage}]"
            }
        elif identity_type == 'natural_only':
            object_count = scene.get('annos', {}).get('gt_num', 0)
            classes_present = list(set(scene.get('annos', {}).get('class', [])))

            scene_identity = {
                'type': 'natural_only',
                'sources': {
                    'natural': {
                        'stage': current_stage,
                        'object_count': object_count,
                        'classes_present': classes_present
                    }
                },
                'display_tag': f"[Natural:Stage{current_stage}]"
            }
        else:
            raise ValueError(f"Unknown identity_type: {identity_type}")

        scene['scene_identity'] = scene_identity
        return scene

    def _apply_reviewing_sampling_if_enabled(self) -> None:
        """Resample training set for segmented reviewing/LD runs.

        Coverage-preserving policy:
          - Every memory candidate is visited at least once per segment.
          - Additional memory revisits are sampled by reviewing weights.
          - Extra memory revisits are paid by reducing natural samples.
        """
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
        seed = int(seed)

        baseline_inner_len = cfg.get('baseline_inner_len', None)
        if baseline_inner_len is not None:
            baseline_inner_len = int(baseline_inner_len)
            if baseline_inner_len <= 0:
                baseline_inner_len = None

        strict_memory_coverage = bool(cfg.get('strict_memory_coverage', True))

        memory_share_max = cfg.get('memory_share_max', None)
        if memory_share_max is not None:
            memory_share_max = float(memory_share_max)
            if memory_share_max <= 0.0 or memory_share_max >= 1.0:
                memory_share_max = None

        candidates = list(self.data_infos)
        if not candidates:
            raise RuntimeError('reviewing_sampling has no candidates to sample from.')
        if baseline_inner_len is not None:
            target_len = int(max(target_len, baseline_inner_len))

        rng = np.random.RandomState(seed)
        sample_weights = np.ones((len(candidates),), dtype=np.float64)
        is_memory = np.zeros((len(candidates),), dtype=bool)
        memory_indices = []
        natural_indices = []

        for i, info in enumerate(candidates):
            if bool(info.get('is_replay', False)):
                uid = str(info.get('replay_unique_id', ''))
                if uid:
                    is_memory[i] = True
                    sample_weights[i] = max(1.0, float(weights_by_uid.get(uid, 1.0)))
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

        if memory_share_max is not None:
            mem_sum = float(sample_weights[is_memory].sum()) if bool(is_memory.any()) else 0.0
            nat_sum = float(sample_weights[~is_memory].sum()) if bool((~is_memory).any()) else 0.0
            denom = mem_sum + nat_sum
            if denom > 0.0 and nat_sum > 0.0:
                share = mem_sum / denom
                if share > memory_share_max and mem_sum > 0.0:
                    scale = (memory_share_max / max(1e-6, (1.0 - memory_share_max))) * (nat_sum / mem_sum)
                    sample_weights[is_memory] *= float(scale)

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
                selected_indices.extend([int(memory_indices[int(j)]) for j in extra_idx.tolist()])

        if k_nat_target > 0:
            if natural_indices:
                if k_nat_target <= len(natural_indices):
                    nat_idx = rng.choice(
                        len(natural_indices),
                        size=int(k_nat_target),
                        replace=False,
                    )
                    selected_indices.extend([int(natural_indices[int(j)]) for j in nat_idx.tolist()])
                else:
                    nat_weights = np.asarray(
                        [float(sample_weights[int(i)]) for i in natural_indices],
                        dtype=np.float64,
                    )
                    nat_total = float(nat_weights.sum())
                    if not np.isfinite(nat_total) or nat_total <= 0.0:
                        nat_probs = np.ones((len(natural_indices),), dtype=np.float64) / float(len(natural_indices))
                    else:
                        nat_probs = nat_weights / nat_total
                    nat_idx = rng.choice(
                        len(natural_indices),
                        size=int(k_nat_target),
                        replace=True,
                        p=nat_probs,
                    )
                    selected_indices.extend([int(natural_indices[int(j)]) for j in nat_idx.tolist()])
            elif n_mem > 0:
                # Fallback: no natural candidates are available.
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
                selected_indices.extend([int(memory_indices[int(j)]) for j in extra_mem_idx.tolist()])

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
            'memory_never_seen_count': int(max(0, n_mem - len(seen_memory_candidates))),
            'memory_coverage_ratio': (
                float(len(seen_memory_candidates) / float(n_mem))
                if n_mem > 0 else float('nan')
            ),
        }
        if hasattr(self, '_set_group_flag'):
            try:
                self._set_group_flag()
            except Exception:
                pass

    def _load_pseudo_labels(self):
        """Load pseudo labels using unified path management."""
        if not self.stage_definition:
            return {}

        stage_id = self.stage_definition.get('stage_id', 1)
        if stage_id <= 1:
            return {}

        # Use unified paths if available
        if self.paths:
            # Try primary location first
            pseudo_label_file = self.paths.pseudo_label_file(stage_id)
            if pseudo_label_file.exists():
                import pickle
                with open(pseudo_label_file, 'rb') as f:
                    pseudo_labels = pickle.load(f)

                # Enforce canonical pseudo schema for all loading paths.
                self._validate_pseudo_canonical(pseudo_labels)

                # Gather statistics
                total_detections = 0
                class_counts = {}
                confidence_stats = []

                for scene_id, labels in pseudo_labels.items():
                    if isinstance(labels, dict) and 'labels' in labels:
                        scene_labels = labels['labels']
                        scene_scores = labels.get('scores', [])
                        total_detections += len(scene_labels)

                        for label in scene_labels:
                            class_counts[int(label)] = class_counts.get(int(label), 0) + 1

                        if len(scene_scores) > 0:
                            confidence_stats.extend(scene_scores)

                # Display statistics
                print(f"\n📊 PSEUDO LABELS LOADED (Stage {stage_id})")
                print(f"  Source: {pseudo_label_file}")
                print(f"  Scenes with labels: {len(pseudo_labels)}")
                print(f"  Total detections: {total_detections:,}")
                if confidence_stats:
                    conf_array = np.array(confidence_stats)
                    print(f"  Confidence range: [{conf_array.min():.3f}, {conf_array.max():.3f}]")
                    print(f"  Mean confidence: {conf_array.mean():.3f}")
                if class_counts:
                    print(f"  Classes covered: {sorted(class_counts.keys())}")
                    print(f"  Detections per class:")
                    for cls_idx in sorted(class_counts.keys()):
                        print(f"    Class {cls_idx}: {class_counts[cls_idx]:,} detections")
                print()

                return pseudo_labels

            # Try legacy locations with helpful migration message
            legacy_path = self.paths.resolve_legacy_pseudo_labels(stage_id)
            if legacy_path:
                import pickle
                with open(legacy_path, 'rb') as f:
                    pseudo_labels = pickle.load(f)

                # Enforce canonical pseudo schema for all loading paths.
                self._validate_pseudo_canonical(pseudo_labels)

                # Gather statistics for legacy path too
                total_detections = 0
                class_counts = {}

                for scene_id, labels in pseudo_labels.items():
                    if isinstance(labels, dict) and 'labels' in labels:
                        scene_labels = labels['labels']
                        total_detections += len(scene_labels)
                        for label in scene_labels:
                            class_counts[int(label)] = class_counts.get(int(label), 0) + 1

                print(f"\n📊 PSEUDO LABELS LOADED (Stage {stage_id}) - Legacy Path")
                print(f"  Source: {legacy_path}")
                print(f"  Scenes with labels: {len(pseudo_labels)}")
                print(f"  Total detections: {total_detections:,}")
                if class_counts:
                    print(f"  Classes covered: {sorted(class_counts.keys())}")
                print(f"  ⚠️ Consider running paths.migrate_from_legacy_structure() to update your experiment")
                print()

                return pseudo_labels

            print(f"\n⚠️ PSEUDO LABELS NOT FOUND")
            print(f"  Stage {stage_id} requires pseudo labels from Stage {stage_id - 1} model")
            print(f"  Expected location: {self.paths.pseudo_label_file(stage_id) if self.paths else 'Unknown'}")
            print()
            return {}
        else:
            # Fallback to legacy loading for backward compatibility
            return self._load_pseudo_labels_legacy()

    def _load_pregenerated_pseudo_labels(self, pregenerated_file: str):
        """
        Load pre-generated pseudo labels from file.

        This method loads pseudo labels that were pre-computed using the
        pregenerate_pseudo_labels module. Pre-generation significantly speeds
        up training by avoiding repeated inference during training iterations.

        Args:
            pregenerated_file: Path to pre-generated pseudo labels file

        Returns:
            Dictionary of pseudo labels by scene_id

        Raises:
            FileNotFoundError: If pre-generated file doesn't exist
            Exception: If loading fails for any reason
        """
        if not os.path.exists(pregenerated_file):
            stage_id = self.stage_definition.get('stage_id', 'Unknown')
            raise FileNotFoundError(
                f"Pre-generated pseudo labels not found: {pregenerated_file}\n"
                f"Stage {stage_id} requires pseudo labels to be pre-generated.\n"
                f"Please ensure pseudo labels have been generated before training."
            )

        try:
            import pickle
            with open(pregenerated_file, 'rb') as f:
                pseudo_labels = pickle.load(f)

            # Ensure canonical spec (strict)
            self._validate_pseudo_canonical(pseudo_labels)

            # Gather comprehensive statistics
            total_detections = 0
            class_counts = {}
            confidence_stats = []
            file_size_mb = os.path.getsize(pregenerated_file) / (1024 * 1024)

            for scene_id, labels in pseudo_labels.items():
                if isinstance(labels, dict):
                    if 'labels' in labels:
                        scene_labels = labels['labels']
                        scene_scores = labels.get('scores', [])
                        total_detections += len(scene_labels)

                        for label in scene_labels:
                            class_counts[int(label)] = class_counts.get(int(label), 0) + 1

                        if len(scene_scores) > 0:
                            confidence_stats.extend(scene_scores)

            # Display detailed statistics
            print(f"\n📊 PRE-GENERATED PSEUDO LABELS LOADED")
            print(f"  File: {pregenerated_file}")
            print(f"  File size: {file_size_mb:.2f} MB")
            print(f"  Scenes with labels: {len(pseudo_labels)}")
            print(f"  Total detections: {total_detections:,}")

            if confidence_stats:
                conf_array = np.array(confidence_stats)
                print(f"  Confidence statistics:")
                print(f"    Range: [{conf_array.min():.3f}, {conf_array.max():.3f}]")
                print(f"    Mean: {conf_array.mean():.3f} ± {conf_array.std():.3f}")
                print(f"    Percentiles: 25%={np.percentile(conf_array, 25):.3f}, 50%={np.percentile(conf_array, 50):.3f}, 75%={np.percentile(conf_array, 75):.3f}")

            if class_counts:
                # Clarify label spaces
                m = getattr(self, 'mappings', {}) or {}
                nyu2gci = m.get('nyu40_to_model_idx', {})
                gci2name = m.get('model_idx_to_name', {})
                detected_nyu = sorted(class_counts.keys())
                if nyu2gci and gci2name:
                    joined = [
                        f"NYU40 {int(ny)} -> GCI {int(nyu2gci.get(int(ny), -1))} ({gci2name.get(int(nyu2gci.get(int(ny), -1)), f'class_{int(nyu2gci.get(int(ny), -1))}')} )"
                        if int(ny) in nyu2gci else f"NYU40 {int(ny)}"
                        for ny in detected_nyu
                    ]
                    print(f"  Classes detected: {joined} ({len(class_counts)} total)")
                else:
                    print(f"  Classes detected (NYU40 ids): {detected_nyu} ({len(class_counts)} total)")
                print(f"  Top 5 classes by detection count:")
                sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for ny, count in sorted_classes:
                    if nyu2gci and gci2name and int(ny) in nyu2gci:
                        gci = nyu2gci[int(ny)]
                        name = gci2name.get(int(gci), f"class_{int(gci)}")
                        print(f"    NYU40 {int(ny)} -> GCI {int(gci)} ({name}): {count:,} detections")
                    else:
                        print(f"    NYU40 {int(ny)}: {count:,} detections")

            # Apply class-based or global confidence thresholding if specified
            cls_thresh_cfg = self.pseudo_label_config.get('class_thresholds_nyu40') or \
                              self.pseudo_label_config.get('class_thresholds')
            if cls_thresh_cfg:
                class_thresholds = {}
                for k, v in cls_thresh_cfg.items():
                    try:
                        class_thresholds[int(k)] = float(v)
                    except Exception:
                        continue
                filtered_labels = {}
                total_before = sum(len(pl.get('scores', [])) for pl in pseudo_labels.values())
                for scene_id, scene_labels in pseudo_labels.items():
                    labs = np.array(scene_labels.get('labels', []))
                    scs = np.array(scene_labels.get('scores', []))
                    if labs.size == 0 or scs.size == 0:
                        continue
                    thr = np.array([class_thresholds.get(int(l), 0.0) for l in labs], dtype=np.float32)
                    mask = scs >= thr
                    if mask.any():
                        filtered_scene = {}
                        for key, value in scene_labels.items():
                            if key in ['boxes', 'scores', 'labels']:
                                filtered_scene[key] = np.array(value)[mask]
                            else:
                                filtered_scene[key] = value
                        filtered_scene['num_detections'] = int(mask.sum())
                        filtered_labels[scene_id] = filtered_scene
                total_after = sum(len(pl.get('scores', [])) for pl in filtered_labels.values())
                print(f"\n🔍 CLASS-BASED CONFIDENCE FILTERING APPLIED")
                print(f"  Classes with thresholds: {sorted(class_thresholds.keys())}")
                print(f"  Detections: {total_before:,} → {total_after:,} ({100*total_after/max(1,total_before):.1f}% retained)")
                return filtered_labels

            # Otherwise, apply global threshold if provided
            confidence_threshold = self.pseudo_label_config.get('confidence_threshold', 0.45)
            if confidence_threshold > 0.1:  # Only filter if threshold is higher than generation threshold
                filtered_labels = {}
                total_before = sum(len(pl.get('scores', [])) for pl in pseudo_labels.values())

                for scene_id, scene_labels in pseudo_labels.items():
                    if 'scores' in scene_labels and len(scene_labels['scores']) > 0:
                        scores = np.array(scene_labels['scores'])
                        mask = scores >= confidence_threshold

                        if mask.any():
                            filtered_scene = {}
                            for key, value in scene_labels.items():
                                if key in ['boxes', 'scores', 'labels']:
                                    filtered_scene[key] = np.array(value)[mask]
                                else:
                                    filtered_scene[key] = value
                            filtered_scene['num_detections'] = mask.sum()
                            filtered_labels[scene_id] = filtered_scene

                total_after = sum(len(pl.get('scores', [])) for pl in filtered_labels.values())
                # Gather class distribution after filtering
                filtered_class_counts = {}
                for scene_id, labels in filtered_labels.items():
                    if 'labels' in labels:
                        for label in labels['labels']:
                            filtered_class_counts[int(label)] = filtered_class_counts.get(int(label), 0) + 1

                print(f"\n🔍 CONFIDENCE FILTERING APPLIED")
                print(f"  Threshold: {confidence_threshold}")
                print(f"  Detections: {total_before:,} → {total_after:,} ({100*total_after/total_before:.1f}% retained)")
                print(f"  Scenes with detections: {len(pseudo_labels)} → {len(filtered_labels)}")
                if filtered_class_counts:
                    m = getattr(self, 'mappings', {}) or {}
                    nyu2gci = m.get('nyu40_to_model_idx', {})
                    gci2name = m.get('model_idx_to_name', {})
                    detected_nyu = sorted(filtered_class_counts.keys())
                    if nyu2gci and gci2name:
                        joined = [
                            f"NYU40 {int(ny)} -> GCI {int(nyu2gci.get(int(ny), -1))} ({gci2name.get(int(nyu2gci.get(int(ny), -1)), f'class_{int(nyu2gci.get(int(ny), -1))}')} )"
                            if int(ny) in nyu2gci else f"NYU40 {int(ny)}"
                            for ny in detected_nyu
                        ]
                        print(f"  Classes after filtering: {joined}")
                    else:
                        print(f"  Classes after filtering (NYU40 ids): {detected_nyu}")
                print()
                return filtered_labels
            else:
                return pseudo_labels

        except Exception as e:
            stage_id = self.stage_definition.get('stage_id', 'Unknown')
            raise Exception(
                f"Failed to load pre-generated pseudo labels: {e}\n"
                f"File: {pregenerated_file}\n"
                f"Stage {stage_id} training cannot continue without valid pseudo labels."
            )

    def _validate_pseudo_canonical(self, pseudo_labels):
        """Validate that pseudo labels comply with the canonical coordinate spec.

        Raises ValueError on mismatch with actionable hints.
        """
        if not isinstance(pseudo_labels, dict) or not pseudo_labels:
            raise ValueError("Pseudo labels must be a non-empty dict mapping scene_id → record")
        import numpy as _np
        checked = 0
        for sid, rec in pseudo_labels.items():
            if sid == '__meta__':
                continue
            if not isinstance(rec, dict):
                raise ValueError(f"Scene {sid}: record must be dict")
            for k in ['boxes', 'labels', 'scores', 'num_detections', 'label_space', 'center_type', 'axis_aligned', 'box_type']:
                if k not in rec:
                    raise ValueError(f"Scene {sid}: missing key '{k}'. Convert with tools/diagnostics/convert_pseudo_to_canonical.py")
            boxes = _np.asarray(rec['boxes'])
            labels = _np.asarray(rec['labels'])
            scores = _np.asarray(rec['scores'])
            if boxes.ndim != 2 or boxes.shape[1] != 6:
                raise ValueError(f"Scene {sid}: boxes must be (N,6)")
            if rec['label_space'] != 'nyu40':
                raise ValueError(f"Scene {sid}: label_space must be 'nyu40'")
            if str(rec['center_type']).lower() != 'gravity':
                raise ValueError(f"Scene {sid}: center_type must be 'gravity'")
            if rec['axis_aligned'] is not True:
                raise ValueError(f"Scene {sid}: axis_aligned must be True")
            if rec['box_type'] != 'upright_depth_6d':
                raise ValueError(f"Scene {sid}: box_type must be 'upright_depth_6d'")
            if labels.shape[0] != boxes.shape[0] or scores.shape[0] != boxes.shape[0]:
                raise ValueError(f"Scene {sid}: labels/scores must match boxes length")
            if int(rec['num_detections']) != boxes.shape[0]:
                raise ValueError(f"Scene {sid}: num_detections incorrect")
            checked += 1
        if checked == 0:
            raise ValueError("No scenes validated")
        print(f"✅ Canonical pseudo labels validated: {checked} scenes")

    def _load_pseudo_labels_legacy(self):
        """Legacy pseudo label loading (kept for reference)."""
        if not self.stage_definition:
            return {}

        stage_id = self.stage_definition.get('stage_id', 1)
        if stage_id <= 1:
            return {}

        # Primary: Look in current work_dir/pseudo_labels/
        if self.work_dir:
            pseudo_label_file = os.path.join(
                self.work_dir,
                'pseudo_labels',
                f'stage_{stage_id}_pseudo_labels.pkl'
            )
        # Fallback: User-specified directory
        elif self.pseudo_label_dir:
            pseudo_label_file = os.path.join(
                self.pseudo_label_dir,
                f'stage_{stage_id}_pseudo_labels.pkl'
            )
        else:
            # Last resort: Look in incremental_logs/pseudo_label_based/
            pseudo_label_file = f'incremental_logs/pseudo_label_based/stage_{stage_id}_pseudo_labels.pkl'

        if os.path.exists(pseudo_label_file):
            import pickle
            with open(pseudo_label_file, 'rb') as f:
                pseudo_labels = pickle.load(f)
            self._validate_pseudo_canonical(pseudo_labels)
            print(f"✅ Loaded pseudo labels for {len(pseudo_labels)} scenes from {pseudo_label_file}")
            return pseudo_labels
        else:
            print(f"⚠️ No pseudo labels found at {pseudo_label_file}")

            # Auto-generate pseudo labels if enabled and model is available
            if self.pseudo_label_config.get('auto_generate', True):
                print("🔄 Attempting to auto-generate pseudo labels...")
                try:
                    # Check if previous stage model exists for generation
                    prev_stage = stage_id - 1
                    if self.paths:
                        model_path = str(self.paths.checkpoint_file(prev_stage))
                    elif self.work_dir:
                        model_path = os.path.join(self.work_dir, f'stage_{prev_stage}', 'latest.pth')
                    else:
                        model_path = None

                    if model_path and os.path.exists(model_path):
                        print(f"📦 Found model at {model_path}, generating pseudo labels...")
                        pseudo_labels = self._generate_pseudo_labels_from_model(model_path)

                        # Save generated pseudo labels
                        if pseudo_labels:
                            os.makedirs(os.path.dirname(pseudo_label_file), exist_ok=True)
                            import pickle
                            with open(pseudo_label_file, 'wb') as f:
                                pickle.dump(pseudo_labels, f)
                            print(f"💾 Saved {len(pseudo_labels)} pseudo labels to {pseudo_label_file}")
                            return pseudo_labels
                        else:
                            print(f"⚠️ Previous stage model not found at {model_path}")
                except Exception as e:
                    print(f"❌ Failed to auto-generate pseudo labels: {e}")

            return {}

    def prepare_train_data(self, index):
        """Prepare training data with pseudo labels if available."""
        # Get standard training data
        data = super().prepare_train_data(index)

        # Add pseudo labels if available (post-pipeline) ONLY when pre-injection was not applied
        if self.use_pseudo_labels and self.pseudo_labels and not getattr(self, 'pseudo_injected_pre_pipeline', False):
            # Extract scene_id from nested structure (handle different formats)
            info = self.data_infos[index]
            if ((not bool(getattr(self, 'apply_pseudo_to_memory_scenes', False)))
                    and is_memory_or_merged_scene(info)):
                return data
            scene_id = None

            if 'sample_idx' in info:
                scene_id = info['sample_idx']  # Direct format
            elif 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
                scene_id = info['point_cloud']['lidar_idx']  # ScanNet nested format
            else:
                # Fallback: try to extract from file path
                if 'pts_path' in info:
                    import os
                    scene_id = os.path.basename(info['pts_path']).split('.')[0]

            if scene_id and scene_id in self.pseudo_labels:
                data = self._add_pseudo_labels_to_data(data, scene_id)

        return data

    def _add_pseudo_labels_to_data(self, data, scene_id):
        """Add pseudo labels to training data with NYU40 to model index conversion.

        This method now properly converts NYU40 IDs from stored pseudo labels
        to stage-specific model indices, ensuring compatibility with the current
        training stage.
        """
        pseudo_data = self.pseudo_labels[scene_id]

        # Convert pseudo labels to the same format as ground truth
        import torch
        from mmdet3d.core.bbox import DepthInstance3DBoxes

        # Get existing data (unwrap DataContainers if present)
        gt_bboxes_3d = data['gt_bboxes_3d']
        gt_labels_3d = data['gt_labels_3d']
        from mmcv.parallel import DataContainer as DC
        bboxes_dc = None
        labels_dc = None
        if isinstance(gt_bboxes_3d, DC):
            bboxes_dc = gt_bboxes_3d
            gt_bboxes_3d = gt_bboxes_3d.data
        if isinstance(gt_labels_3d, DC):
            labels_dc = gt_labels_3d
            gt_labels_3d = gt_labels_3d.data

        # Get pseudo label NYU40 IDs and convert to model indices
        pseudo_nyu40_ids = pseudo_data['labels']  # These are NYU40 IDs
        # Be robust to missing or list-form boxes
        pseudo_boxes_np = pseudo_data.get('boxes', None)
        if pseudo_boxes_np is None:
            # No boxes available; skip adding pseudo labels for this scene
            if self.debug_mode:
                print(f"Scene {scene_id}: No 'boxes' in pseudo labels; skipping pseudo augmentation")
            return data
        import numpy as _np
        if not hasattr(pseudo_boxes_np, 'shape'):
            try:
                pseudo_boxes_np = _np.asarray(pseudo_boxes_np, dtype=_np.float32)
            except Exception:
                if self.debug_mode:
                    print(f"Scene {scene_id}: Failed to convert 'boxes' to ndarray; skipping pseudo augmentation")
                return data
        pseudo_scores = pseudo_data.get('scores', np.ones(len(pseudo_nyu40_ids)))

        # Optional box-space correction if declared as bottom-centered
        try:
            if isinstance(pseudo_data.get('box_space', ''), str) and pseudo_data.get('box_space', '').lower() == 'bottom':
                if pseudo_boxes_np.shape[1] >= 6:
                    pseudo_boxes_np[:, 2] = pseudo_boxes_np[:, 2] + pseudo_boxes_np[:, 5] * 0.5
        except Exception:
            pass

        # CRITICAL: For pseudo labels, use PREVIOUS-STAGE classes only (exclude current stage)
        # to avoid leaking new-class labels from a model that hasn't learned them yet.
        valid_classes_for_pseudo = set()
        try:
            # Determine index of current stage within all_stage_definitions
            curr_id = self.stage_definition.get('stage_id', None) if self.stage_definition else None
            idx = None
            if curr_id is not None and hasattr(self, 'all_stage_definitions'):
                for i, sd in enumerate(self.all_stage_definitions):
                    if sd.get('stage_id') == curr_id:
                        idx = i
                        break
            if idx is None and hasattr(self, 'stage_idx'):
                idx = int(self.stage_idx)
            # Aggregate class indices from previous stages only
            if idx is not None and hasattr(self, 'all_stage_definitions'):
                for sd in self.all_stage_definitions[:max(0, idx)]:
                    valid_classes_for_pseudo.update(sd.get('class_indices', []))
            # Fallback: if we couldn't determine previous stages, keep it empty to avoid leakage
        except Exception:
            valid_classes_for_pseudo = set()

        # Convert NYU40 IDs to model indices and filter by all seen classes
        nyu40_to_model = self.mappings.get('nyu40_to_model_idx', {}) if hasattr(self, 'mappings') else {}
        if not nyu40_to_model and hasattr(self, 'nyu40_to_model_id'):
            nyu40_to_model = self.nyu40_to_model_id

        valid_mask = []
        converted_labels = []

        for nyu40_id in pseudo_nyu40_ids:
            if nyu40_id in nyu40_to_model:
                model_idx = nyu40_to_model[nyu40_id]
                # Keep if this class is in all seen classes (preserves old knowledge)
                if model_idx in valid_classes_for_pseudo:
                    valid_mask.append(True)
                    converted_labels.append(model_idx)
                else:
                    valid_mask.append(False)
            else:
                valid_mask.append(False)

        valid_mask = _np.array(valid_mask)

        if not valid_mask.any():
            # No valid pseudo labels for this stage, return original data
            return data

        # Filter boxes and convert labels
        filtered_boxes = pseudo_boxes_np[valid_mask]
        filtered_labels = _np.array(converted_labels, dtype=_np.int64)

        # Convert to torch tensors
        pseudo_boxes = torch.from_numpy(filtered_boxes).float()
        pseudo_labels = torch.from_numpy(filtered_labels).long()

        # Create pseudo bboxes in the same format
        pseudo_bboxes_3d = DepthInstance3DBoxes(pseudo_boxes, box_dim=pseudo_boxes.shape[-1])

        # Combine real and pseudo labels
        combined_bboxes = gt_bboxes_3d.cat([gt_bboxes_3d, pseudo_bboxes_3d])
        combined_labels = torch.cat([gt_labels_3d, pseudo_labels])

        # Update data
        # Rewrap into DataContainers if originally wrapped
        if bboxes_dc is not None:
            # DepthInstance3DBoxes must be kept on CPU to avoid scatter errors
            data['gt_bboxes_3d'] = DC(combined_bboxes, cpu_only=True, stack=False)
        else:
            data['gt_bboxes_3d'] = combined_bboxes
        if labels_dc is not None:
            data['gt_labels_3d'] = DC(combined_labels, stack=False)
        else:
            data['gt_labels_3d'] = combined_labels

        # Log filtering statistics for debugging
        if self.debug_mode and valid_mask.sum() < len(valid_mask):
            print(f"Scene {scene_id}: Filtered {len(valid_mask) - valid_mask.sum()}/{len(valid_mask)} "
                  f"pseudo labels (kept {valid_mask.sum()} for all seen classes {sorted(valid_classes_for_pseudo)})")

        return data

    def _generate_pseudo_labels_from_model(self, model_path):
        """Generate pseudo labels using a trained model with proper coordinate alignment.

        Args:
            model_path (str): Path to trained model checkpoint

        Returns:
            dict: Generated pseudo labels {scene_id: predictions}
        """
        try:
            from mmdet3d.apis import init_detector
            import torch

            # Load the model
            print(f"   Loading model from {model_path}")

            # Get config path from work directory structure
            config_path = None
            if self.work_dir:
                # Look for config in parent directory
                parent_dir = os.path.dirname(self.work_dir)
                for f in os.listdir(parent_dir):
                    if f.endswith('.py') and 'config' in f.lower():
                        config_path = os.path.join(parent_dir, f)
                        break

            if not config_path:
                # Fallback to portable dynamic-head config (relative path)
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                           'configs', 'incremental', 'scene_based',
                                           'tr3d_pure_finetuning_frequency_correct.py')

            # Use init_model to ensure we can tweak config if needed
            from mmcv import Config
            from mmdet3d.apis import init_model
            cfg = Config.fromfile(config_path)
            # Ensure dynamic_head variant for evaluation
            if hasattr(cfg.data, 'val'):
                cfg.data.val.type = 'ScanNetDataset'
                cfg.data.val.variant = 'dynamic_head'
                cfg.data.val.test_mode = True
            model = init_model(cfg, model_path, device='cuda:0')
            print(f"   Model loaded successfully")

            # Use the existing method but with proper coordinate alignment
            return self._generate_pseudo_labels_with_model(model)

        except Exception as e:
            print(f"❌ Failed to load model and generate pseudo labels: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _generate_pseudo_labels_with_model(self, model):
        """Generate pseudo labels using provided model with proper coordinate alignment."""
        stage_id = self.stage_definition.get('stage_id', 1)
        if stage_id <= 1:
            print("⚠️ No previous stage for pseudo labels")
            return {}

        # Extract stage definitions from config (avoid hardcoding)
        if not self.all_stage_definitions:
            print("⚠️ No stage definitions provided in config")
            return {}

        # Extract class indices from stage definitions
        stage_class_indices = []
        for stage_def in self.all_stage_definitions:
            if isinstance(stage_def, dict) and 'class_indices' in stage_def:
                stage_class_indices.append(stage_def['class_indices'])
            else:
                # Fallback for list format
                stage_class_indices.append(stage_def)

        if stage_id - 2 < 0 or stage_id - 2 >= len(stage_class_indices):
            print(f"⚠️ Invalid stage_id {stage_id} for {len(stage_class_indices)} stages")
            return {}

        previous_stage_classes = stage_class_indices[stage_id - 2]  # Previous stage classes
        confidence_threshold = self.pseudo_label_config.get('confidence_threshold', 0.3)

        print(f"   Previous stage classes: {previous_stage_classes}")
        print(f"   Confidence threshold: {confidence_threshold}")

        # Import the fixed inference function
        from mmdet3d.apis import inference_detector

        # Generate using the fixed coordinate alignment approach
        pseudo_labels = {}
        processed = 0
        for i, data_info in enumerate(self.data_infos):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(self.data_infos)}")

            scene_id = data_info['point_cloud']['lidar_idx']
            if not getattr(self, 'data_root', None):
                continue
            points_file = os.path.join(str(self.data_root), 'points', f'{scene_id}.bin')

            if not os.path.exists(points_file):
                continue

            # Run inference with standard inference (for now)
            with torch.no_grad():
                try:
                    results = inference_detector(model, points_file)

                    if isinstance(results, tuple) and len(results) > 0:
                        predictions_list = results[0]
                        if isinstance(predictions_list, list) and len(predictions_list) > 0:
                            predictions = predictions_list[0]

                            # Filter predictions
                            filtered = self._filter_predictions_simple(
                                predictions,
                                previous_stage_classes,
                                confidence_threshold,
                                axis_align_matrix=data_info['annos'].get('axis_align_matrix'),
                                assume_aligned=True,
                            )

                            if filtered is not None:
                                pseudo_labels[scene_id] = filtered
                                processed += 1

                except Exception as e:
                    print(f"❌ Failed inference for {scene_id}: {e}")
                    continue

        print(f"✅ Generated pseudo labels for {processed} scenes")
        return pseudo_labels

    def generate_pseudo_labels_for_training_scenes(self, model, device='cuda'):
        """
        Generate pseudo labels on-the-fly during training.

        NOTE: MARKED FOR FUTURE INVESTIGATION
        This on-the-fly approach is kept for research purposes but
        pre-generated pseudo labels are the standard approach for:
        - 3-5x faster training
        - Reproducible results
        - Lower memory usage during training

        Future investigation may explore adaptive on-the-fly generation
        or hybrid approaches, but current standard is pre-generation.

        Generate pseudo labels for all training scenes using simplified approach.
        """
        import pickle

        try:
            print(
                "⚠️ On-the-fly pseudo generation is EXPERIMENTAL for ScanNet. "
                "Pre-generated pseudo labels remain the production path."
            )
            stage_id = self.stage_definition.get('stage_id', 1)
            if stage_id <= 1:
                print("⚠️ No previous stage for pseudo labels")
                return {}
            if not getattr(self, 'data_root', None):
                print("⚠️ Cannot generate pseudo labels: data_root is missing.")
                return {}

            model.eval()
            pseudo_labels = self._generate_pseudo_labels_with_model(model)
            print(f"   ✅ Generated pseudo labels for {len(pseudo_labels)} scenes")

            # Save using unified paths when available.
            output_file = None
            if self.paths:
                output_file = self.paths.pseudo_label_file(stage_id)
            elif self.work_dir:
                pseudo_dir = os.path.join(self.work_dir, 'pseudo_labels')
                os.makedirs(pseudo_dir, exist_ok=True)
                output_file = os.path.join(
                    pseudo_dir, f'stage_{stage_id}_pseudo_labels.pkl'
                )

            if output_file:
                with open(output_file, 'wb') as f:
                    pickle.dump(pseudo_labels, f)
                print(f"   💾 Saved {len(pseudo_labels)} pseudo labels to {output_file}")
            else:
                print("   ⚠️ No path management available, cannot save pseudo labels")

            return pseudo_labels
        except Exception as e:
            print(f"❌ Pseudo label generation failed with exception: {e}")
            return {}

    def _filter_predictions_simple(self, predictions, target_classes,
                                   confidence_threshold=0.3,
                                   axis_align_matrix=None,
                                   assume_aligned=True):
        """Filter predictions by class and confidence - converts to NYU40 IDs.

        This method now converts model class indices to NYU40 IDs for storage,
        ensuring compatibility across different incremental learning stages.
        """
        if not predictions or 'scores_3d' not in predictions:
            return None

        # Get predictions
        scores = predictions['scores_3d'].cpu().numpy()
        labels = predictions['labels_3d'].cpu().numpy()  # These are model class indices

        # Handle different bounding box formats
        boxes_3d = predictions['boxes_3d']
        if hasattr(boxes_3d, 'tensor'):
            boxes = boxes_3d.tensor.cpu().numpy()
        elif hasattr(boxes_3d, 'cpu'):
            boxes = boxes_3d.cpu().numpy()
        else:
            boxes = np.array(boxes_3d)

        # Filter by target classes and confidence
        class_mask = np.isin(labels, target_classes)
        conf_mask = scores >= confidence_threshold
        final_mask = class_mask & conf_mask

        if not final_mask.any():
            return None

        # Convert model indices to NYU40 IDs for storage
        filtered_labels = labels[final_mask]
        nyu40_labels = np.zeros_like(filtered_labels, dtype=np.int64)

        # Get model_idx_to_nyu40 mapping
        if hasattr(self, 'mappings') and self.mappings:
            model_to_nyu40 = self.mappings.get('model_idx_to_nyu40', {})
        else:
            # Fallback: build mapping from parent class if available
            model_to_nyu40 = {}
            if hasattr(self, 'model_id_to_nyu40'):
                model_to_nyu40 = self.model_id_to_nyu40
            else:
                # Build from VALID_CLASS_IDS if available
                if hasattr(self, 'VALID_CLASS_IDS') and hasattr(self, 'CLASSES'):
                    for idx, cls_name in enumerate(self.CLASSES[:35]):
                        # This is a simplified fallback - proper mapping should come from incremental_mappings
                        model_to_nyu40[idx] = self.VALID_CLASS_IDS[idx] if idx < len(self.VALID_CLASS_IDS) else idx + 1

        # Convert each label
        for i, model_idx in enumerate(filtered_labels):
            if model_idx in model_to_nyu40:
                nyu40_labels[i] = model_to_nyu40[model_idx]
            else:
                print(f"Warning: No NYU40 mapping for model index {model_idx}, using {model_idx + 1}")
                nyu40_labels[i] = model_idx + 1  # Emergency fallback

        canonical_boxes = canonicalize_bottom_center_boxes(
            boxes[final_mask],
            axis_align_matrix=axis_align_matrix,
            assume_aligned=assume_aligned,
        )

        record = build_canonical_pseudo_record(
            canonical_boxes,
            nyu40_labels,
            scores[final_mask],
        )
        record['count'] = record['num_detections']
        record['original_model_indices'] = filtered_labels  # Debugging aid
        return record

    def generate_pseudo_labels_for_replay(self, model, device='cuda'):
        """Legacy method name for backward compatibility."""
        return self.generate_pseudo_labels_for_training_scenes(model, device)

    def apply_pseudo_labels_to_scene(self, scene_info, scene_id):
        """Apply cached pseudo labels to a specific scene.

        Args:
            scene_info: Scene data info dictionary
            scene_id: Scene identifier

        Returns:
            Updated scene info with pseudo labels integrated
        """
        if not self.use_pseudo_labels or scene_id not in self.cached_pseudo_labels:
            return scene_info

        pseudo_data = self.cached_pseudo_labels[scene_id]

        # Create a copy to avoid modifying original
        enhanced_scene = copy.deepcopy(scene_info)

        # Get existing annotations
        existing_boxes = enhanced_scene['annos'].get('gt_boxes_upright_depth', [])
        existing_labels = enhanced_scene['annos'].get('class', [])
        existing_count = enhanced_scene['annos'].get('gt_num', 0)

        # Convert to lists if they're numpy arrays
        if isinstance(existing_boxes, np.ndarray):
            existing_boxes = existing_boxes.tolist()
        if isinstance(existing_labels, np.ndarray):
            existing_labels = existing_labels.tolist()

        # Add pseudo labels
        pseudo_boxes = pseudo_data.get('boxes')
        pseudo_labels = pseudo_data.get('labels')
        if pseudo_boxes is None:
            pseudo_boxes = pseudo_data.get('gt_boxes_upright_depth')
        if pseudo_labels is None:
            pseudo_labels = pseudo_data.get('class')
        pseudo_scores = pseudo_data.get('scores') or pseudo_data.get('confidence_scores')
        axis_align_matrix = None
        if isinstance(scene_info, dict):
            annos = scene_info.get('annos')
            if isinstance(annos, dict):
                axis_align_matrix = annos.get('axis_align_matrix')

        if len(pseudo_boxes) > 0:
            # Combine existing and pseudo annotations
            pseudo_boxes_np = np.asarray(pseudo_boxes, dtype=np.float32)
            if pseudo_boxes_np.ndim != 2 or pseudo_boxes_np.shape[1] < 6:
                raise ValueError('Pseudo boxes must be (N,6)')
            need_canonical = (
                pseudo_data.get('center_type') != 'gravity'
                or pseudo_data.get('axis_aligned') is not True
                or pseudo_boxes_np.shape[1] > 6
            )
            if need_canonical:
                pseudo_boxes_np = canonicalize_bottom_center_boxes(
                    pseudo_boxes_np,
                    axis_align_matrix=axis_align_matrix,
                    assume_aligned=True,
                )
            elif pseudo_boxes_np.shape[1] > 6:
                pseudo_boxes_np = pseudo_boxes_np[:, :6]

            pseudo_labels_np = np.asarray(pseudo_labels, dtype=np.int64)
            all_boxes = existing_boxes + pseudo_boxes_np.tolist()
            all_labels = existing_labels + pseudo_labels_np.tolist()

            # Update scene annotations
            enhanced_scene['annos']['gt_boxes_upright_depth'] = np.array(all_boxes)
            enhanced_scene['annos']['class'] = np.array(all_labels)
            enhanced_scene['annos']['gt_num'] = len(all_boxes)

            # Add metadata about pseudo labels
            enhanced_scene['has_pseudo_labels'] = True
            enhanced_scene['original_gt_num'] = existing_count
            enhanced_scene['pseudo_gt_num'] = len(pseudo_boxes)
            threshold = pseudo_data.get('confidence_threshold', 0.7)
            if pseudo_scores is not None and len(pseudo_scores) > 0:
                threshold = float(np.min(pseudo_scores))
            enhanced_scene['pseudo_confidence_threshold'] = threshold

            if self.pseudo_label_generator and self.pseudo_label_generator.debug_mode:
                print(f"    Enhanced scene {scene_id}: {existing_count} original + {len(pseudo_boxes)} pseudo = {len(all_boxes)} total")

        return enhanced_scene

    # Legacy label mapping method removed - was dead code never called
    # Referenced legacy class ordering (cabinet, chair, table...) which is incorrect
    # Current system uses frequency-based ordering (chair, door, otherfurniture...)

    # Legacy _convert_labels_to_sequential method removed - was dead code that referenced
    # the removed old_to_new_gci mapping and was never called

    def get_classes_for_stage(self, stage_id):
        """Get sequential GCI class indices for a specific stage."""
        if stage_id == 1:
            return list(range(0, 7))
        elif stage_id == 2:
            return list(range(0, 14))
        elif stage_id == 3:
            return list(range(0, 21))
        elif stage_id == 4:
            return list(range(0, 28))
        elif stage_id == 5:
            return list(range(0, 35))
        else:
            raise ValueError(f"Invalid stage_id: {stage_id}")

    def get_new_classes_for_stage(self, stage_id):
        """Get only the new classes introduced in a specific stage."""
        start = (stage_id - 1) * 7
        end = stage_id * 7
        return list(range(start, end))

    def prepare_test_data(self, index):
        """Prepare test data with annotations for evaluation on training set.

        This method overrides the parent's prepare_test_data to include ground truth
        annotations when evaluation_mode=True, allowing proper evaluation on the
        training dataset while maintaining compatibility with regular test mode.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict with annotations if in evaluation mode.
        """
        input_dict = self.get_data_info(index)

        # Include ground truth annotations when evaluating on training dataset
        if (self.test_mode and hasattr(self, 'evaluation_mode') and
            getattr(self, 'evaluation_mode', False)):
            # Get annotations like in training mode, but handle the incremental dataset structure
            try:
                annos = self.get_ann_info(index)
                input_dict['ann_info'] = annos
            except (KeyError, IndexError) as e:
                # Fallback: if get_ann_info fails due to filtering, create minimal annotations
                print(f"Warning: Could not get full annotations for scene {index}, using minimal annotations: {e}")
                info = self.data_infos[index]

                # Create empty annotations with required axis_align_matrix
                input_dict['ann_info'] = dict(
                    gt_bboxes_3d=np.zeros((0, 6), dtype=np.float32),
                    gt_labels_3d=np.zeros((0, ), dtype=np.int64),
                    axis_align_matrix=self._get_axis_align_matrix(info)
                )
        else:
            # Default behavior for regular test mode (validation dataset)
            input_dict['ann_info'] = dict(
                axis_align_matrix=self._get_axis_align_matrix(
                    self.data_infos[index]))

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 classwise=False,
                 by_epoch=True):
        """Incremental learning evaluation using incremental_indoor_eval.

        This method overrides ScanNetDataset.evaluate to use the incremental
        evaluation protocol, ensuring proper class ordering and filtering
        to only evaluate on classes seen so far in the incremental learning process.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
            iou_thr (list[float]): IoU thresholds. Default: (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
            classwise (bool, optional): Whether to print classwise evaluation.
                Default: False.

        Returns:
            dict[str, float]: Dict of evaluation results.
        """
        assert isinstance(results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(results[0], dict), f'Expect elements in results to be dict, got {type(results[0])}.'

        if logger:
            logger.info(f"🎯 INCREMENTAL EVALUATION: Using incremental_indoor_eval")
            logger.info(f"   Stage {self.stage_idx + 1}: Evaluating on {len(self.all_seen_classes)} seen classes")
            logger.info(f"   Seen classes: {self.all_seen_classes}")

        # Build ground truth annotations (accessing from 'annos' field)
        gt_annos = []
        for i in range(len(self.data_infos)):
            info = self.data_infos[i]
            if info['annos']['gt_num'] != 0:
                gt_annos.append({
                    'gt_boxes_upright_depth': info['annos']['gt_boxes_upright_depth'],
                    'gt_num': info['annos']['gt_num'],
                    'class': info['annos']['class']
                })
            else:
                gt_annos.append({'gt_num': 0})

        # Create corrected ground truth annotations with mapped class IDs.
        nyu40_to_model = self.mappings.get('nyu40_to_model_idx', {}) if hasattr(self, 'mappings') else {}
        if not nyu40_to_model and hasattr(self, 'nyu40_to_model_id'):
            nyu40_to_model = self.nyu40_to_model_id
        if not nyu40_to_model:
            raise RuntimeError(
                "IncrementalScanNetDataset.evaluate requires non-empty NYU40->model "
                "mapping (mappings['nyu40_to_model_idx'])."
            )

        gt_annos_corrected = []
        dropped_nyu40_counts = {}

        for anno in gt_annos:
            if anno['gt_num'] == 0:
                gt_annos_corrected.append(anno)
                continue

            gt_labels_nyu40 = np.asarray(anno['class'], dtype=np.int64).reshape(-1)
            gt_boxes = np.asarray(anno['gt_boxes_upright_depth'])
            if int(gt_boxes.shape[0]) != int(gt_labels_nyu40.shape[0]):
                raise RuntimeError(
                    "IncrementalScanNetDataset.evaluate GT shape mismatch: "
                    f"boxes={gt_boxes.shape[0]}, labels={gt_labels_nyu40.shape[0]}."
                )

            converted_labels = []
            valid_boxes = []
            for idx, nyu40_id in enumerate(gt_labels_nyu40.tolist()):
                nyu_key = int(nyu40_id)
                if nyu_key not in nyu40_to_model:
                    dropped_nyu40_counts[nyu_key] = (
                        dropped_nyu40_counts.get(nyu_key, 0) + 1
                    )
                    continue
                model_idx = int(nyu40_to_model[nyu_key])
                converted_labels.append(model_idx)
                valid_boxes.append(gt_boxes[int(idx)])

            if converted_labels:
                gt_annos_corrected.append({
                    'gt_boxes_upright_depth': np.array(valid_boxes),
                    'gt_num': len(converted_labels),
                    'class': np.array(converted_labels)
                })
            else:
                gt_annos_corrected.append({'gt_num': 0})

        if logger and dropped_nyu40_counts:
            dropped_ids = sorted(dropped_nyu40_counts.keys())
            dropped_total = int(sum(dropped_nyu40_counts.values()))
            suffix = '' if len(dropped_ids) <= 10 else ' (truncated)'
            logger.warning(
                "IncrementalScanNetDataset.evaluate dropped %d GT boxes with "
                "unmapped NYU40 IDs: %s%s",
                dropped_total,
                dropped_ids[:10],
                suffix,
            )

        # Get all class names for mapping
        all_class_names = [self.model_id_to_name.get(i, f'class_{i}') for i in range(35)]

        # Build class meta (GCI -> NYU40, Stage, Name) for enhanced evaluation table
        class_meta = {}
        try:
            model_idx_to_nyu40 = {}
            if hasattr(self, 'mappings') and self.mappings:
                model_idx_to_nyu40 = self.mappings.get('model_idx_to_nyu40', {})
            # Stage map: GCI -> stage_id
            class_to_stage = {}
            if hasattr(self, 'all_stage_definitions') and self.all_stage_definitions:
                for sd in self.all_stage_definitions:
                    sid = int(sd.get('stage_id', 0))
                    for gci in sd.get('class_indices', []):
                        class_to_stage[int(gci)] = sid
            for gci in self.all_seen_classes:
                class_meta[int(gci)] = {
                    'nyu40': int(model_idx_to_nyu40.get(int(gci), -1)) if model_idx_to_nyu40 else '',
                    'stage': int(class_to_stage.get(int(gci), -1)) if class_to_stage else '',
                    'name': self.model_id_to_name.get(int(gci), f'class_{int(gci)}')
                }
        except Exception:
            class_meta = {}

        # Use incremental evaluation with proper class ordering
        ret_dict = incremental_indoor_eval(
            gt_annos_corrected,
            results,
            iou_thr,
            seen_classes=self.all_seen_classes,
            class_names=all_class_names,
            stage_idx=self.stage_idx,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            class_meta=class_meta
        )

        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict
