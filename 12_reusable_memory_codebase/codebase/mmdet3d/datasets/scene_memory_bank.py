"""
Scene-Based Memory Bank for Incremental Learning

This memory bank stores references to complete scenes (not individual objects)
from previous stages to prevent catastrophic forgetting. Unlike the object-based
approach, this preserves spatial context and object relationships.

CRITICAL DESIGN PRINCIPLES:
1. NO point clouds are saved - only scene metadata and references
2. Labels are filtered to match the stage when scenes were saved
3. Same scene can have multiple snapshots from different stages
4. Handles scene duplication (replay vs natural occurrence)
"""

import numpy as np
import copy
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import torch

from .memory_bank.compat_weighting import (
    compute_class_balance_weights,
    compute_class_balance_weights_v2,
    compute_comp_mean,
    compute_percentile_rank,
    compute_redundancy_penalty,
    ratio_targeted_swap_update,
    stage1_greedy_fill,
)
from mmdet3d.utils.ld_strategy_config import (
    LD_DESIGN1_STRATEGY,
    LD_DESIGN2_STRATEGY,
    validate_scene_memory_ld_strategy_config,
)


class SceneMemoryBank:
    """Scene-based memory bank that stores scene references for replay.
    
    Key differences from object-based approach:
    - Stores complete scene references, not individual objects
    - Preserves spatial context and object relationships
    - No point cloud storage - only metadata
    - Handles label filtering for correct incremental learning
    """
    
    def __init__(self,
                 # Global budget approach (NEW)
                 memory_budget_ratio: float = 0.1,
                 max_memory_scenes: Optional[int] = None,
                 total_training_scenes: int = 1201,
                 # Deterministic selection
                 random_seed: int = 0,
                 # Optional quota policy for memory composition
                 quota_strategy: Optional[str] = None,
                 stage_scene_counts: Optional[List[int]] = None,
                 # SUNRGBD: forgetness-based eviction (optional; explicitly enabled in config)
                 forgetness_eviction: Optional[Dict[str, Any]] = None,
                 # SUNRGBD: under-learning based insertion (optional; explicitly enabled in config)
                 underlearning_insertion: Optional[Dict[str, Any]] = None,
                 # Learning-dynamics scoring update (optional; explicitly enabled in config)
                 learning_dynamics_update: Optional[Dict[str, Any]] = None,
                 # Learning-dynamics Design-1 scoring update (optional; SUNRGBD only)
                 learning_dynamics_design1: Optional[Dict[str, Any]] = None,
                 # Learning-dynamics Design-2 scoring update (optional; SUNRGBD/ScanNet)
                 learning_dynamics_design2: Optional[Dict[str, Any]] = None,
                 
                 # Selection and replacement strategies
                 selection_strategy: str = 'balanced',
                 replacement_strategy: str = 'balanced_importance',
                 
                 # Importance scoring weights
                 class_balance_weight: float = 0.3,
                 diversity_weight: float = 0.3,
                 recency_weight: float = 0.2,
                 density_weight: float = 0.2,
                 importance_decay: float = 0.9,
                 
                 # Precomputed score settings (NEW)
                 score_files_dir: Optional[str] = None,
                 score_criteria: Optional[str] = None,
                 
                 # Legacy support (will be deprecated)
                 scenes_per_class: Optional[int] = None,
                 
                 # Other settings
                 dedup_strategy: str = 'keep_both',
                 # Enforce unique scene IDs in the active memory bank (no duplicate scenes).
                 # When True, the bank stores at most one snapshot per scene_id.
                 enforce_unique_scene_ids: bool = True,
                 min_objects_per_scene: int = 2,
                 prefer_diverse_scenes: bool = True,
                 debug_mode: bool = True,
                 
                 # Discovery system support
                 forced_scene_list: Optional[List[str]] = None,
                 
                 # Drop/difficulty-aware scoring (NEW)
                 metrics_dir: Optional[str] = None,
                 use_drop_weights: bool = True,
                 drop_alpha: float = 0.6,
                 drop_weight_strength: float = 0.3,
                 
                 # Current-stage consolidation weighting (NEW)
                 use_current_stage_weights: bool = True,
                 current_weight_strength: float = 0.25,
                 current_alpha: float = 0.0,
                 stage_class_map: Optional[Dict[int, List[int]]] = None,
                 enforce_current_quota: bool = True,
                 min_current_stage_quota: int = 10,

                 # Stage ratio alignment (NEW)
                 stage_ratio_counts: Optional[List[int]] = None,
                 stage_ratio_gamma: float = 1.0,

                 # Uncertainty / diversity-based selection (NEW)
                 uncertainty_conf_thresh: float = 0.15,
                 diversity_conf_thresh: Optional[float] = None,
                 iou_match_thr: float = 0.25,
                 undet_classes: str = 'old_and_current',
                 uncertainty_focus: str = 'high',
                 uncertainty_pool_split: float = 0.5,
                 diversity_beta: float = 0.2,
                 combined_alpha: float = 0.5,
                 combined_pre_screen_ratio: float = 0.5,
                 max_boxes_per_scene: Optional[int] = None):
        """
        Args:
            memory_budget_ratio: Fraction of total training scenes to keep (e.g., 0.1 = 10%)
            max_memory_scenes: Explicit maximum scenes (overrides memory_budget_ratio)
            total_training_scenes: Total scenes in training set (used for budget calculation)
            selection_strategy: How to select scenes ('balanced', 'diversity', 'random',
                'precomputed', 'forced', 'gt_object_count_desc')
            replacement_strategy: How to replace scenes when budget is full
            class_balance_weight: Weight for class balance in importance scoring (0-1)
            diversity_weight: Weight for scene diversity in importance scoring (0-1)
            recency_weight: Weight for stage recency in importance scoring (0-1)
            density_weight: Weight for object density in importance scoring (0-1)
            importance_decay: Decay factor for existing scene importance (0-1)
            score_files_dir: Directory containing precomputed score files (for 'precomputed' strategy)
            score_criteria: Scoring criteria folder name (any folder name in score_files_dir)
            scenes_per_class: DEPRECATED - Legacy per-class limit (use global budget instead)
            dedup_strategy: How to handle duplicates ('keep_both', 'prefer_replay', 'prefer_natural')
            min_objects_per_scene: Minimum objects required for scene selection
            prefer_diverse_scenes: Whether to prefer diverse scene types
            debug_mode: Enable extensive debug logging
            forced_scene_list: List of specific scene IDs to use (overrides selection strategy)
        """
        # Capture constructor semantics before normalization, for bank checkpoints.
        self._checkpoint_config = copy.deepcopy({
            key: value for key, value in locals().items() if key != 'self'})
        self.loaded_stage_id = None
        # Determine memory budget
        if max_memory_scenes is not None:
            self.memory_budget = max_memory_scenes
        else:
            self.memory_budget = int(total_training_scenes * memory_budget_ratio)
        
        # Global budget parameters
        self.memory_budget_ratio = memory_budget_ratio
        self.total_training_scenes = total_training_scenes

        # Deterministic selection controls
        self.random_seed = int(random_seed) if random_seed is not None else 0

        # Quota strategy (SUNRGBD baseline uses stage-ratio quotas)
        self.quota_strategy = str(quota_strategy) if quota_strategy is not None else None
        self.stage_scene_counts = {}
        if stage_scene_counts is not None:
            try:
                if isinstance(stage_scene_counts, dict):
                    for k, v in stage_scene_counts.items():
                        self.stage_scene_counts[int(k)] = int(v)
                else:
                    # List/tuple is interpreted as [N1, N2, ...] with stage_id starting at 1
                    for i, v in enumerate(list(stage_scene_counts), start=1):
                        self.stage_scene_counts[int(i)] = int(v)
            except Exception:
                self.stage_scene_counts = {}

        # Forgetness-based eviction controls (SUNRGBD incremental)
        self.forgetness_eviction = forgetness_eviction or {}
        self.forgetness_eviction_enabled = bool(self.forgetness_eviction.get('enabled', False))
        self.forgetness_score_mode = str(self.forgetness_eviction.get('score_mode', 'presence_sum'))
        self.forgetness_protect_new_stage = bool(self.forgetness_eviction.get('protect_new_stage', True))

        # Under-learning based insertion controls (SUNRGBD incremental)
        self.underlearning_insertion = underlearning_insertion or {}
        self.underlearning_insertion_enabled = bool(self.underlearning_insertion.get('enabled', False))
        self.underlearning_score_mode = str(self.underlearning_insertion.get('score_mode', 'object_count_sum'))
        if self.underlearning_insertion_enabled and self.underlearning_score_mode not in (
                'object_count_sum',
                'presence_sum'):
            raise ValueError(
                "underlearning_insertion.score_mode must be one of "
                "['object_count_sum', 'presence_sum'], "
                f"but got '{self.underlearning_score_mode}'."
            )
        # Store eval settings for rank0 train-eval (implemented in tools/train_incremental_scene.py).
        self.underlearning_ap_iou_thr = float(self.underlearning_insertion.get('ap_iou_thr', 0.25))
        self.underlearning_eval_max_scenes = self.underlearning_insertion.get('eval_max_scenes', None)
        self.underlearning_eval_seed_offset = int(self.underlearning_insertion.get('eval_seed_offset', 11000))

        # Learning-dynamics scoring update controls (per-seat trajectories).
        # NOTE: the strategy is enabled via `selection_strategy='learning_dynamics'`
        # (option B). `learning_dynamics_update.enabled` is intentionally unsupported
        # to avoid confusing parallel knobs.
        self.learning_dynamics_update = learning_dynamics_update or {}
        if isinstance(self.learning_dynamics_update, dict):
            if 'enabled' in self.learning_dynamics_update:
                raise ValueError(
                    "learning_dynamics_update.enabled is not supported. "
                    "Enable learning-dynamics updates via "
                    "scene_memory_config.selection_strategy='learning_dynamics'."
                )
            if 'add_count' in self.learning_dynamics_update:
                raise ValueError(
                    "learning_dynamics_update.add_count is not supported. "
                    "This repo enforces stage-ratio quotas when using "
                    "selection_strategy='learning_dynamics'."
                )
            if 'protect_new_stage' in self.learning_dynamics_update:
                raise ValueError(
                    "learning_dynamics_update.protect_new_stage is not supported. "
                    "This repo enforces stage-ratio quotas per stage."
                )

        # Strict strategy/block validation for Design-1 vs Design-2.
        ld_design_cfg_meta = validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=selection_strategy,
                learning_dynamics_design1=learning_dynamics_design1,
                learning_dynamics_design2=learning_dynamics_design2,
            ),
            context='scene_memory_config',
        )
        self.selection_strategy = str(selection_strategy).strip().lower()
        self.learning_dynamics_design1 = learning_dynamics_design1 or {}
        self.learning_dynamics_design2 = learning_dynamics_design2 or {}
        self._ld_design_block_key = ld_design_cfg_meta.get('active_ld_block_key', None)
        self._ld_design_cfg = ld_design_cfg_meta.get('active_ld_config', {}) or {}

        # Learning-dynamics design controls (active block: design1 or design2).
        ld_cfg = dict(self._ld_design_cfg)
        if self._ld_design_block_key in ('learning_dynamics_design1', 'learning_dynamics_design2'):
            if 'q_metric' not in ld_cfg:
                raise ValueError(
                    f"{self._ld_design_block_key}.q_metric must be explicitly configured "
                    "(no implicit fallback)."
                )
            self.learning_dynamics_design1_q_metric = str(ld_cfg.get('q_metric')).strip().lower()
        else:
            # Legacy non-design LD path keeps fixed F1 q-definition.
            self.learning_dynamics_design1_q_metric = 'f1'
        if self.learning_dynamics_design1_q_metric not in ('f1', 'recall'):
            raise ValueError(
                f"{self._ld_design_block_key or 'learning_dynamics_design1'}.q_metric "
                "must be one of ['f1', 'recall'], "
                f"but got '{self.learning_dynamics_design1_q_metric}'."
            )
        self.learning_dynamics_design1_min_add_lower_bound = int(
            ld_cfg.get('min_add_lower_bound', 1)
        )
        if self.learning_dynamics_design1_min_add_lower_bound < 0:
            raise ValueError(
                f"{self._ld_design_block_key or 'learning_dynamics_design1'}.min_add_lower_bound "
                "must be >= 0, "
                f"but got {self.learning_dynamics_design1_min_add_lower_bound}."
            )
        self.learning_dynamics_design1_use_compatibility_kernel = bool(
            ld_cfg.get('use_compatibility_kernel', True)
        )
        self.learning_dynamics_design1_use_class_balance = bool(
            ld_cfg.get('use_class_balance', True)
        )
        self.learning_dynamics_design1_compatibility_weight = float(
            ld_cfg.get('compatibility_weight', 1.0)
        )
        if (not np.isfinite(self.learning_dynamics_design1_compatibility_weight)
                or self.learning_dynamics_design1_compatibility_weight < 0.0):
            raise ValueError(
                f"{self._ld_design_block_key or 'learning_dynamics_design1'}.compatibility_weight "
                "must be a finite value >= 0, "
                f"but got {self.learning_dynamics_design1_compatibility_weight}."
            )
        self.learning_dynamics_design1_supply_scaling_mode = str(
            ld_cfg.get('supply_scaling_mode', 'raw')
        ).strip().lower()
        if self.learning_dynamics_design1_supply_scaling_mode not in (
                'raw', 'cap', 'log1p', 'cap_log1p'):
            raise ValueError(
                f"{self._ld_design_block_key or 'learning_dynamics_design1'}.supply_scaling_mode "
                "must be one of "
                "['raw', 'cap', 'log1p', 'cap_log1p'], "
                f"but got '{self.learning_dynamics_design1_supply_scaling_mode}'."
            )
        self.learning_dynamics_design1_supply_cap = None
        supply_cap_cfg = ld_cfg.get('supply_cap', None)
        if self.learning_dynamics_design1_supply_scaling_mode in ('cap', 'cap_log1p'):
            if supply_cap_cfg is None:
                raise ValueError(
                    f"{self._ld_design_block_key or 'learning_dynamics_design1'}.supply_cap "
                    "is required when "
                    f"supply_scaling_mode='{self.learning_dynamics_design1_supply_scaling_mode}'."
                )
            try:
                self.learning_dynamics_design1_supply_cap = int(supply_cap_cfg)
            except Exception as e:
                raise ValueError(
                    f"{self._ld_design_block_key or 'learning_dynamics_design1'}.supply_cap "
                    "must be an integer > 0, "
                    f"but got {supply_cap_cfg!r}."
                ) from e
            if self.learning_dynamics_design1_supply_cap <= 0:
                raise ValueError(
                    f"{self._ld_design_block_key or 'learning_dynamics_design1'}.supply_cap "
                    "must be > 0, "
                    f"but got {self.learning_dynamics_design1_supply_cap}."
                )
        elif supply_cap_cfg is not None:
            raise ValueError(
                f"{self._ld_design_block_key or 'learning_dynamics_design1'}.supply_cap "
                "is only supported when "
                "supply_scaling_mode in ['cap', 'cap_log1p']."
            )
        self.learning_dynamics_design1_force_accept_until_lower_bound = bool(
            ld_cfg.get(
                'force_accept_until_lower_bound',
                True,
            )
        )
        self.learning_dynamics_design1_allow_missing_seat_terms = bool(
            ld_cfg.get('allow_missing_seat_terms', False)
        )

        # Explicit design mode from strategy (no design_version config fallback).
        self.learning_dynamics_design_version = (
            2 if self.selection_strategy == LD_DESIGN2_STRATEGY else 1
        )
        # w_max cap for the stronger 1/(1+count) balance formula (design 2 only).
        self.learning_dynamics_design2_w_max = float(
            ld_cfg.get('w_max', 10.0)
        )
        # Redundancy penalty lambda (design 2): score = (1-lam)*unary - lam*redundancy.
        # lambda=0 => pure unary (useful ablation).  Default 0.3 gives moderate
        # diversity pressure without overwhelming unary quality signal.
        self.learning_dynamics_design2_redundancy_lambda = float(
            ld_cfg.get('redundancy_lambda', 0.3)
        )
        if not (0.0 <= self.learning_dynamics_design2_redundancy_lambda <= 1.0):
            raise ValueError(
                f"{self._ld_design_block_key or 'learning_dynamics_design1'}.redundancy_lambda "
                "must be in [0,1], "
                f"but got {self.learning_dynamics_design2_redundancy_lambda}."
            )
        # Top-k for redundancy penalty (how many bank neighbours to average).
        self.learning_dynamics_design2_redundancy_topk = int(
            ld_cfg.get('redundancy_topk', 5)
        )
        # Minimum per-class scene quota (design 2 only).  If any class has
        # fewer than this many scenes in the bank, candidates containing that
        # class receive a priority boost.
        self.learning_dynamics_design2_min_class_quota = int(
            ld_cfg.get('min_class_quota', 5)
        )

        # Stage-ratio quota strategy is currently only implemented for the SUNRGBD
        # baselines ('random') and learning-dynamics ('learning_dynamics').
        if str(quota_strategy) == 'stage_ratio':
            if str(selection_strategy) not in (
                    'random',
                    'learning_dynamics',
                    'learning_dynamics_design1',
                    'learning_dynamics_design2'):
                raise ValueError(
                    "quota_strategy='stage_ratio' currently supports "
                    "selection_strategy in ['random', 'learning_dynamics', "
                    "'learning_dynamics_design1', 'learning_dynamics_design2'], "
                    f"but got '{selection_strategy}'."
                )

        # Learning-dynamics selection requires stage-ratio quotas and is not
        # compatible with other SUNRGBD update policies in this repo.
        if str(selection_strategy) == 'learning_dynamics':
            if str(quota_strategy) != 'stage_ratio':
                raise ValueError(
                    "selection_strategy='learning_dynamics' requires "
                    "quota_strategy='stage_ratio'."
                )
            if self.forgetness_eviction_enabled or self.underlearning_insertion_enabled:
                raise ValueError(
                    "selection_strategy='learning_dynamics' is not compatible with "
                    "forgetness_eviction/underlearning_insertion in this repo. "
                    "Disable those knobs to avoid ambiguous update policies."
                )

        # Learning-dynamics Design-1 selection also requires stage-ratio quotas and
        # remains mutually exclusive with legacy SUNRGBD update policies.
        if str(selection_strategy) == 'learning_dynamics_design1':
            if str(quota_strategy) != 'stage_ratio':
                raise ValueError(
                    "selection_strategy='learning_dynamics_design1' requires "
                    "quota_strategy='stage_ratio'."
                )
            if self.forgetness_eviction_enabled or self.underlearning_insertion_enabled:
                raise ValueError(
                    "selection_strategy='learning_dynamics_design1' is not compatible with "
                    "forgetness_eviction/underlearning_insertion in this repo. "
                    "Disable those knobs to avoid ambiguous update policies."
                )
        if str(selection_strategy) == 'learning_dynamics_design2':
            if str(quota_strategy) != 'stage_ratio':
                raise ValueError(
                    "selection_strategy='learning_dynamics_design2' requires "
                    "quota_strategy='stage_ratio'."
                )
            if self.forgetness_eviction_enabled or self.underlearning_insertion_enabled:
                raise ValueError(
                    "selection_strategy='learning_dynamics_design2' is not compatible with "
                    "forgetness_eviction/underlearning_insertion in this repo. "
                    "Disable those knobs to avoid ambiguous update policies."
                )
        
        # Strategy parameters
        self.selection_strategy = str(selection_strategy).strip().lower()
        self.replacement_strategy = replacement_strategy
        
        # Importance scoring weights
        self.class_balance_weight = class_balance_weight
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
        self.density_weight = density_weight
        self.importance_decay = importance_decay
        
        # Legacy support
        self.scenes_per_class = scenes_per_class
        self.use_legacy_mode = scenes_per_class is not None
        
        # Other settings
        self.dedup_strategy = dedup_strategy
        self.enforce_unique_scene_ids = bool(enforce_unique_scene_ids)
        self.min_objects_per_scene = min_objects_per_scene
        self.prefer_diverse_scenes = prefer_diverse_scenes
        self.debug_mode = debug_mode
        
        # Discovery system support
        self.forced_scene_list = forced_scene_list
        
        # Metrics for drop/difficulty-aware scoring
        self.metrics_dir = metrics_dir
        self.use_drop_weights = use_drop_weights
        self.drop_alpha = float(drop_alpha)
        self.drop_weight_strength = float(drop_weight_strength)
        self._class_weight_cache = {}

        # Current-stage consolidation controls
        self.use_current_stage_weights = bool(use_current_stage_weights)
        self.current_weight_strength = float(current_weight_strength)
        self.current_alpha = float(current_alpha)
        self.stage_class_map = stage_class_map or {}
        self.enforce_current_quota = bool(enforce_current_quota)
        self.min_current_stage_quota = int(min_current_stage_quota)
        
        # Stage ratio alignment weights
        self.stage_ratio_counts = stage_ratio_counts or []
        self.stage_ratio_gamma = float(stage_ratio_gamma)
        self._stage_ratio_weights = {}
        if self.stage_ratio_counts:
            try:
                import numpy as _np
                counts = _np.array(self.stage_ratio_counts, dtype=float)
                counts[counts <= 0] = 1.0
                weights = _np.power(counts, self.stage_ratio_gamma)
                mean_w = weights.mean() if weights.size > 0 else 1.0
                if mean_w == 0:
                    mean_w = 1.0
                weights = weights / mean_w
                for i, w in enumerate(weights, start=1):
                    self._stage_ratio_weights[i] = float(w)
            except Exception:
                self._stage_ratio_weights = {}

        # Uncertainty / diversity configuration
        self.uncertainty_conf_thresh = float(uncertainty_conf_thresh)
        # If diversity threshold is not specified, reuse uncertainty threshold
        self.diversity_conf_thresh = (
            float(diversity_conf_thresh)
            if diversity_conf_thresh is not None else
            float(uncertainty_conf_thresh)
        )
        self.iou_match_thr = float(iou_match_thr)
        # Which classes to use for FN computation: 'old_only' or 'old_and_current'
        self.undet_classes = str(undet_classes)
        # How to interpret uncertainty scores during selection:
        # 'high' (prefer high S_unc), 'low' (prefer low S_unc), 'two_pool' (split)
        self.uncertainty_focus = str(uncertainty_focus)
        self.uncertainty_pool_split = float(uncertainty_pool_split)
        self.diversity_beta = float(diversity_beta)
        self.combined_alpha = float(combined_alpha)
        self.combined_pre_screen_ratio = float(combined_pre_screen_ratio)
        self.max_boxes_per_scene = (
            None if max_boxes_per_scene is None else int(max_boxes_per_scene)
        )
        
        # Precomputed scores configuration
        self.score_files_dir = score_files_dir
        self.score_criteria = score_criteria
        self.precomputed_scores = {}  # stage_id -> {scene_id: score}
        
        # CRITICAL VALIDATION: Ensure configuration consistency
        if score_criteria is not None:
            if selection_strategy != 'precomputed':
                raise ValueError(
                    f"score_criteria='{score_criteria}' requires "
                    f"selection_strategy='precomputed', but got '{selection_strategy}'. "
                    f"Please set selection_strategy='precomputed' in your config."
                )
        
        # Validate precomputed strategy has required settings
        if selection_strategy == 'precomputed':
            if score_criteria is None:
                raise ValueError(
                    "selection_strategy='precomputed' requires score_criteria to be set. "
                    "Please specify which score file to use (e.g., 'discovery_trial_49')."
                )
            self._load_precomputed_scores()
        
        # Legacy candidate cache: scene_id -> stage_id -> snapshot
        # NOTE: In global-budget mode we intentionally avoid populating this
        # cache to keep state strictly limited to active memory seats.
        self.scene_snapshots = {}
        
        # Global budget memory: scene_id -> {stages: {stage_id: data}, metadata}
        self.memory_scenes = {}  # Active scenes within budget
        # Structure: {
        #   'scene0515_00': {
        #     'stages': {1: {'snapshot': {...}, 'importance': 0.8}, 2: {...}},
        #     'latest_stage': 2,
        #     'total_importance': 1.5,
        #     'present_classes': {0, 2, 4, 5}
        #   }
        # }
        
        # Legacy compatibility - will be removed after refactoring
        self.scene_importance = {}
        
        # Legacy tracking: stage_id -> list of scene_ids
        self.stage_scenes = defaultdict(list)
        
        # Legacy tracking: class_id -> list of (scene_id, stage_id) tuples
        self.class_scenes = defaultdict(list)
        
        # Class distribution statistics
        self.class_distribution = defaultdict(int)  # class_id -> scene_count
        self.class_object_counts = defaultdict(int)  # class_id -> total_objects
        
        # Statistics
        self.total_scenes_stored = 0
        self.scenes_per_stage = defaultdict(int)
        self.label_filtering_stats = defaultdict(lambda: {'before': 0, 'after': 0})
        
        # Enhanced debugging statistics
        self.scene_selection_stats = defaultdict(dict)  # Track why scenes were selected/rejected
        self.deduplication_stats = defaultdict(dict)   # Track deduplication decisions
        self.class_filtering_stats = defaultdict(dict) # Track which classes were filtered per scene
        
        if self.debug_mode:
            print(f"SceneMemoryBank initialized:")
            if self.use_legacy_mode:
                print(f"   - Mode: Legacy (per-class limits)")
                print(f"   - Scenes per class: {scenes_per_class}")
            else:
                print(f"   - Mode: Global budget ({self.memory_budget} scenes total)")
                print(f"   - Budget ratio: {self.memory_budget_ratio * 100:.1f}% of {self.total_training_scenes} scenes")
            print(f"   - Selection strategy: {selection_strategy}")
            if selection_strategy == 'precomputed':
                print(f"   - Score files directory: {score_files_dir}")
                if score_criteria:
                    print(f"   - Score criteria: {score_criteria}")
            print(f"   - Dedup strategy: {dedup_strategy}")
            if self.selection_strategy in (
                    'uncertainty_only',
                    'diversity_only',
                    'uncertainty_diversity_combined'):
                print(f"   - Uncertainty/conf threshold: "
                      f"{self.uncertainty_conf_thresh}")
                print(f"   - Diversity/conf threshold: "
                      f"{self.diversity_conf_thresh}")
                print(f"   - IoU match threshold: {self.iou_match_thr}")
                print(f"   - Undetected-classes mode: {self.undet_classes}")
                print(f"   - Uncertainty focus: {self.uncertainty_focus}")
                print(f"   - Diversity beta: {self.diversity_beta}")
                print(f"   - Combined alpha: {self.combined_alpha}")
            if self._stage_ratio_weights:
                print(f"   - Stage ratio weights (mean=1.0): {self._stage_ratio_weights}")
            if self.use_drop_weights:
                print(f"   - Drop-aware scoring: alpha={self.drop_alpha}, strength={self.drop_weight_strength}")
            if self.use_current_stage_weights:
                print(f"   - Current-stage weighting: alpha={self.current_alpha}, strength={self.current_weight_strength}")
            if self.underlearning_insertion_enabled:
                print(
                    "   - Under-learning insertion: "
                    f"score_mode={self.underlearning_score_mode}, "
                    f"ap_iou_thr={self.underlearning_ap_iou_thr}"
                )
            if self.selection_strategy in (
                    LD_DESIGN1_STRATEGY,
                    LD_DESIGN2_STRATEGY):
                design_label = (
                    "Design-2"
                    if self.selection_strategy == LD_DESIGN2_STRATEGY
                    else "Design-1"
                )
                print(
                    f"   - Learning-dynamics {design_label}: "
                    f"q_metric={self.learning_dynamics_design1_q_metric}, "
                    f"min_add_lower_bound={self.learning_dynamics_design1_min_add_lower_bound}, "
                    f"use_compatibility_kernel={self.learning_dynamics_design1_use_compatibility_kernel}, "
                    f"use_class_balance={self.learning_dynamics_design1_use_class_balance}, "
                    f"compatibility_weight={self.learning_dynamics_design1_compatibility_weight}, "
                    f"supply_scaling_mode={self.learning_dynamics_design1_supply_scaling_mode}, "
                    f"supply_cap={self.learning_dynamics_design1_supply_cap}, "
                    f"force_accept_until_lower_bound={self.learning_dynamics_design1_force_accept_until_lower_bound}"
                )
            if self.enforce_current_quota and not self.use_legacy_mode:
                print(f"   - Stage-{1}..T min quota: {self.min_current_stage_quota} (per current build stage)")
            if self.quota_strategy:
                print(f"   - Quota strategy: {self.quota_strategy}")
                if self.stage_scene_counts:
                    print(f"   - Stage scene counts (N_s): {self.stage_scene_counts}")

    def _rng(self, stage_id: int, salt: int = 0) -> np.random.RandomState:
        """Deterministic RNG for selection/eviction."""
        sid = int(stage_id) if stage_id is not None else 0
        return np.random.RandomState(self.random_seed + 1000 * sid + int(salt))

    def _importance_placeholder_mode(self) -> bool:
        """Whether `importance` is a non-action placeholder in this run mode.

        In this repo, stage-ratio SUNRGBD updates (`random` / `learning_dynamics`)
        admit seats with constant `importance=1.0`; actions are driven by other
        scores or random draws.
        """
        return (
            (str(self.quota_strategy).strip().lower() == 'stage_ratio')
            and (str(self.selection_strategy).strip().lower() in (
                'random',
                'learning_dynamics',
                'learning_dynamics_design1',
                'learning_dynamics_design2',
            ))
        )

    def _importance_semantics(self) -> str:
        """Human-readable semantics for `importance` in state summaries."""
        if self._importance_placeholder_mode():
            return 'placeholder_constant_1.0_not_action_driving'
        return 'selection_score'

    def _compute_stage_ratio_quotas(self, upto_stage: int) -> Dict[int, int]:
        """Compute per-stage quotas Q_s(t) with deterministic largest-remainder rounding."""
        assert upto_stage >= 1
        if not self.stage_scene_counts:
            raise ValueError(
                "quota_strategy='stage_ratio' requires stage_scene_counts to be provided "
                "(e.g., {1: N1, 2: N2, ...} or [N1, N2, ...])."
            )

        stages = list(range(1, int(upto_stage) + 1))
        counts = [int(self.stage_scene_counts.get(s, 0)) for s in stages]
        denom = int(sum(counts))
        if denom <= 0:
            raise ValueError(
                f"Invalid stage_scene_counts for upto_stage={upto_stage}: {self.stage_scene_counts}"
            )

        b = int(self.memory_budget)
        raw = [b * float(c) / float(denom) for c in counts]
        floors = [int(np.floor(x)) for x in raw]
        remaining = b - int(sum(floors))
        fracs = [float(r - f) for r, f in zip(raw, floors)]

        # Assign remaining slots to largest fractional remainders (tie-break by stage id).
        order = sorted(
            range(len(stages)),
            key=lambda i: (-fracs[i], stages[i]),
        )
        for i in order[:max(0, remaining)]:
            floors[i] += 1

        quotas = {stages[i]: int(floors[i]) for i in range(len(stages))}
        # Safety: enforce exact sum
        assert sum(quotas.values()) == b, (quotas, b)
        return quotas

    def _recompute_class_distribution(self):
        """Recompute class_distribution/class_object_counts from current memory contents."""
        self.class_distribution.clear()
        self.class_object_counts.clear()
        for _, scene_data in self.memory_scenes.items():
            try:
                stages = scene_data.get('stages', {}) or {}
                for _, stage_data in stages.items():
                    snapshot = stage_data.get('snapshot', {}) or {}
                    present = snapshot.get('present_classes', []) or []
                    obj_counts = snapshot.get('object_counts', {}) or {}
                    for cid in present:
                        self.class_distribution[int(cid)] += 1
                        self.class_object_counts[int(cid)] += int(obj_counts.get(int(cid), 0))
            except Exception:
                continue

    def _apply_random_stage_ratio_update(self,
                                        stage_id: int,
                                        candidate_scenes: List[Dict],
                                        *,
                                        underlearning_class_ap: Optional[Dict[int, float]] = None,
                                        underlearning_new_classes: Optional[List[int]] = None
                                        ) -> Tuple[List[Dict], Dict[int, int], Dict[int, int]]:
        """Random selection with stage-ratio quotas (SUNRGBD baseline).

        Returns:
            selected_all: list of dicts with keys {scene_id, stage_id}
            quotas: target quotas Q_s(t)
            actual: actual counts per stage after update
        """
        stage_id = int(stage_id)
        duplicate_removed = self._enforce_unique_scene_ids_in_memory()
        if duplicate_removed > 0:
            print(
                f"WARNING: SceneMemoryBank removed {int(duplicate_removed)} duplicate "
                f"scene-stage entries to enforce unique scene IDs."
            )
        quotas = self._compute_stage_ratio_quotas(stage_id)

        # Stage→(scene_id, save_stage) pairs currently in memory.
        stage_to_pairs = defaultdict(list)
        for sid, sdata in self.memory_scenes.items():
            stages = (sdata.get('stages', {}) or {})
            for saved_stage in stages.keys():
                try:
                    st = int(saved_stage)
                except Exception:
                    continue
                stage_to_pairs[st].append((str(sid), int(st)))

        # Downsample previous stages to match new quotas.
        for s in range(1, stage_id):
            target = int(quotas.get(s, 0))
            pairs = sorted(stage_to_pairs.get(s, []))
            if len(pairs) <= target:
                continue
            rng = self._rng(stage_id, salt=10000 + s)
            keep_idx = rng.choice(len(pairs), size=target, replace=False).tolist() if target > 0 else []
            keep = set(pairs[i] for i in sorted(keep_idx))
            for sid, saved_stage in pairs:
                if (sid, saved_stage) not in keep:
                    self._remove_scene_from_memory(sid, stage_id=int(saved_stage))

        # Add current-stage scenes up to its quota (avoid duplicate *pairs*).
        current_stage_target = int(quotas.get(stage_id, 0))
        current_stage_existing = 0
        for sid, sdata in self.memory_scenes.items():
            try:
                if int(stage_id) in (sdata.get('stages', {}) or {}):
                    current_stage_existing += 1
            except Exception:
                continue
        need = max(0, current_stage_target - current_stage_existing)

        # Only consider candidates from this stage (defensive) and not already stored.
        existing_pairs = set()
        for sid, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    existing_pairs.add((str(sid), int(saved_stage)))
                except Exception:
                    continue
        existing_scene_ids = set(str(sid) for sid in self.memory_scenes.keys())
        if self.enforce_unique_scene_ids:
            stage_candidates = [
                c for c in candidate_scenes
                if int(c.get('stage_id', stage_id)) == stage_id
                and str(c.get('scene_id')) not in existing_scene_ids
            ]
        else:
            stage_candidates = [
                c for c in candidate_scenes
                if int(c.get('stage_id', stage_id)) == stage_id
                and (str(c.get('scene_id')), int(stage_id)) not in existing_pairs
            ]
        stage_candidates = sorted(stage_candidates, key=lambda c: str(c.get('scene_id')))
        # Deduplicate candidate list by scene_id (fill with next candidates if duplicates exist).
        unique_candidates = []
        used_candidate_ids = set()
        for cand in stage_candidates:
            sid = str(cand.get('scene_id'))
            if not sid or sid in used_candidate_ids:
                continue
            used_candidate_ids.add(sid)
            unique_candidates.append(cand)
        stage_candidates = unique_candidates

        if need > 0 and stage_candidates:
            take = min(int(need), len(stage_candidates))

            if self.underlearning_insertion_enabled:
                assert underlearning_class_ap is not None, (
                    "underlearning_insertion.enabled=True requires underlearning_class_ap "
                    "to be passed to _apply_random_stage_ratio_update()."
                )
                assert underlearning_new_classes is not None, (
                    "underlearning_insertion.enabled=True requires underlearning_new_classes "
                    "to be passed to _apply_random_stage_ratio_update()."
                )
                scored = []
                for cand in stage_candidates:
                    snap = cand.get('snapshot', {}) or {}
                    score = self._compute_underlearning_score_for_snapshot(
                        snap,
                        class_ap=underlearning_class_ap,
                        new_classes=underlearning_new_classes,
                    )
                    scored.append((float(score), str(cand.get('scene_id')), cand))
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                chosen_scored = scored[:take]
            else:
                rng = self._rng(stage_id, salt=20000)
                chosen_idx = rng.choice(len(stage_candidates), size=take, replace=False).tolist()
                chosen = [stage_candidates[i] for i in sorted(chosen_idx)]

            if self.underlearning_insertion_enabled:
                for score, _, cand in chosen_scored:
                    sid = str(cand['scene_id'])
                    snap = cand['snapshot']
                    self._add_scene_to_memory(sid, stage_id, snap, importance=1.0)
                    # Persist under-learning score only for chosen inserted seats.
                    try:
                        self.memory_scenes[sid]['stages'][int(stage_id)]['underlearning_score'] = float(score)
                    except Exception:
                        pass
            else:
                for cand in chosen:
                    sid = str(cand['scene_id'])
                    snap = cand['snapshot']
                    self._add_scene_to_memory(sid, stage_id, snap, importance=1.0)

        # Fill remaining budget slots with additional unique current-stage scenes.
        remaining = int(self.memory_budget) - int(self._count_scene_stage_pairs())
        if remaining > 0:
            existing_scene_ids = set(str(sid) for sid in self.memory_scenes.keys())
            if self.enforce_unique_scene_ids:
                extra_candidates = [
                    c for c in candidate_scenes
                    if int(c.get('stage_id', stage_id)) == stage_id
                    and str(c.get('scene_id')) not in existing_scene_ids
                ]
            else:
                extra_candidates = [
                    c for c in candidate_scenes
                    if int(c.get('stage_id', stage_id)) == stage_id
                    and (str(c.get('scene_id')), int(stage_id)) not in existing_pairs
                ]
            extra_candidates = sorted(extra_candidates, key=lambda c: str(c.get('scene_id')))
            unique_extra = []
            used_extra = set()
            for cand in extra_candidates:
                sid = str(cand.get('scene_id'))
                if not sid or sid in used_extra:
                    continue
                used_extra.add(sid)
                unique_extra.append(cand)
            extra_candidates = unique_extra

            if extra_candidates:
                take_extra = min(int(remaining), len(extra_candidates))
                if self.underlearning_insertion_enabled:
                    assert underlearning_class_ap is not None, (
                        "underlearning_insertion.enabled=True requires underlearning_class_ap "
                        "to be passed to _apply_random_stage_ratio_update()."
                    )
                    assert underlearning_new_classes is not None, (
                        "underlearning_insertion.enabled=True requires underlearning_new_classes "
                        "to be passed to _apply_random_stage_ratio_update()."
                    )
                    scored_extra = []
                    for cand in extra_candidates:
                        snap = cand.get('snapshot', {}) or {}
                        score = self._compute_underlearning_score_for_snapshot(
                            snap,
                            class_ap=underlearning_class_ap,
                            new_classes=underlearning_new_classes,
                        )
                        scored_extra.append((float(score), str(cand.get('scene_id')), cand))
                    scored_extra.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    chosen_extra = [c for _, _, c in scored_extra[:take_extra]]
                else:
                    rng_extra = self._rng(stage_id, salt=20001)
                    idx = rng_extra.choice(len(extra_candidates), size=take_extra, replace=False).tolist()
                    chosen_extra = [extra_candidates[int(i)] for i in sorted(idx)]

                for cand in chosen_extra:
                    sid = str(cand.get('scene_id'))
                    snap = cand.get('snapshot', {}) or {}
                    self._add_scene_to_memory(str(sid), int(stage_id), snap, importance=1.0)

        # Recompute distributions for debug/consistency (no weighting in this baseline).
        self._recompute_class_distribution()

        # Actual counts per stage in memory
        actual = defaultdict(int)
        for _, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    actual[int(saved_stage)] += 1
                except Exception:
                    continue

        selected_all = []
        for sid, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    selected_all.append(dict(scene_id=str(sid), save_stage=int(saved_stage)))
                except Exception:
                    continue
        selected_all = sorted(selected_all, key=lambda d: (str(d.get('scene_id')), int(d.get('save_stage', -1))))
        return selected_all, quotas, dict(actual)

    def _compute_forgetness_score_for_snapshot(self,
                                               snapshot: Dict[str, Any],
                                               class_drops: Dict[int, float]) -> float:
        """Compute a per-seat forgetness score from per-class AP drops.

        The default mode is presence-only: sum drops for present classes.
        """
        present = snapshot.get('present_classes', []) or []
        if self.forgetness_score_mode == 'object_count_sum':
            obj_counts = snapshot.get('object_counts', {}) or {}
            score = 0.0
            for cid in present:
                try:
                    cid_int = int(cid)
                except Exception:
                    continue
                cnt = obj_counts.get(cid_int, obj_counts.get(str(cid_int), 0))
                score += float(class_drops.get(cid_int, 0.0)) * float(cnt)
            return float(score)

        # presence_sum (default)
        score = 0.0
        for cid in present:
            try:
                cid_int = int(cid)
            except Exception:
                continue
            score += float(class_drops.get(cid_int, 0.0))
        return float(score)

    def _apply_random_stage_ratio_update_forgetness_eviction(
        self,
        stage_id: int,
        candidate_scenes: List[Dict[str, Any]],
        forgetness_class_drops: Dict[int, float],
        *,
        underlearning_class_ap: Optional[Dict[int, float]] = None,
        underlearning_new_classes: Optional[List[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, int], Dict[str, Any]]:
        """Random insertion + global forgetness-based eviction (SUNRGBD).

        - Insertion remains random and the number of new scenes added for the
          current stage uses the stage-ratio quota.
        - Eviction is global across the whole bank (no per-stage quotas).

        NOTE: This intentionally does NOT keep stage quotas during eviction.
        In the future, we may enforce stage-wise quotas when evicting.
        """
        stage_id = int(stage_id)
        duplicate_removed = self._enforce_unique_scene_ids_in_memory()
        if duplicate_removed > 0:
            print(
                f"WARNING: SceneMemoryBank removed {int(duplicate_removed)} duplicate "
                f"scene-stage entries to enforce unique scene IDs."
            )
        quotas = self._compute_stage_ratio_quotas(stage_id)

        # Stage-ratio quota decides how many *current stage* seats we aim to keep.
        current_stage_target = int(quotas.get(stage_id, 0))

        # Avoid duplicate *pairs*.
        existing_pairs = set()
        current_stage_existing = 0
        for sid, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    st = int(saved_stage)
                except Exception:
                    continue
                existing_pairs.add((str(sid), int(st)))
                if int(st) == int(stage_id):
                    current_stage_existing += 1

        # Only consider candidates from this stage and not already stored.
        existing_scene_ids = set(str(sid) for sid in self.memory_scenes.keys())
        if self.enforce_unique_scene_ids:
            stage_candidates = [
                c for c in candidate_scenes
                if int(c.get('stage_id', stage_id)) == stage_id
                and str(c.get('scene_id')) not in existing_scene_ids
            ]
        else:
            stage_candidates = [
                c for c in candidate_scenes
                if int(c.get('stage_id', stage_id)) == stage_id
                and (str(c.get('scene_id')), int(stage_id)) not in existing_pairs
            ]
        stage_candidates = sorted(stage_candidates, key=lambda c: str(c.get('scene_id')))
        # Deduplicate by scene_id (fill with next candidates if duplicates exist).
        unique_candidates = []
        used_candidate_ids = set()
        for cand in stage_candidates:
            sid = str(cand.get('scene_id'))
            if not sid or sid in used_candidate_ids:
                continue
            used_candidate_ids.add(sid)
            unique_candidates.append(cand)
        stage_candidates = unique_candidates

        need = max(0, current_stage_target - current_stage_existing)
        add_count = min(int(need), len(stage_candidates))

        # Compute how many old seats to evict BEFORE adding (budget is enforced on add).
        total_pairs = int(self._count_scene_stage_pairs())
        budget = int(self.memory_budget)
        evict_count = max(0, total_pairs + int(add_count) - budget)

        # Sanitize drop keys (JSON loads as str keys)
        class_drops: Dict[int, float] = {}
        for k, v in (forgetness_class_drops or {}).items():
            try:
                class_drops[int(k)] = float(v)
            except Exception:
                continue

        evicted = []
        if evict_count > 0:
            # Build eviction pool: all old seats (save_stage < stage_id). Protect
            # the current stage seats (newly added or pre-existing) by default.
            eviction_pool = []
            for sid, sdata in self.memory_scenes.items():
                stages = (sdata.get('stages', {}) or {})
                for saved_stage, stage_data in stages.items():
                    try:
                        st = int(saved_stage)
                    except Exception:
                        continue
                    # Default: only evict old-stage seats (save_stage < stage_id).
                    # If protect_new_stage=False, allow evicting current-stage seats too.
                    if self.forgetness_protect_new_stage:
                        if st >= stage_id:
                            continue
                    else:
                        if st > stage_id:
                            continue
                    snapshot = stage_data.get('snapshot', {}) or {}
                    eviction_pool.append((str(sid), int(st), snapshot))

            eviction_pool.sort(key=lambda x: (x[0], x[1]))

            if evict_count > len(eviction_pool):
                raise RuntimeError(
                    f"Forgetness eviction cannot remove {evict_count} seats with "
                    f"save_stage < {stage_id}; pool_size={len(eviction_pool)}. "
                    "This is unexpected; check memory budget/quota settings."
                )

            rng = self._rng(stage_id, salt=31000)
            scored = []
            for sid, st, snapshot in eviction_pool:
                score = self._compute_forgetness_score_for_snapshot(snapshot, class_drops)
                # Persist forgetness score only for seats that are actually scored
                # during eviction (old seats in the eviction pool).
                try:
                    self.memory_scenes[str(sid)]['stages'][int(st)]['forgetness_score'] = float(score)
                except Exception:
                    pass
                scored.append((float(score), float(rng.rand()), sid, int(st)))
            scored.sort(key=lambda x: (x[0], x[1]))

            for i in range(int(evict_count)):
                score, _, sid, st = scored[i]
                evicted.append(
                    dict(
                        scene_id=str(sid),
                        save_stage=int(st),
                        score=float(score),
                    )
                )
                self._remove_scene_from_memory(str(sid), stage_id=int(st))

        # Add current-stage scenes after eviction.
        added = []
        if add_count > 0 and stage_candidates:
            if self.underlearning_insertion_enabled:
                assert underlearning_class_ap is not None, (
                    "underlearning_insertion.enabled=True requires underlearning_class_ap "
                    "to be passed to _apply_random_stage_ratio_update_forgetness_eviction()."
                )
                assert underlearning_new_classes is not None, (
                    "underlearning_insertion.enabled=True requires underlearning_new_classes "
                    "to be passed to _apply_random_stage_ratio_update_forgetness_eviction()."
                )
                scored = []
                for cand in stage_candidates:
                    snap = cand.get('snapshot', {}) or {}
                    score = self._compute_underlearning_score_for_snapshot(
                        snap,
                        class_ap=underlearning_class_ap,
                        new_classes=underlearning_new_classes,
                    )
                    scored.append((float(score), str(cand.get('scene_id')), cand))
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                chosen_scored = scored[:int(add_count)]
            else:
                rng_add = self._rng(stage_id, salt=20000)
                chosen_idx = rng_add.choice(
                    len(stage_candidates),
                    size=int(add_count),
                    replace=False,
                ).tolist()
                chosen = [stage_candidates[int(i)] for i in sorted(chosen_idx)]

            if self.underlearning_insertion_enabled:
                for score, _, cand in chosen_scored:
                    sid = str(cand['scene_id'])
                    snap = cand['snapshot']
                    self._add_scene_to_memory(sid, stage_id, snap, importance=1.0)
                    # Persist under-learning score only for chosen inserted seats.
                    try:
                        self.memory_scenes[sid]['stages'][int(stage_id)]['underlearning_score'] = float(score)
                    except Exception:
                        pass
                    added.append(
                        dict(
                            scene_id=str(sid),
                            save_stage=int(stage_id),
                            underlearning_score=float(score),
                        )
                    )
            else:
                for cand in chosen:
                    sid = str(cand['scene_id'])
                    snap = cand['snapshot']
                    self._add_scene_to_memory(sid, stage_id, snap, importance=1.0)
                    added.append(dict(scene_id=str(sid), save_stage=int(stage_id)))

        # Fill remaining budget slots with additional unique current-stage scenes.
        remaining = int(self.memory_budget) - int(self._count_scene_stage_pairs())
        if remaining > 0:
            existing_scene_ids = set(str(sid) for sid in self.memory_scenes.keys())
            if self.enforce_unique_scene_ids:
                extra_candidates = [
                    c for c in candidate_scenes
                    if int(c.get('stage_id', stage_id)) == stage_id
                    and str(c.get('scene_id')) not in existing_scene_ids
                ]
            else:
                extra_candidates = [
                    c for c in candidate_scenes
                    if int(c.get('stage_id', stage_id)) == stage_id
                    and (str(c.get('scene_id')), int(stage_id)) not in existing_pairs
                ]
            extra_candidates = sorted(extra_candidates, key=lambda c: str(c.get('scene_id')))
            unique_extra = []
            used_extra = set()
            for cand in extra_candidates:
                sid = str(cand.get('scene_id'))
                if not sid or sid in used_extra:
                    continue
                used_extra.add(sid)
                unique_extra.append(cand)
            extra_candidates = unique_extra

            if extra_candidates:
                take_extra = min(int(remaining), len(extra_candidates))
                if self.underlearning_insertion_enabled:
                    assert underlearning_class_ap is not None, (
                        "underlearning_insertion.enabled=True requires underlearning_class_ap "
                        "to be passed to _apply_random_stage_ratio_update_forgetness_eviction()."
                    )
                    assert underlearning_new_classes is not None, (
                        "underlearning_insertion.enabled=True requires underlearning_new_classes "
                        "to be passed to _apply_random_stage_ratio_update_forgetness_eviction()."
                    )
                    scored_extra = []
                    for cand in extra_candidates:
                        snap = cand.get('snapshot', {}) or {}
                        score = self._compute_underlearning_score_for_snapshot(
                            snap,
                            class_ap=underlearning_class_ap,
                            new_classes=underlearning_new_classes,
                        )
                        scored_extra.append((float(score), str(cand.get('scene_id')), cand))
                    scored_extra.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    chosen_extra = [c for _, _, c in scored_extra[:take_extra]]
                else:
                    rng_extra = self._rng(stage_id, salt=20001)
                    idx = rng_extra.choice(len(extra_candidates), size=take_extra, replace=False).tolist()
                    chosen_extra = [extra_candidates[int(i)] for i in sorted(idx)]

                for cand in chosen_extra:
                    sid = str(cand.get('scene_id'))
                    snap = cand.get('snapshot', {}) or {}
                    self._add_scene_to_memory(str(sid), int(stage_id), snap, importance=1.0)
                    added.append(dict(scene_id=str(sid), save_stage=int(stage_id), reason='fill_budget_unique'))

        # Recompute distributions for debug/consistency.
        self._recompute_class_distribution()

        # Actual counts per stage in memory
        actual = defaultdict(int)
        for _, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    actual[int(saved_stage)] += 1
                except Exception:
                    continue

        selected_all = []
        for sid, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    selected_all.append(dict(scene_id=str(sid), save_stage=int(saved_stage)))
                except Exception:
                    continue
        selected_all = sorted(
            selected_all,
            key=lambda d: (str(d.get('scene_id')), int(d.get('save_stage', -1))),
        )

        report = dict(
            stage_id=int(stage_id),
            policy='forgetness_eviction',
            quota_note='stage_ratio quotas used for insertion only; eviction is global',
            current_stage_target=int(current_stage_target),
            current_stage_added=int(len(added)),
            evicted_count=int(len(evicted)),
            forgetness_score_mode=str(self.forgetness_score_mode),
            evicted_entries=evicted,
            added_entries=added,
            duplicates_filtered=int(duplicate_removed),
        )
        if self.underlearning_insertion_enabled:
            report['underlearning_insertion'] = dict(
                enabled=True,
                score_mode=str(self.underlearning_score_mode),
                ap_iou_thr=float(self.underlearning_ap_iou_thr),
                new_classes=[int(x) for x in (underlearning_new_classes or [])],
            )
        return selected_all, quotas, dict(actual), report

    def _apply_stage_ratio_update_learning_dynamics(
        self,
        stage_id: int,
        candidate_scenes: List[Dict[str, Any]],
        *,
        # Scores are keyed by (scene_id, save_stage) in a JSON-friendly shape:
        #   {scene_id: {save_stage: score}}
        forgetness_by_seat: Optional[Dict[str, Dict[int, float]]] = None,
        replay_priority_by_seat: Optional[Dict[str, Dict[int, float]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, int], Dict[str, Any]]:
        """Stage-ratio quota update (learning dynamics selection).

        Enforces stage-ratio quotas Q_s(t) for all stages s<=t:
          - Old stages (s<t): keep seats with highest forgetness F_s within each
            intro_stage.
          - Current stage (s=t): admit seats with highest replay priority U_s.

        Scores are expected to be keyed by (scene_id, save_stage).
        """
        stage_id = int(stage_id)
        duplicate_removed = self._enforce_unique_scene_ids_in_memory()
        if duplicate_removed > 0:
            print(
                f"WARNING: SceneMemoryBank removed {int(duplicate_removed)} duplicate "
                f"scene-stage entries to enforce unique scene IDs."
            )
        quotas = self._compute_stage_ratio_quotas(stage_id)
        quota_pruned = []
        added = []

        # --- Prune old stages to match quotas (within each stage) ---
        if stage_id > 1:
            assert isinstance(forgetness_by_seat, dict), (
                "selection_strategy='learning_dynamics' requires forgetness_by_seat "
                "for stage_id>1."
            )

        stage_to_pairs = defaultdict(list)  # save_stage -> [(scene_id, save_stage)]
        for sid, sdata in self.memory_scenes.items():
            stages = (sdata.get('stages', {}) or {})
            for saved_stage in stages.keys():
                try:
                    st = int(saved_stage)
                except Exception:
                    continue
                stage_to_pairs[int(st)].append((str(sid), int(st)))

        missing_forgetness = []
        invalid_forgetness = []
        for s in range(1, int(stage_id)):
            target = int(quotas.get(int(s), 0))
            pairs = sorted(stage_to_pairs.get(int(s), []))
            if len(pairs) <= target:
                continue

            rng = self._rng(stage_id, salt=41000 + int(s))
            scored = []
            for sid, st in pairs:
                if str(sid) not in (forgetness_by_seat or {}) or int(st) not in (forgetness_by_seat or {}).get(str(sid), {}):
                    missing_forgetness.append(dict(scene_id=str(sid), save_stage=int(st)))
                    score = 0.0
                else:
                    score = float((forgetness_by_seat or {}).get(str(sid), {}).get(int(st), 0.0))
                    if not np.isfinite(score):
                        invalid_forgetness.append(dict(scene_id=str(sid), save_stage=int(st), value=repr(score)))
                        score = 0.0

                # Persist forgetness score for seats considered under stage quotas.
                try:
                    self.memory_scenes[str(sid)]['stages'][int(st)]['learning_dynamics_forgetness'] = float(score)
                except Exception:
                    pass

                scored.append((float(score), float(rng.rand()), str(sid), int(st)))

            if missing_forgetness:
                raise RuntimeError(
                    "Learning-dynamics stage-ratio update is missing forgetness scores for "
                    f"{len(missing_forgetness)} old seats (example: {missing_forgetness[0]})."
                )

            evict_count = int(len(pairs) - target)
            scored.sort(key=lambda x: (x[0], x[1]))  # lowest forgetness first
            for i in range(int(evict_count)):
                score, _, sid, st = scored[i]
                quota_pruned.append(dict(
                    scene_id=str(sid),
                    save_stage=int(st),
                    forgetness=float(score),
                    reason='quota_prune_old_stage',
                ))
                self._remove_scene_from_memory(str(sid), stage_id=int(st))

        # --- Enforce current-stage quota (rare; defensive) ---
        current_target = int(quotas.get(int(stage_id), 0))
        current_pairs = []
        for sid, sdata in self.memory_scenes.items():
            if int(stage_id) in (sdata.get('stages', {}) or {}):
                current_pairs.append((str(sid), int(stage_id)))
        current_pairs = sorted(current_pairs)

        if len(current_pairs) > current_target:
            assert isinstance(replay_priority_by_seat, dict), (
                "selection_strategy='learning_dynamics' requires replay_priority_by_seat "
                "to enforce the current-stage quota."
            )
            rng = self._rng(stage_id, salt=42000)
            missing_rp = []
            invalid_rp = []
            scored = []
            for sid, st in current_pairs:
                if str(sid) not in (replay_priority_by_seat or {}) or int(st) not in (replay_priority_by_seat or {}).get(str(sid), {}):
                    missing_rp.append(dict(scene_id=str(sid), save_stage=int(st)))
                    score = 0.0
                else:
                    score = float((replay_priority_by_seat or {}).get(str(sid), {}).get(int(st), 0.0))
                    if not np.isfinite(score):
                        invalid_rp.append(dict(scene_id=str(sid), save_stage=int(st), value=repr(score)))
                        score = 0.0
                try:
                    self.memory_scenes[str(sid)]['stages'][int(st)]['learning_dynamics_replay_priority'] = float(score)
                except Exception:
                    pass
                scored.append((float(score), float(rng.rand()), str(sid), int(st)))
            if missing_rp:
                raise RuntimeError(
                    "Learning-dynamics stage-ratio update is missing replay-priority scores for "
                    f"{len(missing_rp)} current-stage seats (example: {missing_rp[0]})."
                )
            evict_count = int(len(current_pairs) - current_target)
            scored.sort(key=lambda x: (x[0], x[1]))  # lowest priority first
            for i in range(int(evict_count)):
                score, _, sid, st = scored[i]
                quota_pruned.append(dict(
                    scene_id=str(sid),
                    save_stage=int(st),
                    replay_priority=float(score),
                    reason='quota_prune_current_stage',
                ))
                self._remove_scene_from_memory(str(sid), stage_id=int(st))

        # --- Add current-stage seats up to its quota ---
        existing_pairs = set()
        current_existing = 0
        for sid, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    st = int(saved_stage)
                except Exception:
                    continue
                existing_pairs.add((str(sid), int(st)))
                if int(st) == int(stage_id):
                    current_existing += 1

        need = max(0, current_target - int(current_existing))
        existing_scene_ids = set(str(sid) for sid in self.memory_scenes.keys())
        if self.enforce_unique_scene_ids:
            stage_candidates = [
                c for c in candidate_scenes
                if int(c.get('stage_id', stage_id)) == stage_id
                and str(c.get('scene_id')) not in existing_scene_ids
            ]
        else:
            stage_candidates = [
                c for c in candidate_scenes
                if int(c.get('stage_id', stage_id)) == stage_id
                and (str(c.get('scene_id')), int(stage_id)) not in existing_pairs
            ]
        stage_candidates = sorted(stage_candidates, key=lambda c: str(c.get('scene_id')))
        # Deduplicate candidate list by scene_id (skip duplicates and take next).
        unique_candidates = []
        used_candidate_ids = set()
        for cand in stage_candidates:
            sid = str(cand.get('scene_id'))
            if not sid or sid in used_candidate_ids:
                continue
            used_candidate_ids.add(sid)
            unique_candidates.append(cand)
        stage_candidates = unique_candidates

        if need > 0 and stage_candidates:
            take = min(int(need), len(stage_candidates))

            assert isinstance(replay_priority_by_seat, dict) and replay_priority_by_seat, (
                "selection_strategy='learning_dynamics' requires replay_priority_by_seat "
                "for all stages (including stage_id=1)."
            )
            scored_candidates = []
            missing_rp = []
            invalid_rp = []
            rng = self._rng(stage_id, salt=43000)
            for cand in stage_candidates:
                sid = str(cand.get('scene_id'))
                if sid not in (replay_priority_by_seat or {}) or int(stage_id) not in (replay_priority_by_seat or {}).get(sid, {}):
                    missing_rp.append(dict(scene_id=str(sid), save_stage=int(stage_id)))
                    score = 0.0
                else:
                    score = float((replay_priority_by_seat or {}).get(str(sid), {}).get(int(stage_id), 0.0))
                    if not np.isfinite(score):
                        invalid_rp.append(dict(scene_id=str(sid), save_stage=int(stage_id), value=repr(score)))
                        score = 0.0
                scored_candidates.append((float(score), float(rng.rand()), str(sid), cand))
            if missing_rp:
                raise RuntimeError(
                    "Learning-dynamics stage-ratio update is missing replay-priority scores for "
                    f"{len(missing_rp)} candidate seats (example: {missing_rp[0]})."
                )
            if invalid_rp:
                raise RuntimeError(
                    "Learning-dynamics stage-ratio update received non-finite replay-priority scores for "
                    f"{len(invalid_rp)} candidate seats (example: {invalid_rp[0]})."
                )
            scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            chosen_scored = scored_candidates[:take]
            for score, _, sid, cand in chosen_scored:
                snap = cand.get('snapshot', {}) or {}
                self._add_scene_to_memory(str(sid), int(stage_id), snap, importance=1.0)
                try:
                    self.memory_scenes[str(sid)]['stages'][int(stage_id)]['learning_dynamics_replay_priority'] = float(score)
                except Exception:
                    pass
                added.append(dict(
                    scene_id=str(sid),
                    save_stage=int(stage_id),
                    replay_priority=float(score),
                    reason='stage1_admit_by_replay_priority' if int(stage_id) == 1 else 'admit_current_stage',
                ))

        # Fill remaining budget slots with additional unique current-stage scenes.
        remaining = int(self.memory_budget) - int(self._count_scene_stage_pairs())
        if remaining > 0:
            existing_scene_ids = set(str(sid) for sid in self.memory_scenes.keys())
            if self.enforce_unique_scene_ids:
                extra_candidates = [
                    c for c in candidate_scenes
                    if int(c.get('stage_id', stage_id)) == stage_id
                    and str(c.get('scene_id')) not in existing_scene_ids
                ]
            else:
                extra_candidates = [
                    c for c in candidate_scenes
                    if int(c.get('stage_id', stage_id)) == stage_id
                    and (str(c.get('scene_id')), int(stage_id)) not in existing_pairs
                ]
            extra_candidates = sorted(extra_candidates, key=lambda c: str(c.get('scene_id')))
            unique_extra = []
            used_extra = set()
            for cand in extra_candidates:
                sid = str(cand.get('scene_id'))
                if not sid or sid in used_extra:
                    continue
                used_extra.add(sid)
                unique_extra.append(cand)
            extra_candidates = unique_extra

            if extra_candidates:
                take_extra = min(int(remaining), len(extra_candidates))
                assert isinstance(replay_priority_by_seat, dict) and replay_priority_by_seat, (
                    "selection_strategy='learning_dynamics' requires replay_priority_by_seat "
                    "to fill remaining budget."
                )
                rng = self._rng(stage_id, salt=43001)
                scored_extra = []
                missing_rp = []
                invalid_rp = []
                for cand in extra_candidates:
                    sid = str(cand.get('scene_id'))
                    if sid not in (replay_priority_by_seat or {}) or int(stage_id) not in (replay_priority_by_seat or {}).get(sid, {}):
                        missing_rp.append(dict(scene_id=str(sid), save_stage=int(stage_id)))
                        score = 0.0
                    else:
                        score = float((replay_priority_by_seat or {}).get(str(sid), {}).get(int(stage_id), 0.0))
                        if not np.isfinite(score):
                            invalid_rp.append(dict(scene_id=str(sid), save_stage=int(stage_id), value=repr(score)))
                            score = 0.0
                    scored_extra.append((float(score), float(rng.rand()), str(sid), cand))
                if missing_rp:
                    raise RuntimeError(
                        "Learning-dynamics stage-ratio update is missing replay-priority scores for "
                        f"{len(missing_rp)} candidate seats (example: {missing_rp[0]})."
                    )
                if invalid_rp:
                    raise RuntimeError(
                        "Learning-dynamics stage-ratio update received non-finite replay-priority scores for "
                        f"{len(invalid_rp)} candidate seats (example: {invalid_rp[0]})."
                    )
                scored_extra.sort(key=lambda x: (x[0], x[1]), reverse=True)
                for score, _, sid, cand in scored_extra[:take_extra]:
                    snap = cand.get('snapshot', {}) or {}
                    self._add_scene_to_memory(str(sid), int(stage_id), snap, importance=1.0)
                    try:
                        self.memory_scenes[str(sid)]['stages'][int(stage_id)]['learning_dynamics_replay_priority'] = float(score)
                    except Exception:
                        pass
                    added.append(dict(
                        scene_id=str(sid),
                        save_stage=int(stage_id),
                        replay_priority=float(score),
                        reason='fill_budget_unique',
                    ))

        self._recompute_class_distribution()

        actual = defaultdict(int)
        for _, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    actual[int(saved_stage)] += 1
                except Exception:
                    continue

        selected_all = []
        for sid, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    selected_all.append(dict(scene_id=str(sid), save_stage=int(saved_stage)))
                except Exception:
                    continue
        selected_all = sorted(selected_all, key=lambda d: (str(d.get('scene_id')), int(d.get('save_stage', -1))))

        report = dict(
            stage_id=int(stage_id),
            policy='learning_dynamics',
            quota_note='stage_ratio quotas enforced per stage',
            current_stage_target=int(current_target),
            current_stage_existing=int(current_existing),
            current_stage_added=int(len(added)),
            quota_pruned_count=int(len(quota_pruned)),
            quota_pruned_entries=quota_pruned,
            added_entries=added,
            scoring=dict(
                forgetness_score='learning_dynamics_forgetness',
                replay_priority_score='learning_dynamics_replay_priority',
            ),
            warnings=dict(
                invalid_forgetness_count=int(len(invalid_forgetness)) if stage_id > 1 else 0,
                invalid_forgetness_examples=invalid_forgetness[:3] if stage_id > 1 else [],
            ),
            duplicates_filtered=int(duplicate_removed),
        )
        return selected_all, quotas, dict(actual), report

    def _normalize_design1_payload(
        self,
        *,
        payload: Optional[Dict[str, Any]],
        stage_id: int,
    ) -> Tuple[Dict[int, float], Dict[Tuple[str, int], Dict[int, Dict[str, float]]], List[int]]:
        """Validate and normalize Design payload for memory update."""
        ld_block = self._ld_design_block_key or 'learning_dynamics_design1'
        ld_payload_name = f"{ld_block}_payload"
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"selection_strategy='{self.selection_strategy}' requires "
                f"{ld_payload_name} as a dict."
            )

        payload_stage_id = payload.get('stage_id', None)
        if payload_stage_id is None or int(payload_stage_id) != int(stage_id):
            raise RuntimeError(
                f"{ld_block} payload stage mismatch: "
                f"expected stage_id={int(stage_id)}, got {payload_stage_id!r}."
            )

        class_need_raw = payload.get('class_need', None)
        if not isinstance(class_need_raw, dict) or not class_need_raw:
            raise RuntimeError(
                f"{ld_block} payload must include non-empty "
                "'class_need' dict."
            )
        class_need: Dict[int, float] = {}
        need_sum = 0.0
        for cid, val in class_need_raw.items():
            try:
                cid_i = int(cid)
                need = float(val)
            except Exception as e:
                raise RuntimeError(
                    f"{ld_block} payload has invalid class_need entry: "
                    f"{cid!r} -> {val!r}"
                ) from e
            if not np.isfinite(need):
                raise RuntimeError(
                    f"{ld_block} payload has non-finite class_need value: "
                    f"{cid_i} -> {need!r}"
                )
            need = max(0.0, float(need))
            class_need[cid_i] = need
            need_sum += need
        if need_sum <= 0.0:
            raise RuntimeError(
                f"{ld_block} payload class_need sum is non-positive."
            )
        for cid in list(class_need.keys()):
            class_need[cid] = float(class_need[cid] / need_sum)

        terms_raw = payload.get('seat_class_terms', None)
        if not isinstance(terms_raw, dict) or not terms_raw:
            raise RuntimeError(
                f"{ld_block} payload must include non-empty "
                "'seat_class_terms' dict."
            )

        seat_class_terms: Dict[Tuple[str, int], Dict[int, Dict[str, float]]] = {}
        for scene_id, by_stage in terms_raw.items():
            if not isinstance(by_stage, dict):
                raise RuntimeError(
                    f"{ld_block} payload seat_class_terms must be "
                    f"scene_id -> dict(save_stage->terms), got {type(by_stage)} "
                    f"for scene_id={scene_id!r}."
                )
            sid = str(scene_id)
            for save_stage, by_class in by_stage.items():
                try:
                    st = int(save_stage)
                except Exception as e:
                    raise RuntimeError(
                        f"{ld_block} payload has invalid save_stage key: "
                        f"{save_stage!r} (scene_id={sid})."
                    ) from e
                if not isinstance(by_class, dict):
                    raise RuntimeError(
                        f"{ld_block} payload seat_class_terms must map "
                        f"to dict(class_id->term), got {type(by_class)} "
                        f"for {(sid, st)}."
                    )
                seat_key = (sid, int(st))
                parsed_terms: Dict[int, Dict[str, float]] = {}
                for class_id, term in by_class.items():
                    try:
                        cid = int(class_id)
                    except Exception as e:
                        raise RuntimeError(
                            f"{ld_block} payload has invalid class_id key: "
                            f"{class_id!r} for seat {(sid, st)}."
                        ) from e
                    if not isinstance(term, dict):
                        raise RuntimeError(
                            f"{ld_block} payload class term must be dict, "
                            f"got {type(term)} for seat {(sid, st)}, class_id={cid}."
                        )
                    try:
                        g = float(term.get('g', 0.0))
                        r_best = float(term.get('r_best', 0.0))
                        d = float(term.get('d', 0.0))
                        u = float(term.get('u', g * r_best + d))
                    except Exception as e:
                        raise RuntimeError(
                            f"{ld_block} payload class term has invalid "
                            f"numeric values for seat {(sid, st)}, class_id={cid}: {term!r}"
                        ) from e
                    vals = [g, r_best, d, u]
                    if not all(np.isfinite(v) for v in vals):
                        raise RuntimeError(
                            f"{ld_block} payload class term has non-finite "
                            f"values for seat {(sid, st)}, class_id={cid}: {term!r}"
                        )
                    parsed_terms[int(cid)] = dict(
                        g=float(max(0.0, g)),
                        r_best=float(max(0.0, r_best)),
                        d=float(max(0.0, d)),
                        u=float(max(0.0, u)),
                    )
                seat_class_terms[seat_key] = parsed_terms

        new_classes = []
        for x in (payload.get('new_classes', []) or []):
            try:
                new_classes.append(int(x))
            except Exception:
                continue

        return class_need, seat_class_terms, new_classes

    def _design1_object_count(self, snapshot: Dict[str, Any], class_id: int) -> int:
        """Read per-seat object count for class_id from snapshot."""
        obj_counts = snapshot.get('object_counts', {}) or {}
        try:
            return int(obj_counts.get(int(class_id), obj_counts.get(str(int(class_id)), 0)) or 0)
        except Exception:
            return 0

    def _design1_transform_supply(self, supply: int) -> float:
        """Transform raw object-count supply by configured Design-1 scaling mode."""
        ld_block = self._ld_design_block_key or 'learning_dynamics_design1'
        try:
            supply_i = int(supply)
        except Exception:
            supply_i = 0
        supply_i = max(0, int(supply_i))
        if supply_i <= 0:
            return 0.0

        mode = str(self.learning_dynamics_design1_supply_scaling_mode)
        if mode == 'raw':
            out = float(supply_i)
        elif mode == 'cap':
            cap = int(self.learning_dynamics_design1_supply_cap)
            out = float(min(int(supply_i), int(cap)))
        elif mode == 'log1p':
            out = float(np.log1p(float(supply_i)))
        elif mode == 'cap_log1p':
            cap = int(self.learning_dynamics_design1_supply_cap)
            out = float(np.log1p(float(min(int(supply_i), int(cap)))))
        else:
            raise RuntimeError(
                "Invalid Design-1 supply scaling mode in runtime state: "
                f"{mode!r}"
            )

        if (not np.isfinite(out)) or out < 0.0:
            raise RuntimeError(
                f"{ld_block} supply transform produced invalid value: "
                f"mode={mode!r}, supply={supply_i}, out={out}."
            )
        return float(out)

    def _compute_design1_seat_scores(
        self,
        *,
        seat_snapshots: Dict[Tuple[str, int], Dict[str, Any]],
        class_need: Dict[int, float],
        seat_class_terms: Dict[Tuple[str, int], Dict[int, Dict[str, float]]],
    ) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """Compute unary score and 2D embedding for Design-1 seats."""
        ld_block = self._ld_design_block_key or 'learning_dynamics_design1'
        seat_class_terms = dict(seat_class_terms or {})
        missing_keys = [
            k for k in seat_snapshots.keys()
            if k not in seat_class_terms
        ]
        missing_terms = [
            dict(scene_id=str(k[0]), save_stage=int(k[1]))
            for k in missing_keys
        ]
        if missing_terms:
            if not bool(self.learning_dynamics_design1_allow_missing_seat_terms):
                raise RuntimeError(
                    f"{ld_block} payload is missing seat terms for "
                    f"{len(missing_terms)} seats (example: {missing_terms[0]})."
                )
            for k in missing_keys:
                seat_class_terms[k] = {}

        x1_by_class = defaultdict(list)  # class_id -> [G*R_best]
        x2_by_class = defaultdict(list)  # class_id -> [D]
        for seat_key, snapshot in seat_snapshots.items():
            terms = seat_class_terms.get(seat_key, {})
            for cid, term in terms.items():
                supply = self._design1_object_count(snapshot, int(cid))
                if supply <= 0:
                    continue
                x1 = float(term.get('g', 0.0)) * float(term.get('r_best', 0.0))
                x2 = float(term.get('d', 0.0))
                x1_by_class[int(cid)].append(float(x1))
                x2_by_class[int(cid)].append(float(x2))

        zstats = {}
        for cid in set(list(x1_by_class.keys()) + list(x2_by_class.keys())):
            arr1 = np.asarray(x1_by_class.get(int(cid), [0.0]), dtype=float)
            arr2 = np.asarray(x2_by_class.get(int(cid), [0.0]), dtype=float)
            mu1 = float(arr1.mean()) if arr1.size > 0 else 0.0
            mu2 = float(arr2.mean()) if arr2.size > 0 else 0.0
            std1 = float(arr1.std()) if arr1.size > 0 else 1.0
            std2 = float(arr2.std()) if arr2.size > 0 else 1.0
            if std1 <= 1e-12 or (not np.isfinite(std1)):
                std1 = 1.0
            if std2 <= 1e-12 or (not np.isfinite(std2)):
                std2 = 1.0
            zstats[int(cid)] = dict(mu1=mu1, std1=std1, mu2=mu2, std2=std2)

        seat_scores: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for seat_key, snapshot in seat_snapshots.items():
            terms = seat_class_terms.get(seat_key, {})
            unary = 0.0
            e1 = 0.0
            e2 = 0.0
            unary_terms = {}
            supply_counts = {}
            for cid, term in terms.items():
                cid_i = int(cid)
                w = float(class_need.get(cid_i, 0.0))
                if w <= 0.0:
                    continue
                supply = self._design1_object_count(snapshot, cid_i)
                if supply <= 0:
                    continue
                supply_counts[int(cid_i)] = int(supply)
                supply_weight = self._design1_transform_supply(supply)
                if supply_weight <= 0.0:
                    continue

                g = float(term.get('g', 0.0))
                r_best = float(term.get('r_best', 0.0))
                d = float(term.get('d', 0.0))
                x1 = float(g * r_best)
                x2 = float(d)
                u = float(term.get('u', x1 + x2))

                unary_comp = w * float(supply_weight) * float(max(0.0, u))
                unary += float(unary_comp)
                unary_terms[int(cid_i)] = float(
                    unary_terms.get(int(cid_i), 0.0) + float(unary_comp)
                )
                zs = zstats.get(cid_i, dict(mu1=0.0, std1=1.0, mu2=0.0, std2=1.0))
                e1 += w * ((x1 - float(zs['mu1'])) / float(zs['std1']))
                e2 += w * ((x2 - float(zs['mu2'])) / float(zs['std2']))

            unary = float(max(0.0, unary))
            if (not np.isfinite(unary)) or (not np.isfinite(e1)) or (not np.isfinite(e2)):
                raise RuntimeError(
                    f"{ld_block} seat score computation produced "
                    f"non-finite values for seat={seat_key}: unary={unary}, e1={e1}, e2={e2}."
                )
            norm = float(np.sqrt(e1 * e1 + e2 * e2))
            if norm > 0:
                emb = (float(e1 / norm), float(e2 / norm))
            else:
                emb = (0.0, 0.0)
            seat_scores[seat_key] = dict(
                unary=float(unary),
                e1=float(e1),
                e2=float(e2),
                emb=emb,
                unary_terms={int(k): float(v) for k, v in unary_terms.items()},
                supply_counts={int(k): int(v) for k, v in supply_counts.items()},
            )

        return seat_scores

    def _design1_kernel(self,
                        emb_a: Tuple[float, float],
                        emb_b: Tuple[float, float],
                        *,
                        enabled: bool) -> float:
        """Positive-part cosine-like kernel on normalized 2D embeddings."""
        if not enabled:
            return 0.0
        dot = float(emb_a[0] * emb_b[0] + emb_a[1] * emb_b[1])
        if not np.isfinite(dot):
            return 0.0
        return float(max(0.0, dot))

    def _apply_stage_ratio_update_learning_dynamics_design1(
        self,
        stage_id: int,
        candidate_scenes: List[Dict[str, Any]],
        *,
        learning_dynamics_design1_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, int], Dict[str, Any]]:
        """Stage-ratio ratio-targeted update for LD Design-1/Design-2.

        Stage-1:
          - greedy fill with percentile-normalized unary + mean compatibility.

        Stage>=2:
          - derive hard stage targets from stage-ratio quotas.
          - add current-stage seats to close current-stage gap.
          - evict old-stage seats proportionally to old-stage surpluses.
          - enforce exact target composition when no current-stage shortage exists.
          - allow explicit shortage reporting when current-stage candidates are
            insufficient.
        """
        stage_id = int(stage_id)
        ld_policy = (
            LD_DESIGN2_STRATEGY
            if self.selection_strategy == LD_DESIGN2_STRATEGY
            else LD_DESIGN1_STRATEGY
        )
        duplicate_removed = self._enforce_unique_scene_ids_in_memory()
        if duplicate_removed > 0:
            print(
                f"WARNING: SceneMemoryBank removed {int(duplicate_removed)} duplicate "
                f"scene-stage entries to enforce unique scene IDs."
            )
        quotas = self._compute_stage_ratio_quotas(stage_id)

        class_need, seat_class_terms, _ = self._normalize_design1_payload(
            payload=learning_dynamics_design1_payload,
            stage_id=stage_id,
        )

        # Existing seat set.
        existing_pairs: List[Tuple[str, int]] = []
        for sid, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    existing_pairs.append((str(sid), int(saved_stage)))
                except Exception:
                    continue
        existing_pairs = sorted(set(existing_pairs))
        existing_pair_set = set(existing_pairs)

        existing_scene_ids = set(str(sid) for sid in self.memory_scenes.keys())

        # Keep only current-stage candidate seats. When unique-scene mode is
        # enabled, do not admit a candidate whose scene_id is already present
        # in the bank under any save_stage.
        stage_candidates: List[Dict[str, Any]] = []
        seen_candidate_pairs = set()
        for cand in sorted(candidate_scenes, key=lambda c: str(c.get('scene_id'))):
            sid = str(cand.get('scene_id'))
            snap = cand.get('snapshot', {}) or {}
            save_stage = int(snap.get('save_stage', cand.get('stage_id', stage_id)))
            if int(save_stage) != int(stage_id):
                continue
            if self.enforce_unique_scene_ids and sid in existing_scene_ids:
                continue
            pair = (sid, int(save_stage))
            if pair in existing_pair_set or pair in seen_candidate_pairs:
                continue
            stage_candidates.append(cand)
            seen_candidate_pairs.add(pair)
        feasible_candidate_count = int(len(stage_candidates))

        # Build snapshots for all scored seats: current bank + current-stage candidates.
        seat_snapshots: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for sid, sdata in self.memory_scenes.items():
            for saved_stage, stage_data in (sdata.get('stages', {}) or {}).items():
                try:
                    st = int(saved_stage)
                except Exception:
                    continue
                seat_snapshots[(str(sid), int(st))] = stage_data.get('snapshot', {}) or {}
        for cand in stage_candidates:
            sid = str(cand.get('scene_id'))
            snap = cand.get('snapshot', {}) or {}
            st = int(snap.get('save_stage', stage_id))
            seat_snapshots[(sid, int(st))] = snap

        seat_scores = self._compute_design1_seat_scores(
            seat_snapshots=seat_snapshots,
            class_need=class_need,
            seat_class_terms=seat_class_terms,
        )

        # Persist Design-1 seat scores for existing seats for traceability.
        for (sid, st), score in seat_scores.items():
            if sid in self.memory_scenes and st in (self.memory_scenes[sid].get('stages', {}) or {}):
                try:
                    sref = self.memory_scenes[sid]['stages'][int(st)]
                    sref['learning_dynamics_design1_unary'] = float(score['unary'])
                    sref['learning_dynamics_design1_e1'] = float(score['e1'])
                    sref['learning_dynamics_design1_e2'] = float(score['e2'])
                except Exception:
                    pass

        bank_pairs = sorted(existing_pairs)
        candidate_pairs = sorted(
            [
                (
                    str(c.get('scene_id')),
                    int((c.get('snapshot', {}) or {}).get('save_stage', stage_id)),
                )
                for c in stage_candidates
            ],
            key=lambda x: (str(x[0]), int(x[1])),
        )

        token_to_pair: Dict[int, Tuple[str, int]] = {}
        bank_pair_count = int(len(bank_pairs))
        for idx, pair in enumerate(bank_pairs):
            token_to_pair[int(idx)] = pair
        for idx, pair in enumerate(candidate_pairs):
            token_to_pair[int(bank_pair_count + idx)] = pair

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kernel_enabled = bool(self.learning_dynamics_design1_use_compatibility_kernel)
        compat_weight = float(self.learning_dynamics_design1_compatibility_weight)
        lambda_compat = float(compat_weight if kernel_enabled else 0.0)

        bank_indices = torch.arange(
            0, bank_pair_count, dtype=torch.long, device=device
        )
        cand_indices = torch.arange(
            bank_pair_count,
            bank_pair_count + int(len(candidate_pairs)),
            dtype=torch.long,
            device=device,
        )
        bank_stage_ids = torch.tensor(
            [int(pair[1]) for pair in bank_pairs],
            dtype=torch.long,
            device=device,
        ) if bank_pairs else torch.empty((0,), dtype=torch.long, device=device)
        cand_stage_ids = torch.tensor(
            [int(pair[1]) for pair in candidate_pairs],
            dtype=torch.long,
            device=device,
        ) if candidate_pairs else torch.empty((0,), dtype=torch.long, device=device)

        bank_unary_raw = torch.tensor(
            [float((seat_scores.get(pair, {}) or {}).get('unary', 0.0)) for pair in bank_pairs],
            dtype=torch.float32,
            device=device,
        ) if bank_pairs else torch.empty((0,), dtype=torch.float32, device=device)
        cand_unary_raw = torch.tensor(
            [float((seat_scores.get(pair, {}) or {}).get('unary', 0.0)) for pair in candidate_pairs],
            dtype=torch.float32,
            device=device,
        ) if candidate_pairs else torch.empty((0,), dtype=torch.float32, device=device)
        bank_u_norm = compute_percentile_rank(bank_unary_raw)
        cand_u_norm = compute_percentile_rank(cand_unary_raw)

        bank_emb = torch.tensor(
            [tuple((seat_scores.get(pair, {}) or {}).get('emb', (0.0, 0.0))) for pair in bank_pairs],
            dtype=torch.float32,
            device=device,
        ) if bank_pairs else torch.empty((0, 2), dtype=torch.float32, device=device)
        cand_emb = torch.tensor(
            [tuple((seat_scores.get(pair, {}) or {}).get('emb', (0.0, 0.0))) for pair in candidate_pairs],
            dtype=torch.float32,
            device=device,
        ) if candidate_pairs else torch.empty((0, 2), dtype=torch.float32, device=device)
        use_class_balance = bool(self.learning_dynamics_design1_use_class_balance)
        design_v = int(self.learning_dynamics_design_version)

        all_class_ids = set(int(k) for k in (class_need or {}).keys())
        for pair in (bank_pairs + candidate_pairs):
            score = seat_scores.get(pair, {}) or {}
            for k in (score.get('unary_terms', {}) or {}).keys():
                all_class_ids.add(int(k))
            for k in (score.get('supply_counts', {}) or {}).keys():
                all_class_ids.add(int(k))
        all_class_ids = sorted(all_class_ids)
        if bool(use_class_balance) and len(all_class_ids) == 0:
            raise RuntimeError(
                f"{ld_policy}.use_class_balance=True requires "
                "at least one class dimension in unary terms."
            )
        cid_to_col = {
            int(cid): int(col) for col, cid in enumerate(all_class_ids)
        }

        def _build_class_matrix(
            pairs: List[Tuple[str, int]],
            field_name: str,
            value_cast,
        ) -> torch.Tensor:
            if len(pairs) == 0:
                return torch.empty(
                    (0, int(len(all_class_ids))),
                    dtype=torch.float32,
                    device=device,
                )
            if len(all_class_ids) == 0:
                return torch.empty((len(pairs), 0), dtype=torch.float32, device=device)
            mat = torch.zeros(
                (len(pairs), len(all_class_ids)),
                dtype=torch.float32,
                device=device,
            )
            for r, pair in enumerate(pairs):
                vals = (seat_scores.get(pair, {}) or {}).get(field_name, {}) or {}
                for cid, v in vals.items():
                    cid_i = int(cid)
                    if cid_i not in cid_to_col:
                        continue
                    mat[r, int(cid_to_col[cid_i])] = float(value_cast(v))
            return mat

        bank_unary_base = _build_class_matrix(
            bank_pairs, 'unary_terms', float
        )
        cand_unary_base = _build_class_matrix(
            candidate_pairs, 'unary_terms', float
        )
        bank_supply_counts_tensor = _build_class_matrix(
            bank_pairs, 'supply_counts', int
        )
        cand_supply_counts_tensor = _build_class_matrix(
            candidate_pairs, 'supply_counts', int
        )

        added_entries: List[Dict[str, Any]] = []
        pruned_entries: List[Dict[str, Any]] = []

        counts_before = defaultdict(int)
        for _, st in bank_pairs:
            counts_before[int(st)] += 1
        current_stage_existing = int(counts_before.get(int(stage_id), 0))
        current_stage_target = int(quotas.get(int(stage_id), 0))
        required_add_t = int(max(0, current_stage_target - current_stage_existing))
        add_t = int(min(required_add_t, feasible_candidate_count))
        shortfall_t = int(required_add_t - add_t)
        stage2plus_report: Dict[str, Any] = {}

        def _persist_design1_scores(pair_key: Tuple[str, int]) -> None:
            sid, st = pair_key
            if sid not in self.memory_scenes:
                return
            if int(st) not in (self.memory_scenes[sid].get('stages', {}) or {}):
                return
            try:
                sref = self.memory_scenes[sid]['stages'][int(st)]
                sref['learning_dynamics_design1_unary'] = float(
                    (seat_scores.get(pair_key, {}) or {}).get('unary', 0.0)
                )
                sref['learning_dynamics_design1_e1'] = float(
                    (seat_scores.get(pair_key, {}) or {}).get('e1', 0.0)
                )
                sref['learning_dynamics_design1_e2'] = float(
                    (seat_scores.get(pair_key, {}) or {}).get('e2', 0.0)
                )
            except Exception:
                pass

        # Bug C.3: always track RAW object counts for balance weighting,
        # regardless of supply_scaling_mode.  supply_counts in seat_scores
        # are raw int counts (see _design1_object_count), so the tensor
        # built from 'supply_counts' field is already raw.
        bank_raw_counts_tensor = _build_class_matrix(
            bank_pairs, 'supply_counts', int
        )
        cand_raw_counts_tensor = _build_class_matrix(
            candidate_pairs, 'supply_counts', int
        )

        if int(stage_id) == 1:
            free_slots = int(max(0, int(self.memory_budget) - int(len(bank_pairs))))
            if bool(use_class_balance):
                selected_mask = torch.zeros(
                    (int(len(candidate_pairs)),), dtype=torch.bool, device=device
                )
                active_bank_emb = bank_emb.clone()
                # Bug C.3 fix: use raw counts for balance, not scaled supply.
                active_count_bank_raw = (
                    torch.sum(bank_raw_counts_tensor, dim=0)
                    if int(bank_raw_counts_tensor.numel()) > 0
                    else torch.zeros(
                        (int(len(all_class_ids)),),
                        dtype=torch.float32,
                        device=device,
                    )
                )
                for _ in range(int(free_slots)):
                    if int(len(candidate_pairs)) <= 0:
                        break
                    if not bool(torch.any(~selected_mask).item()):
                        break
                    # Bug C.2 fix: balance weights are applied to raw supply
                    # counts via matrix multiply on cand_unary_base (which
                    # already contains class_need * supply_weight * u).
                    # Design 1: 1/sqrt(1+count); Design 2: 1/(1+count) capped at w_max.
                    if int(design_v) >= 2:
                        w_bal = compute_class_balance_weights_v2(
                            active_count_bank_raw,
                            w_max=float(self.learning_dynamics_design2_w_max),
                        )
                    else:
                        w_bal = compute_class_balance_weights(active_count_bank_raw)
                    cand_unary_bal = (
                        torch.mv(cand_unary_base, w_bal)
                        if int(cand_unary_base.numel()) > 0
                        else torch.empty((0,), dtype=torch.float32, device=device)
                    )
                    # Step 1C: minimum per-class scene quota boost (design 2).
                    if int(design_v) >= 2:
                        min_q = int(self.learning_dynamics_design2_min_class_quota)
                        if min_q > 0 and int(cand_raw_counts_tensor.numel()) > 0:
                            under_quota = (active_count_bank_raw < float(min_q))  # (C,)
                            # For each candidate, sum its raw counts for under-quota classes.
                            boost = torch.mv(
                                cand_raw_counts_tensor.clamp(min=0.0),
                                under_quota.to(dtype=torch.float32),
                            )  # (N_cand,)
                            # Normalise boost to [0,1] range via percentile rank
                            # and add to balanced unary so under-represented
                            # candidates get a lift.
                            if float(boost.max().item()) > 0.0:
                                cand_unary_bal = cand_unary_bal + boost
                    cand_u_norm_dyn = compute_percentile_rank(cand_unary_bal)
                    # Step 2: Design 2 uses redundancy penalty; Design 1 uses
                    # compatibility reward.
                    if int(design_v) >= 2:
                        red_lam = float(self.learning_dynamics_design2_redundancy_lambda)
                        if red_lam > 0.0 and int(active_bank_emb.shape[0]) > 0:
                            red_vec = compute_redundancy_penalty(
                                cand_emb, active_bank_emb,
                                topk=int(self.learning_dynamics_design2_redundancy_topk),
                            )
                            score_vec = (
                                (1.0 - red_lam) * cand_u_norm_dyn
                                - red_lam * red_vec
                            )
                        else:
                            score_vec = cand_u_norm_dyn.clone()
                    else:
                        # Design 1 original: compatibility reward.
                        if int(active_bank_emb.shape[0]) == 0 and abs(float(lambda_compat) - 1.0) <= 1e-12:
                            score_vec = cand_u_norm_dyn.clone()
                        else:
                            comp_vec = compute_comp_mean(cand_emb, active_bank_emb)
                            score_vec = (
                                (1.0 - float(lambda_compat)) * cand_u_norm_dyn
                                + float(lambda_compat) * comp_vec
                            )
                    score_vec = score_vec.masked_fill(selected_mask, float('-inf'))
                    best_idx = int(torch.argmax(score_vec).item())
                    if torch.isneginf(score_vec[best_idx]):
                        break
                    selected_mask[best_idx] = True
                    pair = candidate_pairs[best_idx]
                    sid, st = pair
                    snap = seat_snapshots.get(pair, {})
                    self._add_scene_to_memory(str(sid), int(st), snap, importance=1.0)
                    _persist_design1_scores(pair)
                    active_bank_emb = torch.cat(
                        [active_bank_emb, cand_emb[best_idx].view(1, -1)], dim=0
                    )
                    # Bug C.3: update raw count tracker (not scaled supply).
                    if int(cand_raw_counts_tensor.numel()) > 0:
                        active_count_bank_raw = active_count_bank_raw + cand_raw_counts_tensor[best_idx]
                    added_entries.append(dict(
                        scene_id=str(sid),
                        save_stage=int(st),
                        unary=float((seat_scores.get(pair, {}) or {}).get('unary', 0.0)),
                        unary_percentile=float(cand_u_norm_dyn[best_idx].item()),
                        unary_balanced=float(cand_unary_bal[best_idx].item()),
                        reason='stage1_fill_class_balance',
                    ))
            else:
                selected_local = stage1_greedy_fill(
                    u_norm=cand_u_norm,
                    E_all=cand_emb,
                    K=free_slots,
                    lambda_compat=lambda_compat,
                )
                selected_local = [
                    int(x) for x in selected_local.detach().cpu().tolist()
                ]
                for local_idx in selected_local:
                    if local_idx < 0 or local_idx >= int(len(candidate_pairs)):
                        continue
                    pair = candidate_pairs[local_idx]
                    sid, st = pair
                    snap = seat_snapshots.get(pair, {})
                    self._add_scene_to_memory(str(sid), int(st), snap, importance=1.0)
                    _persist_design1_scores(pair)
                    added_entries.append(dict(
                        scene_id=str(sid),
                        save_stage=int(st),
                        unary=float((seat_scores.get(pair, {}) or {}).get('unary', 0.0)),
                        unary_percentile=float(cand_u_norm[local_idx].item()),
                        reason='stage1_fill',
                    ))
            add_t = int(len(added_entries))
            shortfall_t = int(max(0, required_add_t - add_t))
        else:
            # Design 2: use redundancy_lambda instead of compat_weight.
            stage2_lambda = (
                float(self.learning_dynamics_design2_redundancy_lambda)
                if int(design_v) >= 2
                else float(lambda_compat)
            )
            updated_bank_tokens, stage2plus_report = ratio_targeted_swap_update(
                bank_indices=bank_indices,
                bank_stage_ids=bank_stage_ids,
                E_bank=bank_emb,
                u_bank_norm=bank_u_norm,
                cand_indices=cand_indices,
                cand_stage_ids=cand_stage_ids,
                E_cand=cand_emb,
                u_cand_norm=cand_u_norm,
                lambda_compat=stage2_lambda,
                target_stage_counts=quotas,
                current_stage_id=stage_id,
                use_class_balance=bool(use_class_balance),
                bank_unary_base=bank_unary_base,
                cand_unary_base=cand_unary_base,
                bank_supply_counts=bank_supply_counts_tensor,
                cand_supply_counts=cand_supply_counts_tensor,
                design_version=int(design_v),
                w_max=float(self.learning_dynamics_design2_w_max),
                min_class_quota=int(self.learning_dynamics_design2_min_class_quota),
            )

            action_rows = list(stage2plus_report.get('actions', []) or [])
            for action in action_rows:
                ctoken = int(action.get('candidate_token'))
                cpair = token_to_pair.get(ctoken, None)
                if cpair is None:
                    raise RuntimeError(
                        f"{ld_policy} action references unknown "
                        f"candidate token={ctoken}."
                    )

                if str(action.get('reason')) == 'swap':
                    ev_token = int(action.get('evicted_token'))
                    epair = token_to_pair.get(ev_token, None)
                    if epair is None:
                        raise RuntimeError(
                            f"{ld_policy} action references unknown "
                            f"evicted token={ev_token}."
                        )
                    self._remove_scene_from_memory(
                        str(epair[0]), stage_id=int(epair[1])
                    )
                    pruned_entries.append(dict(
                        scene_id=str(epair[0]),
                        save_stage=int(epair[1]),
                        unary=float(
                            (seat_scores.get(epair, {}) or {}).get('unary', 0.0)
                        ),
                        unary_percentile=float(action.get('evicted_unary_percentile', 0.0)),
                        unary_balanced=(
                            float(action.get('evicted_unary_balanced'))
                            if action.get('evicted_unary_balanced', None) is not None
                            else None
                        ),
                        reason='ratio_surplus_swap_evict',
                        swap_delta=float(action.get('swap_delta')),
                    ))

                sid, st = cpair
                snap = seat_snapshots.get(cpair, {})
                self._add_scene_to_memory(str(sid), int(st), snap, importance=1.0)
                _persist_design1_scores(cpair)
                cidx = int(ctoken - bank_pair_count)
                c_u_norm = (
                    float(action.get('candidate_unary_percentile', 0.0))
                    if int(stage_id) > 1 else
                    (
                        float(cand_u_norm[cidx].item())
                        if 0 <= cidx < int(cand_u_norm.shape[0]) else 0.0
                    )
                )
                added_entries.append(dict(
                    scene_id=str(sid),
                    save_stage=int(st),
                    unary=float((seat_scores.get(cpair, {}) or {}).get('unary', 0.0)),
                    unary_percentile=float(c_u_norm),
                    unary_balanced=(
                        float(action.get('candidate_unary_balanced'))
                        if action.get('candidate_unary_balanced', None) is not None
                        else None
                    ),
                    reason=str(action.get('reason')),
                    swap_delta=(
                        float(action.get('swap_delta'))
                        if action.get('swap_delta', None) is not None
                        else None
                    ),
                ))

            expected_pairs = set()
            for tok in updated_bank_tokens.detach().cpu().tolist():
                token_i = int(tok)
                if token_i not in token_to_pair:
                    raise RuntimeError(
                        f"{ld_policy} returned unknown bank token: "
                        f"{token_i}"
                    )
                expected_pairs.add(token_to_pair[token_i])
            current_pairs = set()
            for sid, sdata in self.memory_scenes.items():
                for saved_stage in (sdata.get('stages', {}) or {}).keys():
                    try:
                        current_pairs.add((str(sid), int(saved_stage)))
                    except Exception:
                        continue
            if expected_pairs != current_pairs:
                raise RuntimeError(
                    f"{ld_policy} post-update pair-set mismatch: "
                    f"expected={len(expected_pairs)} pairs, got={len(current_pairs)}."
                )
            add_t = int(stage2plus_report.get('add_t', add_t))
            shortfall_t = int(stage2plus_report.get('shortfall_t', shortfall_t))

        if self.enforce_unique_scene_ids:
            duplicate_scene_id = None
            duplicate_stage_count = 0
            for sid, sdata in self.memory_scenes.items():
                stage_count = len((sdata.get('stages', {}) or {}))
                if stage_count > 1:
                    duplicate_scene_id = str(sid)
                    duplicate_stage_count = int(stage_count)
                    break
            if duplicate_scene_id is not None:
                raise RuntimeError(
                    f"{ld_policy} violated enforce_unique_scene_ids=True: "
                    f"scene_id={duplicate_scene_id} has {duplicate_stage_count} stage snapshots."
                )

        self._recompute_class_distribution()

        actual = defaultdict(int)
        for _, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    actual[int(saved_stage)] += 1
                except Exception:
                    continue

        selected_all = []
        for sid, sdata in self.memory_scenes.items():
            for saved_stage in (sdata.get('stages', {}) or {}).keys():
                try:
                    selected_all.append(dict(scene_id=str(sid), save_stage=int(saved_stage)))
                except Exception:
                    continue
        selected_all = sorted(
            selected_all,
            key=lambda d: (str(d.get('scene_id')), int(d.get('save_stage', -1))),
        )

        current_stage_added = int(
            len([x for x in added_entries if int(x.get('save_stage', -1)) == int(stage_id)])
        )
        exact_target_match = all(
            int(actual.get(int(s), 0)) == int(quotas.get(int(s), 0))
            for s in range(1, int(stage_id) + 1)
        )
        if int(shortfall_t) == 0 and not exact_target_match:
            raise RuntimeError(
                f"{ld_policy} requires exact stage-ratio composition "
                "when shortfall_t==0, but mismatch found: "
                f"stage_id={stage_id}, quotas={quotas}, actual={dict(actual)}."
            )
        if int(shortfall_t) > 0 and self.debug_mode:
            print(
                f"WARNING: {ld_policy} candidate shortage at stage "
                f"{int(stage_id)}: shortfall_t={int(shortfall_t)} "
                f"(required_add_t={int(required_add_t)}, feasible={int(feasible_candidate_count)})."
            )

        report = dict(
            stage_id=int(stage_id),
            policy=str(ld_policy),
            design_version=int(design_v),
            q_metric=str(self.learning_dynamics_design1_q_metric),
            quota_note=(
                "stage_ratio enforced as hard target per stage when feasible; "
                "explicit shortfall allowed only on candidate shortage"
            ),
            target_stage_counts={int(k): int(v) for k, v in quotas.items()},
            current_stage_target=int(current_stage_target),
            current_stage_existing=int(current_stage_existing),
            required_add_t=int(required_add_t),
            feasible_candidate_count=int(feasible_candidate_count),
            add_t=int(add_t),
            current_stage_added=int(current_stage_added),
            shortfall_t=int(shortfall_t),
            exact_target_match=bool(exact_target_match),
            use_compatibility_kernel=bool(kernel_enabled),
            use_class_balance=bool(use_class_balance),
            compatibility_weight=float(compat_weight),
            lambda_compat=float(lambda_compat),
            class_balance_formula=(
                f'w_bal[c]=min(1/(1+count),{self.learning_dynamics_design2_w_max})'
                if int(design_v) >= 2
                else 'w_bal[c]=1/sqrt(1+count_bank[c])'
            ),
            supply_scaling_mode=str(self.learning_dynamics_design1_supply_scaling_mode),
            supply_cap=(
                int(self.learning_dynamics_design1_supply_cap)
                if self.learning_dynamics_design1_supply_cap is not None
                else None
            ),
            enforce_unique_scene_ids=bool(self.enforce_unique_scene_ids),
            duplicate_removed_to_enforce_unique_scene_ids=int(duplicate_removed),
            # Design 2 specific parameters.
            design2_w_max=(
                float(self.learning_dynamics_design2_w_max)
                if int(design_v) >= 2 else None
            ),
            design2_redundancy_lambda=(
                float(self.learning_dynamics_design2_redundancy_lambda)
                if int(design_v) >= 2 else None
            ),
            design2_redundancy_topk=(
                int(self.learning_dynamics_design2_redundancy_topk)
                if int(design_v) >= 2 else None
            ),
            design2_min_class_quota=(
                int(self.learning_dynamics_design2_min_class_quota)
                if int(design_v) >= 2 else None
            ),
            # Backward-compatible fields kept for existing analysis scripts.
            current_stage_upper_bound=int(current_stage_target),
            current_stage_max_additions=int(required_add_t),
            current_stage_lower_bound=0,
            forced_accept_count=0,
            positive_delta_accept_count=int(current_stage_added),
            # Stage>=2 detailed ratio-target update report.
            ratio_targeted_update=stage2plus_report if int(stage_id) > 1 else {},
            added_entries=added_entries,
            pruned_entries=pruned_entries,
        )
        return selected_all, quotas, dict(actual), report

    def _compute_underlearning_score_for_snapshot(self,
                                                  snapshot: Dict[str, Any],
                                                  *,
                                                  class_ap: Dict[int, float],
                                                  new_classes: List[int]) -> float:
        """Compute a per-seat under-learning score from per-class AP on new classes.

        Definitions:
          u(c) = max(0, 1 - AP(c))

        Modes:
          - object_count_sum: sum_c count(c) * u(c)
          - presence_sum: sum_c 1[c present] * u(c)
        """
        if not snapshot or not class_ap or not new_classes:
            return 0.0

        # Sanitize AP keys (JSON loads as str keys)
        ap_map: Dict[int, float] = {}
        for k, v in (class_ap or {}).items():
            try:
                ap_map[int(k)] = float(v)
            except Exception:
                continue
        new_set = {int(x) for x in (new_classes or [])}

        def _w(cid: int) -> float:
            ap = float(ap_map.get(int(cid), 0.0))
            if not np.isfinite(ap):
                ap = 0.0
            ap = float(max(0.0, min(1.0, ap)))
            return float(max(0.0, 1.0 - ap))

        if self.underlearning_score_mode == 'presence_sum':
            present = snapshot.get('present_classes', []) or []
            score = 0.0
            for cid in present:
                try:
                    cid_int = int(cid)
                except Exception:
                    continue
                if cid_int not in new_set:
                    continue
                score += _w(cid_int)
            return float(score)

        # object_count_sum (default)
        obj_counts = snapshot.get('object_counts', {}) or {}
        score = 0.0
        for cid, cnt in obj_counts.items():
            try:
                cid_int = int(cid)
            except Exception:
                continue
            if cid_int not in new_set:
                continue
            try:
                cnt_f = float(cnt)
            except Exception:
                continue
            score += cnt_f * _w(cid_int)
        return float(score)
    
    def _count_scene_stage_pairs(self):
        """Count total scene-stage pairs in memory."""
        return sum(len(scene_data['stages']) for scene_data in self.memory_scenes.values())

    def _count_stage_distribution_in_memory(self) -> Dict[int, int]:
        """Count active memory seats per save stage."""
        counts = defaultdict(int)
        for _, scene_data in self.memory_scenes.items():
            for saved_stage in (scene_data.get('stages', {}) or {}).keys():
                try:
                    counts[int(saved_stage)] += 1
                except Exception:
                    continue
        return dict(counts)

    def _count_total_unique_scenes(self) -> int:
        """Count unique scenes tracked by the active mode."""
        if self.use_legacy_mode:
            return len(self.scene_snapshots)
        return len(self.memory_scenes)

    def _count_total_snapshots(self) -> int:
        """Count scene snapshots tracked by the active mode."""
        if self.use_legacy_mode:
            return sum(len(s) for s in self.scene_snapshots.values())
        return self._count_scene_stage_pairs()

    def _sync_active_stage_stats(self):
        """Refresh stage statistics from active memory seats in global mode."""
        if self.use_legacy_mode:
            return
        self.scenes_per_stage = defaultdict(int)
        active_counts = self._count_stage_distribution_in_memory()
        for stage_id, count in active_counts.items():
            self.scenes_per_stage[int(stage_id)] = int(count)
        total_from_stage_counts = int(sum(self.scenes_per_stage.values()))
        total_pairs = int(self._count_scene_stage_pairs())
        assert total_from_stage_counts == total_pairs, (
            f"Active stage counts mismatch total scene-stage pairs: "
            f"{total_from_stage_counts} != {total_pairs}"
        )
        assert total_pairs <= int(self.memory_budget), (
            f"Memory bank exceeds budget after stage stats sync: "
            f"{total_pairs} > {self.memory_budget}"
        )

    def _enforce_unique_scene_ids_in_memory(self) -> int:
        """Ensure each scene_id has at most one stage snapshot in memory.

        Returns:
            Number of removed (scene_id, stage_id) entries.
        """
        if not bool(getattr(self, 'enforce_unique_scene_ids', True)):
            return 0

        removed = 0
        for sid in list(self.memory_scenes.keys()):
            sdata = self.memory_scenes.get(sid, {}) or {}
            stages = list((sdata.get('stages', {}) or {}).keys())
            if len(stages) <= 1:
                continue

            # Keep the earliest snapshot to maximize unique-scene coverage while
            # preserving stage provenance (later-stage supervision is provided
            # via optional pseudo-label enrichment).
            keep_stage = None
            for st in stages:
                try:
                    st_i = int(st)
                except Exception:
                    continue
                keep_stage = st_i if keep_stage is None else min(int(keep_stage), int(st_i))
            if keep_stage is None:
                continue

            for st in stages:
                try:
                    st_i = int(st)
                except Exception:
                    continue
                if int(st_i) == int(keep_stage):
                    continue
                self._remove_scene_from_memory(str(sid), stage_id=int(st_i))
                removed += 1

        return int(removed)

    def list_memory_entries(self,
                            *,
                            max_save_stage: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all memory entries as explicit (scene_id, save_stage) seats.

        This is the canonical read API for reviewing/replay code that needs
        stage-aligned GT.

        Args:
            max_save_stage: Optional inclusive upper bound on save_stage.

        Returns:
            List of dict entries with keys:
              - scene_id (str)
              - save_stage (int)
              - importance (float)
              - snapshot (dict)  # includes data_info/present_classes/object_counts/save_stage
        """
        entries: List[Dict[str, Any]] = []
        bound = None if max_save_stage is None else int(max_save_stage)
        for scene_id, scene_data in self.memory_scenes.items():
            for saved_stage, stage_data in (scene_data.get('stages', {}) or {}).items():
                try:
                    save_stage = int(saved_stage)
                except Exception:
                    continue
                if bound is not None and save_stage > bound:
                    continue
                snapshot = stage_data.get('snapshot', {}) or {}
                entries.append({
                    'scene_id': str(scene_id),
                    'save_stage': int(snapshot.get('save_stage', save_stage)),
                    'importance': float(stage_data.get('importance', 0.0)),
                    'snapshot': snapshot,
                })
        # Deterministic ordering for reproducibility/debugging
        entries.sort(key=lambda e: (str(e.get('scene_id')), int(e.get('save_stage', -1))))
        return entries
    
    def _add_scene_to_memory(self, scene_id: str, stage_id: int, snapshot: Dict, importance: float):
        """Add a scene to the memory bank with the new nested structure."""
        if scene_id not in self.memory_scenes:
            # Create new scene entry
            self.memory_scenes[scene_id] = {
                'stages': {},
                'latest_stage': stage_id,
                'total_importance': 0.0,
                'present_classes': set()
            }
        
        # Add stage data
        self.memory_scenes[scene_id]['stages'][stage_id] = {
            'snapshot': snapshot,
            'importance': importance,
            'added_time': len(self.memory_scenes)  # Simple ordering
        }
        
        # Update metadata
        scene_data = self.memory_scenes[scene_id]
        scene_data['latest_stage'] = max(scene_data['latest_stage'], stage_id)
        scene_data['total_importance'] = sum(stage_data['importance'] 
                                           for stage_data in scene_data['stages'].values())
        
        # Update present classes (union across all stages)
        scene_data['present_classes'] = set()
        for stage_data in scene_data['stages'].values():
            scene_data['present_classes'].update(stage_data['snapshot'].get('present_classes', []))
        
        # Validate memory budget
        scene_stage_pairs = self._count_scene_stage_pairs()
        if scene_stage_pairs > self.memory_budget:
            raise RuntimeError(
                f"Memory bank exceeds scene-stage budget: {scene_stage_pairs} > {self.memory_budget} "
                f"(unique scenes: {len(self.memory_scenes)})"
            )
    
    def _remove_scene_from_memory(self, scene_id: str, stage_id: int = None):
        """Remove a scene or specific stage from memory bank."""
        if scene_id not in self.memory_scenes:
            return
            
        if stage_id is None:
            # Remove entire scene
            del self.memory_scenes[scene_id]
        else:
            # Remove specific stage
            scene_data = self.memory_scenes[scene_id]
            if stage_id in scene_data['stages']:
                del scene_data['stages'][stage_id]
                
                # Remove scene if no stages left
                if not scene_data['stages']:
                    del self.memory_scenes[scene_id]
                else:
                    # Recalculate metadata
                    scene_data['latest_stage'] = max(scene_data['stages'].keys())
                    scene_data['total_importance'] = sum(stage_data['importance'] 
                                                       for stage_data in scene_data['stages'].values())
                    scene_data['present_classes'] = set()
                    for stage_data in scene_data['stages'].values():
                        scene_data['present_classes'].update(stage_data['snapshot'].get('present_classes', []))
    
    def _get_memory_scene_count(self) -> int:
        """Get the number of unique scenes in memory bank."""
        return len(self.memory_scenes)
    
    def _get_scene_stage_count(self, scene_id: str) -> int:
        """Get the number of stages for a specific scene."""
        return len(self.memory_scenes.get(scene_id, {}).get('stages', {}))
    
    def add_stage_scenes(self, 
                        stage_id: int,
                        scene_infos: List[Dict],
                        seen_classes: List[int],
                        mappings: Dict[str, Any],
                        dataset_ref=None,
                        scene_metrics: Optional[Dict[str, Any]] = None,
                        forgetness_class_drops: Optional[Dict[int, float]] = None,
                        underlearning_class_ap: Optional[Dict[int, float]] = None,
                        underlearning_new_classes: Optional[List[int]] = None,
                        learning_dynamics_forgetness_by_seat: Optional[Dict[str, Dict[int, float]]] = None,
                        learning_dynamics_replay_priority_by_seat: Optional[Dict[str, Dict[int, float]]] = None,
                        learning_dynamics_design1_payload: Optional[Dict[str, Any]] = None,
                        learning_dynamics_design2_payload: Optional[Dict[str, Any]] = None):
        """Add scenes from a completed stage with proper label filtering.
        
        Args:
            stage_id: Stage identifier (1, 2, 3, ...)
            scene_infos: List of scene data_info dicts from the dataset
            seen_classes: Classes visible up to this stage (cumulative)
            mappings: Class mappings for NYU40 -> model conversion
            dataset_ref: Reference to dataset (for future use)
            scene_metrics: Optional per-scene metrics dict keyed by scene_id.
                When provided, entries are attached to candidates under the
                'metrics' key for use by advanced selection strategies
                (uncertainty/diversity/combined).
            forgetness_class_drops: Optional per-class AP drops (old classes)
                used only when `forgetness_eviction.enabled=True` in config.
            underlearning_class_ap: Optional per-class AP on the train split for
                the current stage's new classes. Used only when
                `underlearning_insertion.enabled=True` in config.
            underlearning_new_classes: Optional list of new class indices for
                the current stage. Used only when
                `underlearning_insertion.enabled=True` in config.
            learning_dynamics_forgetness_by_seat: Optional per-seat forgetness
                scores keyed by (scene_id, save_stage) in a JSON-friendly shape:
                  {scene_id: {save_stage: score}}
                Used only when `selection_strategy='learning_dynamics'`.
            learning_dynamics_replay_priority_by_seat: Optional per-seat replay
                priority scores keyed by (scene_id, save_stage) in a JSON-friendly shape:
                  {scene_id: {save_stage: score}}
                Used only when `selection_strategy='learning_dynamics'`.
            learning_dynamics_design1_payload: Optional Design-1 payload with
                class_need + seat_class_terms for
                `selection_strategy='learning_dynamics_design1'`.
            learning_dynamics_design2_payload: Optional Design-2 payload with
                class_need + seat_class_terms for
                `selection_strategy='learning_dynamics_design2'`.
        """
        if self.debug_mode:
            print(f"\n📝 [DEBUG] Adding scenes from stage {stage_id}")
            print(f"   - Input scenes: {len(scene_infos)}")
            print(f"   - Seen classes at save: {seen_classes}")
        
        # Get mappings
        nyu40_to_model = mappings.get('nyu40_to_model_idx', {})
        valid_nyu40_ids = mappings.get('valid_nyu40_ids', [])
        model_to_name = mappings.get('model_idx_to_name', {})
        
        # Process each scene
        scenes_added = 0
        scenes_by_class = defaultdict(list)
        candidate_scenes = []  # For global budget selection
        
        for scene_info in scene_infos:
            # Extract scene ID from the nested structure
            if 'point_cloud' in scene_info and 'lidar_idx' in scene_info['point_cloud']:
                scene_id = scene_info['point_cloud']['lidar_idx']
            else:
                scene_id = scene_info.get('sample_idx', scene_info.get('scene_id', 'unknown'))
            
            # Skip if no annotations
            if scene_info['annos']['gt_num'] == 0:
                continue
            
            # Filter labels to only seen classes
            filtered_info, present_classes = self._filter_scene_labels(
                scene_info, seen_classes, nyu40_to_model, valid_nyu40_ids
            )
            
            # Skip if too few objects after filtering
            if filtered_info['annos']['gt_num'] < self.min_objects_per_scene:
                # Track rejection reason for debugging
                self.scene_selection_stats[stage_id][scene_id] = {
                    'status': 'rejected',
                    'reason': 'insufficient_objects',
                    'objects_after_filtering': filtered_info['annos']['gt_num'],
                    'min_required': self.min_objects_per_scene,
                    'present_classes': present_classes
                }
                # Only show first few skips to avoid cluttering output
                if self.debug_mode and scenes_added < 10:
                    print(f"   Skipping {scene_id}: only {filtered_info['annos']['gt_num']} objects after filtering")
                continue
            
            # Create scene snapshot
            snapshot = {
                'data_info': filtered_info,
                'seen_classes': seen_classes.copy(),
                'present_classes': present_classes,
                'save_stage': stage_id,
                'save_timestamp': time.time(),
                'object_counts': self._count_objects_by_class(filtered_info, nyu40_to_model, valid_nyu40_ids),
            }
            
            if self.use_legacy_mode:
                # Legacy mode keeps a full candidate cache.
                if scene_id not in self.scene_snapshots:
                    self.scene_snapshots[scene_id] = {}
                self.scene_snapshots[scene_id][stage_id] = snapshot
                self.stage_scenes[stage_id].append(scene_id)

                # Track by class for legacy per-class replay selection.
                for class_id in present_classes:
                    self.class_scenes[class_id].append((scene_id, stage_id))
                    scenes_by_class[class_id].append(scene_id)
            
            # For global budget mode, collect candidates for scoring and selection
            if not self.use_legacy_mode:
                metrics_for_scene = None
                if scene_metrics is not None:
                    metrics_for_scene = scene_metrics.get(scene_id)
                scene_candidate = {
                    'scene_id': scene_id,
                    'snapshot': snapshot,
                    'present_classes': present_classes,
                    'stage_id': stage_id
                }
                if metrics_for_scene is not None:
                    scene_candidate['metrics'] = metrics_for_scene
                candidate_scenes.append(scene_candidate)
            
            # Track acceptance for debugging
            self.scene_selection_stats[stage_id][scene_id] = {
                'status': 'candidate' if not self.use_legacy_mode else 'accepted',
                'reason': 'sufficient_objects',
                'objects_after_filtering': filtered_info['annos']['gt_num'],
                'present_classes': present_classes,
                'object_counts': snapshot['object_counts']
            }
            
            scenes_added += 1
            
            if self.debug_mode and scenes_added <= 3:  # Show first 3 scenes
                print(f"   {'Candidate' if not self.use_legacy_mode else 'Added'} scene {scene_id}:")
                print(f"      - Objects: {filtered_info['annos']['gt_num']}")
                print(f"      - Classes present: {present_classes}")
                print(f"      - Per-class counts: {snapshot['object_counts']}")

        # === Stage-ratio quota update (SUNRGBD baseline + learning dynamics) ===
        # This path is intentionally simple and deterministic where possible.
        if (not self.use_legacy_mode and self.quota_strategy == 'stage_ratio'):
            # Track how many *seats* from this stage existed before (usually 0).
            prev_stage_count = 0
            for _, sdata in self.memory_scenes.items():
                try:
                    prev_stage_count += int(
                        int(stage_id) in (sdata.get('stages', {}) or {})
                    )
                except Exception:
                    continue

            eviction_report = None
            ld_report = None
            ld_design_report = None
            if self.selection_strategy == 'learning_dynamics':
                selected_scenes, quotas, actual, ld_report = (
                    self._apply_stage_ratio_update_learning_dynamics(
                        stage_id=stage_id,
                        candidate_scenes=candidate_scenes,
                        forgetness_by_seat=learning_dynamics_forgetness_by_seat,
                        replay_priority_by_seat=learning_dynamics_replay_priority_by_seat,
                    )
                )
            elif self.selection_strategy in (
                    LD_DESIGN1_STRATEGY,
                    LD_DESIGN2_STRATEGY):
                design_payload = (
                    learning_dynamics_design1_payload
                    if self.selection_strategy == LD_DESIGN1_STRATEGY
                    else learning_dynamics_design2_payload
                )
                selected_scenes, quotas, actual, ld_design_report = (
                    self._apply_stage_ratio_update_learning_dynamics_design1(
                        stage_id=stage_id,
                        candidate_scenes=candidate_scenes,
                        learning_dynamics_design1_payload=design_payload,
                    )
                )
            elif self.selection_strategy == 'random' and self.forgetness_eviction_enabled and int(stage_id) > 1:
                assert forgetness_class_drops is not None, (
                    "forgetness_eviction.enabled=True requires forgetness_class_drops "
                    "to be passed to add_stage_scenes() for stage_id>1."
                )
                selected_scenes, quotas, actual, eviction_report = (
                    self._apply_random_stage_ratio_update_forgetness_eviction(
                        stage_id=stage_id,
                        candidate_scenes=candidate_scenes,
                        forgetness_class_drops=forgetness_class_drops,
                        underlearning_class_ap=underlearning_class_ap,
                        underlearning_new_classes=underlearning_new_classes,
                    )
                )
            elif self.selection_strategy == 'random':
                selected_scenes, quotas, actual = self._apply_random_stage_ratio_update(
                    stage_id=stage_id,
                    candidate_scenes=candidate_scenes,
                    underlearning_class_ap=underlearning_class_ap,
                    underlearning_new_classes=underlearning_new_classes,
                )
            else:
                raise RuntimeError(
                    "quota_strategy='stage_ratio' is only supported for "
                    "selection_strategy in ['random', 'learning_dynamics', "
                    "'learning_dynamics_design1', 'learning_dynamics_design2']."
                )
            scenes_added_to_memory = max(
                0, int(actual.get(int(stage_id), 0)) - int(prev_stage_count)
            )
            actual_pairs = int(sum(int(v) for v in (actual or {}).values()))
            memory_pairs = int(self._count_scene_stage_pairs())
            if actual_pairs != memory_pairs:
                raise RuntimeError(
                    "Stage-ratio composition mismatch between returned stage counts "
                    f"and memory contents at stage_id={int(stage_id)}: "
                    f"actual_sum={actual_pairs}, memory_pairs={memory_pairs}"
                )

            # Persist bank contents (IDs) and a composition report.
            # These artifacts are important for reproducibility; do not silently skip.
            if dataset_ref is not None:
                mb_actions_dir = None
                paths_obj = getattr(dataset_ref, 'paths', None)
                if paths_obj is not None:
                    mb_actions_dir = str(paths_obj.memory_bank_actions_dir())
                elif getattr(dataset_ref, 'experiment_dir', None):
                    mb_actions_dir = os.path.join(
                        str(dataset_ref.experiment_dir), 'memory_bank', 'actions'
                    )
                elif getattr(dataset_ref, 'work_dir', None):
                    mb_actions_dir = os.path.join(
                        str(dataset_ref.work_dir), 'memory_bank', 'actions'
                    )
                else:
                    mb_actions_dir = os.path.join(
                        os.getcwd(), 'memory_bank', 'actions'
                    )

                try:
                    os.makedirs(str(mb_actions_dir), exist_ok=True)
                    ids_path = os.path.join(
                        str(mb_actions_dir),
                        f'memory_bank_selected_scenes_stage_{stage_id}.txt',
                    )
                    comp_path = os.path.join(
                        str(mb_actions_dir),
                        f'memory_bank_composition_stage_{stage_id}.json',
                    )

                    # IDs (one per line): explicit seats for deterministic replay
                    seat_ids = [
                        f"{c.get('scene_id')}_stage{int(c.get('save_stage', -1))}"
                        for c in selected_scenes
                    ]
                    with open(ids_path, 'w') as f:
                        for sid in seat_ids:
                            f.write(f"{sid}\n")

                    # Composition report
                    total_pairs = int(self._count_scene_stage_pairs())
                    quota_range = list(range(1, int(stage_id) + 1))
                    quota_diff = {
                        str(int(s)): int(actual.get(int(s), 0)) - int(quotas.get(int(s), 0))
                        for s in quota_range
                    }
                    quota_ok = all(
                        int(actual.get(int(s), 0)) == int(quotas.get(int(s), 0))
                        for s in quota_range
                    )
                    report = {
                        'stage_id': int(stage_id),
                        'selection_strategy': str(self.selection_strategy),
                        'quota_strategy': str(self.quota_strategy),
                        'random_seed': int(self.random_seed),
                        'memory_budget': int(self.memory_budget),
                        'memory_budget_ratio': float(self.memory_budget_ratio),
                        'total_training_scenes': int(self.total_training_scenes),
                        'stage_scene_counts': {str(int(k)): int(v) for k, v in self.stage_scene_counts.items()},
                        'quotas': {str(int(k)): int(v) for k, v in quotas.items()},
                        'actual_counts': {str(int(k)): int(v) for k, v in actual.items()},
                        'quota_diff': quota_diff,
                        'quota_check_passed': bool(quota_ok),
                        'total_memory_pairs': total_pairs,
                        'total_unique_scenes': int(len(self.memory_scenes)),
                    }
                    if eviction_report is not None:
                        report['forgetness_eviction'] = eviction_report
                    if ld_report is not None:
                        report['learning_dynamics'] = ld_report
                    if ld_design_report is not None:
                        if self.selection_strategy == LD_DESIGN2_STRATEGY:
                            report['learning_dynamics_design2'] = ld_design_report
                        else:
                            report['learning_dynamics_design1'] = ld_design_report
                    if self.underlearning_insertion_enabled:
                        report['underlearning_insertion'] = {
                            'enabled': True,
                            'score_mode': str(self.underlearning_score_mode),
                            'ap_iou_thr': float(self.underlearning_ap_iou_thr),
                            'new_classes': [int(x) for x in (underlearning_new_classes or [])],
                        }
                    with open(comp_path, 'w') as f:
                        json.dump(report, f, indent=2)

                    # Fail fast on stage-ratio quota violations (except when the policy
                    # explicitly does not preserve quotas, e.g. forgetness eviction).
                    if (self.selection_strategy in ('random', 'learning_dynamics')
                            and not (self.selection_strategy == 'random'
                                     and self.forgetness_eviction_enabled
                                     and int(stage_id) > 1)
                            and (not quota_ok)):
                        raise RuntimeError(
                            "Memory bank stage-ratio quotas violated at "
                            f"stage_id={int(stage_id)}: diff={quota_diff}. "
                            f"See composition report: {comp_path}"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to write memory bank artifacts under memory_bank/actions "
                        f"for stage_id={int(stage_id)}: {e}"
                    ) from e

            if self.debug_mode:
                if self.selection_strategy == 'learning_dynamics':
                    print("Learning-dynamics stage-ratio selection:")
                elif self.selection_strategy == LD_DESIGN2_STRATEGY:
                    print("Learning-dynamics Design-2 stage-ratio selection:")
                elif self.selection_strategy == LD_DESIGN1_STRATEGY:
                    print("Learning-dynamics Design-1 stage-ratio selection:")
                else:
                    print("Random stage-ratio selection:")
                print(f"   - Candidates (stage {stage_id}): {len(candidate_scenes)}")
                print(f"   - Quotas: {quotas}")
                print(f"   - Actual: {actual}")
                if eviction_report is not None:
                    print(
                        "   - Forgetness eviction: "
                        f"evicted={int(eviction_report.get('evicted_count', 0))}, "
                        f"added={int(eviction_report.get('current_stage_added', 0))}, "
                        f"score_mode={eviction_report.get('forgetness_score_mode', None)}"
                    )
                if ld_design_report is not None:
                    print(
                        "   - Design ratio target: "
                        f"target={int(ld_design_report.get('current_stage_target', 0))}, "
                        f"existing={int(ld_design_report.get('current_stage_existing', 0))}, "
                        f"required_add={int(ld_design_report.get('required_add_t', 0))}, "
                        f"added={int(ld_design_report.get('current_stage_added', 0))}, "
                        f"shortfall={int(ld_design_report.get('shortfall_t', 0))}, "
                        f"exact_match={bool(ld_design_report.get('exact_target_match', False))}, "
                        f"use_class_balance={bool(ld_design_report.get('use_class_balance', False))}"
                    )
                    ratio_report = ld_design_report.get('ratio_targeted_update', {}) or {}
                    if ratio_report:
                        print(
                            "   - Design-1 eviction plan: "
                            f"evict_total={int(ratio_report.get('evict_total', 0))}, "
                            f"plan={ratio_report.get('eviction_plan', {})}, "
                            f"actual={ratio_report.get('eviction_actual', {})}"
                        )
                print(
                    f"   - Memory usage: {self._count_scene_stage_pairs()}/{self.memory_budget} "
                    f"pairs (unique scenes: {len(self.memory_scenes)})"
                )

            # Update statistics and validate, then skip the default score-based path.
            self._sync_active_stage_stats()
            self.total_scenes_stored = self._count_total_unique_scenes()

            if self.debug_mode:
                print(f"Stage {stage_id} scene addition summary:")
                print(f"   - Scenes added to memory bank: {scenes_added_to_memory}")
                print(f"   - Total unique scenes stored: {self.total_scenes_stored}")
                print(f"   - Scenes with multiple snapshots: {self._count_multi_snapshot_scenes()}")
                current_scene_stage_pairs = self._count_scene_stage_pairs()
                print(f"   - Memory bank utilization: {current_scene_stage_pairs}/{self.memory_budget} scene-stage pairs ({current_scene_stage_pairs/self.memory_budget*100:.1f}%)")
                print(f"   - Unique scenes: {len(self.memory_scenes)}")
                print(f"   - Scenes per stage so far: {dict(self.scenes_per_stage)}")
                self._print_class_coverage(model_to_name)

            scene_stage_pairs = self._count_scene_stage_pairs()
            unique_scene_count = len(self.memory_scenes)
            if scene_stage_pairs > self.memory_budget:
                print(f"WARNING: Memory bank contains {scene_stage_pairs} scene-stage pairs, "
                      f"exceeding budget of {self.memory_budget} (unique scenes: {unique_scene_count})")

            return
        
        # CRITICAL FIX: Add existing memory bank scenes as candidates
        # This ensures existing scenes compete with new scenes for memory slots
        if not self.use_legacy_mode and stage_id > 1:  # Only for Stage 2+
            existing_candidates_added = 0
            if self.debug_mode:
                print(f"   📋 Adding existing memory bank scenes as candidates...")
            
            for scene_id, scene_data in self.memory_scenes.items():
                for existing_stage_id, stage_info in scene_data['stages'].items():
                    # Add existing scene-stage pair as candidate
                    metrics_for_scene = None
                    if scene_metrics is not None:
                        metrics_for_scene = scene_metrics.get(scene_id)
                    existing_candidate = {
                        'scene_id': scene_id,
                        'snapshot': stage_info['snapshot'],
                        'present_classes': stage_info['snapshot']['present_classes'],
                        'stage_id': existing_stage_id  # Preserve original stage
                    }
                    if metrics_for_scene is not None:
                        existing_candidate['metrics'] = metrics_for_scene
                    candidate_scenes.append(existing_candidate)
                    existing_candidates_added += 1
            
            if self.debug_mode:
                print(f"   📋 Added {existing_candidates_added} existing scene-stage pairs as candidates")
                print(f"   📋 Total candidates: {len(candidate_scenes)} (new: {len(candidate_scenes) - existing_candidates_added}, existing: {existing_candidates_added})")
        
        # CRITICAL FIX: Clear existing memory bank since we're re-selecting from all candidates
        if not self.use_legacy_mode and stage_id > 1:
            if self.debug_mode:
                old_count = len(self.memory_scenes)
                print(f"   🗑️ Clearing existing memory bank ({old_count} scenes) for fresh selection...")
            self.memory_scenes.clear()
            self.class_distribution.clear()
            self.class_object_counts.clear()
        
        # Apply selection strategy based on mode
        scenes_added_to_memory = 0  # Track scenes actually added to memory bank
        
        if self.use_legacy_mode:
            # Legacy mode: select best scenes per class
            self._select_best_scenes_per_class(stage_id, scenes_by_class, model_to_name)
            # In legacy mode, scenes_added is the count of candidates processed
            scenes_added_to_memory = scenes_added
        else:
            # Global budget mode: select scenes within budget
            if (self.forced_scene_list is not None and
                    self.selection_strategy == 'forced'):
                # Discovery system mode: use forced scene list
                selected_scenes = self._select_forced_scenes(
                    candidate_scenes, stage_id
                )
            elif self.selection_strategy in (
                    'diversity_only',
                    'uncertainty_diversity_combined'):
                # Specialized selection using uncertainty/diversity metrics
                selected_scenes = self._select_scenes_uncertainty_diversity(
                    candidate_scenes, stage_id
                )
            else:
                # Normal selection within budget
                selected_scenes = self._select_scenes_within_budget(
                    candidate_scenes, stage_id
                )
            scenes_added_to_memory = len(selected_scenes)

            # Optionally persist selected scene IDs for this stage (best-effort).
            if dataset_ref is not None:
                try:
                    mb_actions_dir = None
                    paths_obj = getattr(dataset_ref, 'paths', None)
                    if paths_obj is not None:
                        mb_actions_dir = paths_obj.memory_bank_actions_dir()
                    elif getattr(dataset_ref, 'work_dir', None):
                        mb_actions_dir = os.path.join(
                            str(dataset_ref.work_dir),
                            'memory_bank',
                            'actions',
                        )

                    if mb_actions_dir is None:
                        raise RuntimeError("memory_bank/actions dir resolution failed.")

                    if hasattr(mb_actions_dir, 'mkdir'):
                        mb_actions_dir.mkdir(parents=True, exist_ok=True)
                        out_path = mb_actions_dir / f'memory_bank_selected_scenes_stage_{stage_id}.txt'
                    else:
                        os.makedirs(str(mb_actions_dir), exist_ok=True)
                        out_path = os.path.join(
                            str(mb_actions_dir),
                            f'memory_bank_selected_scenes_stage_{stage_id}.txt',
                        )

                    seat_ids = [
                        f"{c.get('scene_id')}_stage{int((c.get('snapshot', {}) or {}).get('save_stage', stage_id))}"
                        for c in selected_scenes
                    ]
                    if hasattr(out_path, 'open'):
                        with out_path.open('w') as f:
                            for sid in seat_ids:
                                f.write(f"{sid}\n")
                    else:
                        with open(out_path, 'w') as f:
                            for sid in seat_ids:
                                f.write(f"{sid}\n")
                except Exception:
                    # ID logging is best-effort only; never break training.
                    pass
            
            if self.debug_mode:
                print(f"Global budget selection:")
                print(f"   - Candidates: {len(candidate_scenes)}")
                print(f"   - Selected for memory bank: {len(selected_scenes)}")
                print(f"   - Memory usage: {len(self.memory_scenes)}/{self.memory_budget}")
            scenes_added_to_memory = len(selected_scenes)
        
        # Update statistics.
        if self.use_legacy_mode:
            self.total_scenes_stored = self._count_total_unique_scenes()
            self.scenes_per_stage[stage_id] = int(scenes_added_to_memory)
        else:
            self._sync_active_stage_stats()
            self.total_scenes_stored = self._count_total_unique_scenes()
        
        if self.debug_mode:
            print(f"Stage {stage_id} scene addition summary:")
            print(f"   - Scenes added to memory bank: {scenes_added_to_memory}")
            print(f"   - Total unique scenes stored: {self.total_scenes_stored}")
            print(f"   - Scenes with multiple snapshots: {self._count_multi_snapshot_scenes()}")
            if not self.use_legacy_mode:
                # Count scene-stage pairs (not unique scenes) for budget utilization
                current_scene_stage_pairs = self._count_scene_stage_pairs()
                print(f"   - Memory bank utilization: {current_scene_stage_pairs}/{self.memory_budget} scene-stage pairs ({current_scene_stage_pairs/self.memory_budget*100:.1f}%)")
                print(f"   - Unique scenes: {len(self.memory_scenes)}")
                print(f"   - Scenes per stage so far: {dict(self.scenes_per_stage)}")
            self._print_class_coverage(model_to_name)
        
        # Validation: check memory bank health (with new nested structure)
        if not self.use_legacy_mode:
            # Count scene-stage pairs for budget compliance
            scene_stage_pairs = self._count_scene_stage_pairs()
            unique_scene_count = len(self.memory_scenes)
            
            # Check budget compliance
            if scene_stage_pairs > self.memory_budget:
                print(f"WARNING: Memory bank contains {scene_stage_pairs} scene-stage pairs, "
                      f"exceeding budget of {self.memory_budget} (unique scenes: {unique_scene_count})")
            
            # Optional: Check for scenes with excessive stage snapshots
            multi_stage_scenes = []
            for scene_id, scene_data in self.memory_scenes.items():
                stage_count = len(scene_data['stages'])
                if stage_count > 2:  # More than 2 stages might indicate an issue
                    multi_stage_scenes.append((scene_id, stage_count))
            
            if multi_stage_scenes and self.debug_mode:
                print(f"Scenes with multiple stage snapshots:")
                for scene_id, count in multi_stage_scenes[:5]:
                    stages = list(self.memory_scenes[scene_id]['stages'].keys())
                    print(f"   - {scene_id}: {count} stages {stages}")
                if len(multi_stage_scenes) > 5:
                    print(f"   ... and {len(multi_stage_scenes) - 5} more")
    
    def get_replay_scenes(self, 
                         previous_classes: List[int],
                         current_stage: int) -> Tuple[List[Dict], set]:
        """Get scenes for replay from previous stages.
        
        Args:
            previous_classes: Classes from previous stages to replay
            current_stage: Current training stage
            
        Returns:
            (replay_scenes, replay_scene_ids): List of scene infos and set of IDs
        """
        if self.debug_mode:
            print(f"\n🔄 [DEBUG] Getting replay scenes for stage {current_stage}")
            print(f"   - Previous classes to replay: {previous_classes}")
            if not self.use_legacy_mode:
                print(f"   - Memory bank scenes: {len(self.memory_scenes)}")
                print(f"   - Memory budget: {self.memory_budget}")
                print(f"   - Total snapshots: {self._count_total_snapshots()}")
            else:
                print(f"   - Legacy mode active")
        
        replay_scenes = []
        used_scene_ids = set()
        scenes_per_class_actual = defaultdict(int)
        
        if self.use_legacy_mode:
            # Legacy mode: get scenes per class
            for class_id in previous_classes:
                if class_id not in self.class_scenes:
                    if self.debug_mode:
                        print(f"   No scenes found for class {class_id}")
                    continue
                
                # Get candidate scenes for this class
                class_scene_refs = self.class_scenes[class_id]
                
                # Select best scenes based on strategy
                selected_refs = self._select_scenes_for_class(
                    class_id, class_scene_refs, current_stage
                )
                
                for scene_id, stage_id in selected_refs:
                    # Skip if already added
                    unique_id = f"{scene_id}_stage{stage_id}"
                    if unique_id in used_scene_ids:
                        continue
                    
                    # Get the snapshot
                    snapshot = self.scene_snapshots[scene_id][stage_id]
                    
                    # Create replay scene info
                    replay_info = copy.deepcopy(snapshot['data_info'])
                    replay_info['is_replay'] = True
                    replay_info['replay_from_stage'] = stage_id
                    replay_info['replay_unique_id'] = unique_id
                    replay_info['original_scene_id'] = scene_id
                    
                    replay_scenes.append(replay_info)
                    used_scene_ids.add(unique_id)
                    scenes_per_class_actual[class_id] += 1
        else:
            # Global budget mode: get all scene-stage *seats* from memory bank that contain
            # previous classes (multi-seat semantics).
            for scene_id, scene_data in self.memory_scenes.items():
                previous_classes_set = set(previous_classes)

                for saved_stage, stage_data in (scene_data.get('stages', {}) or {}).items():
                    try:
                        saved_stage_i = int(saved_stage)
                    except Exception:
                        continue
                    if saved_stage_i >= int(current_stage):
                        continue
                    snapshot = stage_data.get('snapshot', {}) or {}
                    seat_classes = set(snapshot.get('present_classes', []) or [])
                    if not seat_classes.intersection(previous_classes_set):
                        continue

                    unique_id = f"{scene_id}_stage{saved_stage_i}"
                    replay_info = copy.deepcopy(snapshot.get('data_info', {}) or {})
                    replay_info['is_replay'] = True
                    replay_info['replay_from_stage'] = int(snapshot.get('save_stage', saved_stage_i))
                    replay_info['replay_unique_id'] = unique_id
                    replay_info['original_scene_id'] = str(scene_id)

                    replay_scenes.append(replay_info)
                    used_scene_ids.add(unique_id)

                    for class_id in seat_classes.intersection(previous_classes_set):
                        scenes_per_class_actual[int(class_id)] += 1
            
            # Apply importance decay only when importance is action-driving.
            # In stage-ratio modes (`random` / `learning_dynamics`), importance
            # is a placeholder and must stay constant (1.0).
            if not self._importance_placeholder_mode():
                for scene_id, scene_data in self.memory_scenes.items():
                    for stage_id, stage_data in scene_data['stages'].items():
                        stage_data['importance'] *= self.importance_decay
                    # Recalculate total importance
                    scene_data['total_importance'] = sum(stage_data['importance'] 
                                                       for stage_data in scene_data['stages'].values())
        
        if self.debug_mode:
            print(f"\n📦 [DEBUG] Replay scenes prepared:")
            print(f"   - Total replay scenes: {len(replay_scenes)}")
            print(f"   - Unique replay seats: {len(used_scene_ids)}")
            if self.use_legacy_mode:
                print(f"   - Scenes per class: {dict(scenes_per_class_actual)}")
            else:
                print(f"   - Classes covered: {len(scenes_per_class_actual)}")
                print(f"   - Class distribution: {dict(scenes_per_class_actual)}")
            
            # Show first 3 replay scenes
            for i, scene in enumerate(replay_scenes[:3]):
                print(f"   📄 [DEBUG] Replay scene {i+1}:")
                print(f"      - Original ID: {scene['original_scene_id']}")
                print(f"      - From stage: {scene['replay_from_stage']}")
                print(f"      - Objects: {scene['annos']['gt_num']}")
        
        return replay_scenes, used_scene_ids
    
    def _filter_scene_labels(self, 
                            scene_info: Dict,
                            allowed_classes: List[int],
                            nyu40_to_model: Dict,
                            valid_nyu40_ids: List[int]) -> Tuple[Dict, List[int]]:
        """Filter scene annotations to only include allowed classes.
        
        This is CRITICAL for preventing label leakage in incremental learning.
        """
        filtered_info = copy.deepcopy(scene_info)

        annos = filtered_info.get('annos', {})
        if not isinstance(annos, dict) or 'class' not in annos:
            return filtered_info, []

        # Detect label space:
        # - ScanNet: annos['class'] are NYU40 ids and mappings provide nyu40_to_model + valid_nyu40_ids
        # - SUNRGBD: annos['class'] are model indices directly (no NYU40 mapping in mappings)
        use_nyu40 = bool(nyu40_to_model) and bool(valid_nyu40_ids)

        labels_raw = np.asarray(annos['class']).astype(np.int64)
        original_count = int(labels_raw.shape[0])

        if original_count <= 0:
            annos['gt_num'] = 0
            return filtered_info, []

        keep_mask = np.zeros(original_count, dtype=bool)
        present_classes = set()

        if use_nyu40:
            for i, nyu40_id in enumerate(labels_raw):
                if nyu40_id in valid_nyu40_ids and nyu40_id in nyu40_to_model:
                    model_id = int(nyu40_to_model[nyu40_id])
                    if model_id in allowed_classes:
                        keep_mask[i] = True
                        present_classes.add(model_id)
        else:
            allowed = np.array([int(x) for x in allowed_classes], dtype=np.int64)
            keep_mask = np.isin(labels_raw, allowed)
            present_classes.update([int(x) for x in labels_raw[keep_mask].tolist()])

        filtered_count = int(keep_mask.sum())

        # Filter all per-object anno fields consistently (numpy arrays / lists aligned with len(class))
        idxs = np.where(keep_mask)[0].tolist()
        new_annos = {}
        for key, value in annos.items():
            if key == 'gt_num':
                continue
            if isinstance(value, np.ndarray) and value.shape[0] == original_count:
                new_annos[key] = value[keep_mask]
            elif isinstance(value, list) and len(value) == original_count:
                new_annos[key] = [value[i] for i in idxs]
            else:
                new_annos[key] = value

        # Ensure class field is preserved in the dataset's native label space
        # - ScanNet: keep NYU40 ids
        # - SUNRGBD: keep model indices
        new_annos['class'] = labels_raw[keep_mask]
        new_annos['gt_num'] = int(filtered_count)
        new_annos['index'] = np.arange(int(filtered_count), dtype=np.int32)
        filtered_info['annos'] = new_annos

        # Extract scene ID consistently
        if 'point_cloud' in scene_info and 'lidar_idx' in scene_info['point_cloud']:
            scene_id = scene_info['point_cloud']['lidar_idx']
        else:
            scene_id = scene_info.get('sample_idx', 'unknown')
        
        # Enhanced debugging: track which specific classes were filtered
        original_class_counts = {}
        filtered_class_counts = {}
        removed_class_counts = {}
        
        if use_nyu40:
            for i, nyu40_id in enumerate(labels_raw):
                if nyu40_id in valid_nyu40_ids and nyu40_id in nyu40_to_model:
                    model_id = int(nyu40_to_model[nyu40_id])
                    original_class_counts[model_id] = original_class_counts.get(model_id, 0) + 1

                    if keep_mask[i]:
                        filtered_class_counts[model_id] = filtered_class_counts.get(model_id, 0) + 1
                    else:
                        removed_class_counts[model_id] = removed_class_counts.get(model_id, 0) + 1
        else:
            for i, model_id in enumerate(labels_raw):
                model_id = int(model_id)
                original_class_counts[model_id] = original_class_counts.get(model_id, 0) + 1
                if keep_mask[i]:
                    filtered_class_counts[model_id] = filtered_class_counts.get(model_id, 0) + 1
                else:
                    removed_class_counts[model_id] = removed_class_counts.get(model_id, 0) + 1
        
        # Store detailed filtering stats for debugging
        # FIXED: Store the complete allowed_classes (cumulative classes up to current stage)
        self.class_filtering_stats[scene_id] = {
            'original_classes': original_class_counts,
            'kept_classes': filtered_class_counts,
            'removed_classes': removed_class_counts,
            'allowed_classes': list(allowed_classes),  # This is the cumulative list
            'objects_before': original_count,
            'objects_after': filtered_count
        }
        
        self.label_filtering_stats[scene_id]['before'] += original_count
        self.label_filtering_stats[scene_id]['after'] += filtered_count
        
        if self.debug_mode and original_count != filtered_count:
            print(f"      Label filtering: {original_count} -> {filtered_count} objects")
            if removed_class_counts:
                print(f"      🚫 [DEBUG] Removed future classes: {removed_class_counts}")
        
        return filtered_info, list(present_classes)
    
    def _count_objects_by_class(self, 
                               scene_info: Dict,
                               nyu40_to_model: Dict,
                               valid_nyu40_ids: List[int]) -> Dict[int, int]:
        """Count objects per class in the scene."""
        counts = defaultdict(int)

        annos = scene_info.get('annos', {})
        if not isinstance(annos, dict) or 'class' not in annos:
            return {}

        labels_raw = np.asarray(annos['class']).astype(np.int64)
        use_nyu40 = bool(nyu40_to_model) and bool(valid_nyu40_ids)

        if use_nyu40:
            for nyu40_id in labels_raw.tolist():
                if nyu40_id in valid_nyu40_ids and nyu40_id in nyu40_to_model:
                    model_id = int(nyu40_to_model[nyu40_id])
                    counts[model_id] += 1
        else:
            for model_id in labels_raw.tolist():
                counts[int(model_id)] += 1
        
        return dict(counts)
    
    def _select_scenes_for_class(self,
                                class_id: int,
                                scene_refs: List[Tuple[str, int]],
                                current_stage: int) -> List[Tuple[str, int]]:
        """Select best scenes for a class based on strategy."""
        if self.selection_strategy == 'random':
            # Random selection
            import random
            selected = random.sample(scene_refs, 
                                   min(len(scene_refs), self.scenes_per_class))
        
        elif self.selection_strategy == 'balanced':
            # Prefer scenes with balanced class representation
            scored_refs = []
            for scene_id, stage_id in scene_refs:
                snapshot = self.scene_snapshots[scene_id][stage_id]
                
                # Score based on:
                # 1. Object count for this class
                # 2. Total objects (prefer moderate density)
                # 3. Stage recency (prefer more recent stages)
                class_count = snapshot['object_counts'].get(class_id, 0)
                total_count = snapshot['data_info']['annos']['gt_num']
                stage_score = stage_id / current_stage  # Higher for recent stages
                
                score = (class_count * 2 +  # Weight class presence
                        min(total_count / 10, 2) +  # Moderate density preferred
                        stage_score)  # Slight recency bias
                
                scored_refs.append((score, (scene_id, stage_id)))
            
            # Sort by score and select top
            scored_refs.sort(key=lambda x: x[0], reverse=True)
            selected = [ref for _, ref in scored_refs[:self.scenes_per_class]]
        
        elif self.selection_strategy == 'diversity':
            # Prefer diverse scenes (different scene types)
            # For now, use random sampling as proxy for diversity
            # Could be enhanced with actual scene type analysis
            import random
            selected = random.sample(scene_refs, 
                                   min(len(scene_refs), self.scenes_per_class))
        
        else:
            # Default to first N scenes
            selected = scene_refs[:self.scenes_per_class]
        
        return selected
    
    def _select_scenes_within_budget(self, 
                                    candidate_scenes: List[Dict], 
                                    stage_id: int) -> List[Dict]:
        """Select scenes within global budget using importance scoring."""
        if not candidate_scenes:
            return []
        
        # Compute importance scores for all candidates
        scored_candidates = []
        if self.debug_mode and self.selection_strategy == 'precomputed':
            self._debug_score_count = 0  # Initialize debug counter
        
        scoring_debug = []
        for candidate in candidate_scenes:
            importance_or_tuple = self._compute_scene_importance(candidate, stage_id)
            if isinstance(importance_or_tuple, tuple):
                importance, dbg = importance_or_tuple
                if dbg is not None:
                    scoring_debug.append(dbg)
            else:
                importance = importance_or_tuple
            scored_candidates.append((importance, candidate))
        
        # Sort by importance (highest first)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # Select scenes to exactly fill the budget
        # We select until we have exactly self.memory_budget scene-stage pairs
        selected_scenes = []

        # Enforce minimum quota for current stage snapshots, if enabled
        added_stage_t = 0
        stage_quota = 0
        if getattr(self, 'enforce_current_quota', True) and not self.use_legacy_mode:
            try:
                stage_quota = max(0, int(getattr(self, 'min_current_stage_quota', 0)))
            except Exception:
                stage_quota = 0

        if stage_quota > 0:
            stage_t_candidates = [(imp, cand) for (imp, cand) in scored_candidates
                                  if int(cand['snapshot'].get('save_stage', -1)) == int(stage_id)]
            for importance, candidate in stage_t_candidates:
                if added_stage_t >= stage_quota:
                    break
                # Avoid duplicates
                scene_id = candidate['scene_id']
                save_stage = candidate['snapshot']['save_stage']
                if scene_id in self.memory_scenes and save_stage in self.memory_scenes[scene_id]['stages']:
                    continue
                # Check budget
                if self._count_scene_stage_pairs() >= self.memory_budget:
                    break
                # Add candidate
                self._add_scene_to_memory(scene_id, save_stage, candidate['snapshot'], importance)
                for class_id in candidate['present_classes']:
                    self.class_distribution[class_id] += 1
                    obj_count = candidate['snapshot']['object_counts'].get(class_id, 0)
                    self.class_object_counts[class_id] += obj_count
                selected_scenes.append(candidate)
                if stage_id not in self.scene_selection_stats:
                    self.scene_selection_stats[stage_id] = {}
                if candidate['scene_id'] not in self.scene_selection_stats[stage_id]:
                    self.scene_selection_stats[stage_id][candidate['scene_id']] = {}
                self.scene_selection_stats[stage_id][candidate['scene_id']]['status'] = 'selected_memory_quota'
                self.scene_selection_stats[stage_id][candidate['scene_id']]['importance_score'] = importance
                added_stage_t += 1

        # Fill remaining slots by importance order
        for importance, candidate in scored_candidates:
            scene_id = candidate['scene_id']
            
            # Check if this would be a new scene-stage pair (avoid duplicates)
            if scene_id in self.memory_scenes and stage_id in self.memory_scenes[scene_id]['stages']:
                continue  # Skip if this exact scene-stage pair already exists
            
            # Check if we have budget for one more scene-stage pair
            current_scene_stage_pairs = self._count_scene_stage_pairs()
            if current_scene_stage_pairs >= self.memory_budget:
                break  # Budget is full
            
            # Add the scene (either new scene or additional stage to existing scene)
            # CRITICAL FIX: Use the stage where the scene was originally saved, not current stage
            save_stage = candidate['snapshot']['save_stage']
            self._add_scene_to_memory(scene_id, save_stage, candidate['snapshot'], importance)
            
            # Update class distribution
            for class_id in candidate['present_classes']:
                self.class_distribution[class_id] += 1
                obj_count = candidate['snapshot']['object_counts'].get(class_id, 0)
                self.class_object_counts[class_id] += obj_count
            
            selected_scenes.append(candidate)
            
            # Update selection stats
            if stage_id not in self.scene_selection_stats:
                self.scene_selection_stats[stage_id] = {}
            if candidate['scene_id'] not in self.scene_selection_stats[stage_id]:
                self.scene_selection_stats[stage_id][candidate['scene_id']] = {}
            self.scene_selection_stats[stage_id][candidate['scene_id']]['status'] = 'selected_memory'
            self.scene_selection_stats[stage_id][candidate['scene_id']]['importance_score'] = importance
        
        if self.debug_mode:
            print(f"   💡 [DEBUG] Scene importance scores (top 5):")
            for i, (importance, candidate) in enumerate(scored_candidates[:5]):
                print(f"      {i+1}. {candidate['scene_id']}: {importance:.3f}")
            # Store debug details to be saved alongside state
            try:
                self._last_scoring_details = {
                    'stage_id': stage_id,
                    'top_scores': [
                        {
                            'rank': i + 1,
                            'scene_id': cand['scene_id'],
                            'save_stage': cand['snapshot'].get('save_stage'),
                            'importance': float(imp)
                        }
                        for i, (imp, cand) in enumerate(scored_candidates[:50])
                    ],
                    'components': scoring_debug[:50]
                }
            except Exception:
                self._last_scoring_details = None

        return selected_scenes

    def _select_scenes_uncertainty_diversity(self,
                                             candidate_scenes: List[Dict],
                                             stage_id: int) -> List[Dict]:
        """Select scenes using uncertainty/diversity-aware strategies.

        This method supports:
        - selection_strategy == 'diversity_only'
        - selection_strategy == 'uncertainty_diversity_combined'

        It expects per-scene metrics to be attached to candidates under the
        'metrics' key by add_stage_scenes (see scene_metrics argument).
        """
        if not candidate_scenes:
            return []

        import math
        from collections import defaultdict as _dd

        def _build_hist(hist_dict: Optional[Dict[int, int]]) -> Dict[int, int]:
            if not hist_dict:
                return {}
            return {int(k): int(v) for k, v in hist_dict.items()}

        def _j_diversity(hist: Dict[int, int]) -> float:
            """Entropy + beta * coverage for a single histogram."""
            total = sum(hist.values())
            if total <= 0:
                return 0.0
            probs = [v / float(total) for v in hist.values() if v > 0]
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
            return float(entropy + self.diversity_beta * float(total))

        # Prepare candidate info with uncertainty and diversity histograms
        prepared = []
        for cand in candidate_scenes:
            metrics = cand.get('metrics') or {}
            unc_metrics = metrics.get('uncertainty') or {}
            s_unc = unc_metrics.get('S_unc', None)
            div_hist = _build_hist(metrics.get('diversity_hist'))
            if not div_hist:
                # Fallback: use per-class object counts as diversity proxy
                div_hist = _build_hist(cand['snapshot'].get('object_counts'))
            prepared.append({
                'candidate': cand,
                's_unc': None if s_unc is None else float(s_unc),
                'div_hist': div_hist,
                'div_score': _j_diversity(div_hist),
            })

        selected: List[Dict] = []
        used_pairs = set()  # (scene_id, save_stage)

        # Helper to add a scene to memory bank and bookkeeping
        def _add_to_memory(cand: Dict, importance: float):
            scene_id = cand['scene_id']
            save_stage = cand['snapshot']['save_stage']
            self._add_scene_to_memory(scene_id, save_stage,
                                      cand['snapshot'], importance)
            # Update class distribution
            for class_id in cand['present_classes']:
                self.class_distribution[class_id] += 1
                obj_cnt = cand['snapshot']['object_counts'].get(class_id, 0)
                self.class_object_counts[class_id] += obj_cnt
            # Update selection stats
            if stage_id not in self.scene_selection_stats:
                self.scene_selection_stats[stage_id] = {}
            if scene_id not in self.scene_selection_stats[stage_id]:
                self.scene_selection_stats[stage_id][scene_id] = {}
            self.scene_selection_stats[stage_id][scene_id]['status'] = (
                'selected_diversity'
                if self.selection_strategy == 'diversity_only'
                else 'selected_uncertainty_diversity'
            )
            self.scene_selection_stats[stage_id][scene_id]['importance_score'] = float(importance)

        # Diversity-only: greedy maximization of J(M) over candidate set
        if self.selection_strategy == 'diversity_only':
            current_hist: Dict[int, int] = {}
            # Precompute current J for fast gain calculation
            current_j = _j_diversity(current_hist)

            while self._count_scene_stage_pairs() < self.memory_budget:
                best_idx = None
                best_gain = None

                for idx, item in enumerate(prepared):
                    cand = item['candidate']
                    pair_key = (cand['scene_id'], cand['snapshot']['save_stage'])
                    if pair_key in used_pairs:
                        continue

                    # Compute marginal J gain if we add this scene
                    tmp_hist = _dd(int)
                    # Start from current_hist
                    for k, v in current_hist.items():
                        tmp_hist[int(k)] += int(v)
                    # Add scene histogram
                    for k, v in item['div_hist'].items():
                        tmp_hist[int(k)] += int(v)
                    new_j = _j_diversity(tmp_hist)
                    gain = new_j - current_j

                    if best_gain is None or gain > best_gain:
                        best_gain = gain
                        best_idx = idx

                if best_idx is None or best_gain is None:
                    break

                chosen = prepared[best_idx]
                cand = chosen['candidate']
                pair_key = (cand['scene_id'], cand['snapshot']['save_stage'])
                used_pairs.add(pair_key)
                selected.append(cand)

                # Update histogram and J(M)
                for k, v in chosen['div_hist'].items():
                    current_hist[int(k)] = current_hist.get(int(k), 0) + int(v)
                current_j = _j_diversity(current_hist)

                # Use gain as a proxy importance score when recording
                _add_to_memory(cand, importance=float(current_j))

                # Stop if we reached budget
                if self._count_scene_stage_pairs() >= self.memory_budget:
                    break

        else:
            # Combined uncertainty + diversity (per-scene score)
            # Normalize uncertainty and diversity scores separately to [0, 1]
            unc_raw = [
                (item['s_unc'] if item['s_unc'] is not None else 0.0)
                for item in prepared
            ]
            div_raw = [item['div_score'] for item in prepared]

            def _normalize_seq(values: List[float]) -> List[float]:
                if not values:
                    return []
                mean_v = sum(values) / float(len(values))
                var = sum((v - mean_v) ** 2 for v in values) / float(len(values))
                std_v = math.sqrt(max(var, 1e-12))
                normed = []
                for v in values:
                    z = (v - mean_v) / (3.0 * std_v)
                    normed.append(max(0.0, min(1.0, 0.5 + z)))
                return normed

            unc_norm = _normalize_seq(unc_raw)
            div_norm = _normalize_seq(div_raw)

            scored = []
            alpha = float(self.combined_alpha)
            for idx, item in enumerate(prepared):
                cand = item['candidate']
                u_norm = unc_norm[idx] if idx < len(unc_norm) else 0.5
                d_norm = div_norm[idx] if idx < len(div_norm) else 0.5

                # Apply uncertainty focus
                if self.uncertainty_focus == 'low':
                    u_norm = 1.0 - u_norm

                combined = alpha * d_norm + (1.0 - alpha) * u_norm
                scored.append((combined, cand))

            # Sort by combined score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            for combined_score, cand in scored:
                pair_key = (cand['scene_id'], cand['snapshot']['save_stage'])
                if pair_key in used_pairs:
                    continue
                if self._count_scene_stage_pairs() >= self.memory_budget:
                    break

                used_pairs.add(pair_key)
                selected.append(cand)
                _add_to_memory(cand, importance=float(combined_score))

        if self.debug_mode:
            print(f"   🎯 [DEBUG] Uncertainty/diversity selection:")
            print(f"      - Strategy: {self.selection_strategy}")
            print(f"      - Candidates: {len(candidate_scenes)}")
            print(f"      - Selected: {len(selected)}")
            print(f"      - Memory usage: {self._count_scene_stage_pairs()}/{self.memory_budget}")

        return selected
    
    def _select_forced_scenes(self,
                            candidate_scenes: List[Dict],
                            stage_id: int) -> List[Dict]:
        """Select specific scenes from forced scene list for discovery system.
        
        Args:
            candidate_scenes: List of candidate scene dictionaries
            stage_id: Current stage ID
            
        Returns:
            List of selected scenes matching the forced scene list
        """
        if not self.forced_scene_list:
            if self.debug_mode:
                print(f"   💡 [DEBUG] No forced scene list provided, returning empty selection")
            return []
        
        selected_scenes = []
        found_scenes = set()
        
        if self.debug_mode:
            print(f"   🎯 [DEBUG] Forced scene selection mode:")
            print(f"      - Target scenes: {len(self.forced_scene_list)}")
            print(f"      - Available candidates: {len(candidate_scenes)}")
        
        # Find scenes from forced list in candidates
        for candidate in candidate_scenes:
            scene_id = candidate['scene_id']
            
            if scene_id in self.forced_scene_list:
                selected_scenes.append(candidate)
                found_scenes.add(scene_id)
                
                # Add the scene to memory bank
                save_stage = candidate['snapshot']['save_stage']
                importance_score = 1.0  # All forced scenes have equal high importance
                self._add_scene_to_memory(scene_id, save_stage, candidate['snapshot'], importance_score)
                
                if self.debug_mode:
                    print(f"      ✅ Found and selected: {scene_id}")
        
        # Report missing scenes
        missing_scenes = set(self.forced_scene_list) - found_scenes
        if missing_scenes and self.debug_mode:
            print(f"      ⚠️  Scenes not found in candidates: {list(missing_scenes)[:5]}...")
            if len(missing_scenes) > 5:
                print(f"         (and {len(missing_scenes)-5} more)")
        
        if self.debug_mode:
            print(f"      📊 Selection summary: {len(selected_scenes)}/{len(self.forced_scene_list)} forced scenes found")
        
        return selected_scenes
    
    def _load_precomputed_scores(self):
        """Load precomputed scores from unified JSON files.
        
        Only supports unified format: analysis/{criteria}.json
        """
        if not self.score_files_dir:
            raise ValueError("score_files_dir must be specified when using 'precomputed' selection strategy")
        
        if not self.score_criteria:
            raise ValueError("score_criteria must be specified when using 'precomputed' selection strategy")
        
        # Load unified JSON format
        unified_file = os.path.join(self.score_files_dir, f"{self.score_criteria}.json")
        
        if not os.path.exists(unified_file):
            available_files = [f for f in os.listdir(self.score_files_dir) if f.endswith('.json')]
            raise FileNotFoundError(f"Score file not found: {unified_file}\nAvailable files: {available_files}")
        
        if self.debug_mode:
            print(f"   Loading unified scores from: {unified_file}")
        
        try:
            with open(unified_file, 'r') as f:
                data = json.load(f)
            
            if 'scores' not in data:
                raise ValueError(f"Invalid score file format: missing 'scores' key in {unified_file}")
            
            # Determine which stages exist in this score file.
            stage_ids_to_load = []
            try:
                stage_class_map = getattr(self, 'stage_class_map', None)
            except Exception:
                stage_class_map = None

            if isinstance(stage_class_map, dict) and stage_class_map:
                for k in stage_class_map.keys():
                    try:
                        stage_ids_to_load.append(int(k))
                    except Exception:
                        continue
                stage_ids_to_load = sorted(set(stage_ids_to_load))
            else:
                stage_ids = set()
                for scene_data in data.get('scores', {}).values():
                    if not isinstance(scene_data, dict):
                        continue
                    for key in scene_data.keys():
                        if not isinstance(key, str) or not key.startswith('stage_'):
                            continue
                        try:
                            stage_ids.add(int(key.split('_', 1)[1]))
                        except Exception:
                            continue
                stage_ids_to_load = sorted(stage_ids)

            if not stage_ids_to_load:
                raise ValueError(
                    "Could not infer stage ids from score file. "
                    "Provide `stage_class_map` to SceneMemoryBank for deterministic loading."
                )

            loaded_stages = []
            for stage_id in stage_ids_to_load:
                stage_scores = {}
                stage_key = f'stage_{stage_id}'
                
                for scene_id, scene_data in data['scores'].items():
                    if stage_key in scene_data and scene_data[stage_key] != -999:
                        stage_scores[scene_id] = scene_data[stage_key]
                
                if stage_scores:
                    self.precomputed_scores[stage_id] = stage_scores
                    loaded_stages.append(stage_id)
                    if self.debug_mode:
                        print(f"   Loaded scores for stage {stage_id}: {len(stage_scores)} scenes")
            
            if not loaded_stages:
                raise ValueError(f"No valid stages found in score file: {unified_file}")
            
            if self.debug_mode:
                print(f"SceneMemoryBank: Loaded unified scores for stages {loaded_stages}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load scores from {unified_file}: {e}")
    
    def _compute_scene_importance(self, candidate: Dict, current_stage: int,
                                  return_debug: bool = False):
        """Compute importance score for a scene candidate with optional debug breakdown."""
        use_strict_object_count = False
        # Base score: strategy-dependent (precomputed / uncertainty / fallback)
        if self.selection_strategy == 'precomputed':
            scene_id = candidate['scene_id']
            candidate_save_stage = candidate['snapshot']['save_stage']
            if candidate_save_stage in self.precomputed_scores:
                stage_scores = self.precomputed_scores[candidate_save_stage]
                if scene_id in stage_scores:
                    score = stage_scores[scene_id]
                    if self.debug_mode and hasattr(self, '_debug_score_count'):
                        if self._debug_score_count < 5:
                            print(f"      Precomputed score for {scene_id} (from stage {candidate_save_stage}): {score}")
                            self._debug_score_count += 1
                    base_score = float(score)
                else:
                    if self.debug_mode:
                        print(f"      WARNING: No precomputed score for scene {scene_id}, using fallback")
                    base_score = self._fallback_importance(candidate)
            else:
                if self.debug_mode:
                    print(f"      WARNING: No precomputed scores for stage {candidate_save_stage}, using fallback")
                base_score = self._fallback_importance(candidate)
        elif self.selection_strategy == 'uncertainty_only':
            # Use per-scene uncertainty score (if available) as base importance.
            metrics = candidate.get('metrics') or {}
            unc = (metrics.get('uncertainty') or {}).get('S_unc', None)
            if unc is None:
                # Fall back to heuristic importance if no metrics provided
                base_score = self._fallback_importance(candidate)
            else:
                s_unc = float(unc)
                if self.uncertainty_focus == 'low':
                    # Prefer low-uncertainty scenes (clean exemplars)
                    base_score = -s_unc
                else:
                    # Default/high: prefer high-uncertainty scenes (hard examples)
                    base_score = s_unc
        elif self.selection_strategy == 'gt_object_count_desc':
            # Simple heuristic baseline: rank seats by stage-filtered GT object
            # count only (descending), with no auxiliary weighting.
            annos = (
                candidate.get('snapshot', {})
                .get('data_info', {})
                .get('annos', {})
            )
            try:
                base_score = float(annos.get('gt_num', 0))
            except Exception:
                base_score = 0.0
            use_strict_object_count = True
        else:
            base_score = self._fallback_importance(candidate)

        if use_strict_object_count:
            # Keep this strategy as a pure heuristic baseline.
            save_stage = candidate['snapshot'].get('save_stage')
            stage_mult = 1.0
            drop_mult = 1.0
            curr_mult = 1.0
            weighted_need = 0.0
            curr_weighted_need = 0.0
        else:
            # Stage-ratio multiplier (normalized to mean 1)
            save_stage = candidate['snapshot'].get('save_stage')
            stage_mult = self._stage_ratio_weights.get(save_stage, 1.0)

            # Drop/difficulty multiplier from previous stage metrics
            drop_mult = 1.0
            weighted_need = 0.0
            if self.use_drop_weights:
                try:
                    prev_stage = max(1, current_stage - 1)
                    class_weights = self._get_class_weights(prev_stage)
                    obj_counts = candidate['snapshot'].get('object_counts', {})
                    num = 0.0
                    den = 0.0
                    for cid, cnt in obj_counts.items():
                        w = class_weights.get(int(cid), 0.0)
                        num += float(w) * float(cnt)
                        den += float(cnt)
                    weighted_need = (num / den) if den > 0 else 0.0
                    drop_mult = 1.0 + self.drop_weight_strength * float(weighted_need)
                except Exception:
                    drop_mult = 1.0
                    weighted_need = 0.0

            # Current-stage consolidation multiplier (deficit-only by default)
            curr_mult = 1.0
            curr_weighted_need = 0.0
            if self.use_current_stage_weights:
                try:
                    class_weights_curr = self._get_class_weights(
                        current_stage, alpha_override=self.current_alpha)
                    obj_counts = candidate['snapshot'].get('object_counts', {})
                    restrict_classes = (
                        set(self.stage_class_map.get(int(current_stage), []))
                        if self.stage_class_map else None
                    )
                    num_c = 0.0
                    den_c = 0.0
                    for cid, cnt in obj_counts.items():
                        if (restrict_classes is not None
                                and int(cid) not in restrict_classes):
                            continue
                        w = class_weights_curr.get(int(cid), 0.0)
                        num_c += float(w) * float(cnt)
                        den_c += float(cnt)
                    curr_weighted_need = (num_c / den_c) if den_c > 0 else 0.0
                    curr_mult = 1.0 + self.current_weight_strength * float(curr_weighted_need)
                except Exception:
                    curr_mult = 1.0
                    curr_weighted_need = 0.0

        final_score = float(base_score) * float(stage_mult) * float(drop_mult) * float(curr_mult)
        if not return_debug:
            return final_score
        return final_score, {
            'scene_id': candidate['scene_id'],
            'save_stage': save_stage,
            'base_score': float(base_score),
            'stage_multiplier': float(stage_mult),
            'weighted_class_need': float(weighted_need),
            'drop_multiplier': float(drop_mult),
            'curr_weighted_class_need': float(curr_weighted_need),
            'curr_multiplier': float(curr_mult),
            'final_score': float(final_score)
        }

    def _fallback_importance(self, candidate: Dict) -> float:
        """Heuristic importance used when no precomputed score is available."""
        snapshot = candidate['snapshot']
        present_classes = candidate['present_classes']
        class_balance_score = 0.0
        for class_id in present_classes:
            current_count = self.class_distribution.get(class_id, 0)
            class_balance_score += max(0, 10 - current_count) / 10.0
        class_balance_score /= max(1, len(present_classes))
        diversity_score = min(len(present_classes) / 5.0, 1.0)
        recency_score = 1.0
        total_objects = snapshot['data_info']['annos']['gt_num']
        density_score = min(total_objects / 15.0, 1.0)
        importance = (
            self.class_balance_weight * class_balance_score +
            self.diversity_weight * diversity_score +
            self.recency_weight * recency_score +
            self.density_weight * density_score
        )
        return importance

    def _get_class_weights(self, prev_stage_id: int, alpha_override: Optional[float] = None) -> Dict[int, float]:
        """Compute and cache per-class need weights from saved metrics.
        
        Args:
            prev_stage_id: Stage id (1-based) whose metrics to load
            alpha_override: If provided, override self.drop_alpha when blending
                            drop vs deficit. Use 0.0 for current-stage deficit-only.
        """
        cache_key = (int(prev_stage_id), None if alpha_override is None else float(alpha_override))
        if cache_key in self._class_weight_cache:
            return self._class_weight_cache[cache_key]
        weights: Dict[int, float] = {}
        try:
            if not self.metrics_dir:
                self._class_weight_cache[cache_key] = weights
                return weights
            import json as _json
            import os as _os
            prev_path = _os.path.join(self.metrics_dir, f'stage_{prev_stage_id}_metrics.json')
            if not _os.path.exists(prev_path):
                self._class_weight_cache[cache_key] = weights
                return weights
            with open(prev_path, 'r') as f:
                prev_data = _json.load(f)
            ap_prev = {int(e['model_idx']): float(e.get('AP_0.25', 0.0)) for e in prev_data.get('classes', [])}
            ap_prevprev = {}
            if prev_stage_id - 1 >= 1:
                prevprev_path = _os.path.join(self.metrics_dir, f'stage_{prev_stage_id-1}_metrics.json')
                if _os.path.exists(prevprev_path):
                    with open(prevprev_path, 'r') as f:
                        prevprev_data = _json.load(f)
                    ap_prevprev = {int(e['model_idx']): float(e.get('AP_0.25', 0.0)) for e in prevprev_data.get('classes', [])}
            alpha = self.drop_alpha if alpha_override is None else float(alpha_override)
            max_w = 0.0
            for cid, ap in ap_prev.items():
                d_prev = max(0.0, 1.0 - ap)
                if ap_prevprev and alpha > 0.0:
                    delta = ap - ap_prevprev.get(cid, ap)
                    drop = max(0.0, -delta)
                    w = alpha * drop + (1.0 - alpha) * d_prev
                else:
                    w = d_prev
                w = float(max(0.0, w))
                weights[int(cid)] = w
                max_w = max(max_w, w)
            if max_w > 0:
                for cid in list(weights.keys()):
                    weights[cid] = float(weights[cid] / max_w)
        except Exception:
            weights = {}
        self._class_weight_cache[cache_key] = weights
        return weights
    
    def _replace_scenes_in_memory(self, 
                                 scored_candidates: List[Tuple[float, Dict]], 
                                 stage_id: int) -> List[Dict]:
        """Replace existing scenes in memory with better candidates."""
        if not scored_candidates:
            return []
        
        # Get current memory scenes sorted by total importance (lowest first)
        # With new structure, each scene may have multiple stages
        current_scenes = []
        for scene_id, scene_data in self.memory_scenes.items():
            total_importance = scene_data['total_importance']
            current_scenes.append((total_importance, scene_id, scene_data))
        current_scenes.sort(key=lambda x: x[0])  # Lowest importance first
        
        selected_scenes = []
        
        for candidate_importance, candidate in scored_candidates:
            if not current_scenes:
                break
                
            # Get the least important scene in memory
            worst_importance, worst_scene_id, worst_scene_data = current_scenes[0]
            
            # Replace if candidate is more important than worst scene's total importance
            if candidate_importance > worst_importance:
                # Update class distribution (remove old scene's contribution)
                for class_id in worst_scene_data['present_classes']:
                    self.class_distribution[class_id] = max(0, self.class_distribution[class_id] - 1)
                    # Calculate total objects across all stages of this scene
                    total_objects = 0
                    for stage_data in worst_scene_data['stages'].values():
                        total_objects += stage_data['snapshot']['object_counts'].get(class_id, 0)
                    self.class_object_counts[class_id] = max(0, self.class_object_counts[class_id] - total_objects)
                
                # Remove worst scene completely (all stages)
                self._remove_scene_from_memory(worst_scene_id)
                current_scenes.pop(0)
                
                # Add new scene to memory using new structure
                candidate_scene_id = candidate['scene_id']
                # CRITICAL FIX: Use the stage where the scene was originally saved, not current stage
                save_stage = candidate['snapshot']['save_stage']
                self._add_scene_to_memory(candidate_scene_id, save_stage, candidate['snapshot'], candidate_importance)
                
                # Update class distribution (add new scene's contribution)
                for class_id in candidate['present_classes']:
                    self.class_distribution[class_id] += 1
                    obj_count = candidate['snapshot']['object_counts'].get(class_id, 0)
                    self.class_object_counts[class_id] += obj_count
                
                selected_scenes.append(candidate)
                
                # Update selection stats
                if stage_id not in self.scene_selection_stats:
                    self.scene_selection_stats[stage_id] = {}
                if candidate['scene_id'] not in self.scene_selection_stats[stage_id]:
                    self.scene_selection_stats[stage_id][candidate['scene_id']] = {}
                self.scene_selection_stats[stage_id][candidate['scene_id']]['status'] = 'replaced_in_memory'
                self.scene_selection_stats[stage_id][candidate['scene_id']]['importance_score'] = candidate_importance
                self.scene_selection_stats[stage_id][candidate['scene_id']]['replaced_scene'] = worst_scene_id
                
                if self.debug_mode:
                    print(f"   🔄 [DEBUG] Replaced {worst_scene_id} (total score: {worst_importance:.3f}) with {candidate['scene_id']} (score: {candidate_importance:.3f})")
            else:
                # Candidate not good enough, mark as rejected
                if stage_id not in self.scene_selection_stats:
                    self.scene_selection_stats[stage_id] = {}
                if candidate['scene_id'] not in self.scene_selection_stats[stage_id]:
                    self.scene_selection_stats[stage_id][candidate['scene_id']] = {}
                self.scene_selection_stats[stage_id][candidate['scene_id']]['status'] = 'rejected_low_importance'
                self.scene_selection_stats[stage_id][candidate['scene_id']]['importance_score'] = candidate_importance
                self.scene_selection_stats[stage_id][candidate['scene_id']]['min_required_score'] = worst_importance
        
        return selected_scenes
    
    def _select_best_scenes_per_class(self,
                                     stage_id: int,
                                     scenes_by_class: Dict[int, List[str]],
                                     model_to_name: Dict[int, str]):
        """Legacy method: Reduce scenes if we exceed per-class limits."""
        # For legacy mode, we keep all scenes added
        # Could implement reduction logic here if needed
        pass
    
    def _count_multi_snapshot_scenes(self) -> int:
        """Count scenes that have snapshots from multiple stages."""
        if self.use_legacy_mode:
            return sum(1 for snapshots in self.scene_snapshots.values()
                      if len(snapshots) > 1)
        return sum(
            1 for scene_data in self.memory_scenes.values()
            if len((scene_data.get('stages', {}) or {})) > 1
        )
    
    def _print_class_coverage(self, model_to_name: Dict[int, str]):
        """Print coverage statistics per class."""
        if not self.debug_mode:
            return

        if self.use_legacy_mode:
            coverage = {class_id: len(scene_refs)
                        for class_id, scene_refs in self.class_scenes.items()}
        else:
            # Global-budget mode should report selected memory seats, not pre-selection candidates.
            coverage = {int(class_id): int(scene_count)
                        for class_id, scene_count in self.class_distribution.items()
                        if int(scene_count) > 0}

        print(f"   Class coverage in memory bank:")
        for class_id in sorted(coverage.keys())[:10]:  # Show first 10
            class_name = model_to_name.get(class_id, f"class_{class_id}")
            scene_count = coverage[class_id]
            print(f"      - {class_name} (ID {class_id}): {scene_count} scenes")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory bank statistics."""
        if self.use_legacy_mode:
            scenes_per_class = {class_id: len(scene_refs)
                                for class_id, scene_refs in self.class_scenes.items()}
            scenes_per_stage = dict(self.scenes_per_stage)
        else:
            # Global-budget mode should report selected memory seats, not pre-selection candidates.
            scenes_per_class = {int(class_id): int(scene_count)
                                for class_id, scene_count in self.class_distribution.items()
                                if int(scene_count) > 0}
            scenes_per_stage = self._count_stage_distribution_in_memory()

        base_stats = {
            'total_unique_scenes': self._count_total_unique_scenes(),
            'total_snapshots': self._count_total_snapshots(),
            'multi_snapshot_scenes': self._count_multi_snapshot_scenes(),
            'scenes_per_stage': scenes_per_stage,
            'classes_covered': len(scenes_per_class),
            'scenes_per_class': scenes_per_class,
            'label_filtering_impact': {
                'total_objects_before': sum(s['before'] for s in self.label_filtering_stats.values()),
                'total_objects_after': sum(s['after'] for s in self.label_filtering_stats.values()),
            }
        }
        
        # Add global budget specific statistics
        if not self.use_legacy_mode:
            # Count scene-stage pairs for budget statistics
            scene_stage_pairs = self._count_scene_stage_pairs()
            budget_stats = {
                'memory_budget': {
                    'total_budget': self.memory_budget,
                    'budget_ratio': self.memory_budget_ratio,
                    'budget_used': scene_stage_pairs,
                    'budget_remaining': self.memory_budget - scene_stage_pairs,
                    'budget_utilization': scene_stage_pairs / self.memory_budget if self.memory_budget > 0 else 0.0,
                    'unique_scenes': len(self.memory_scenes)  # Also report unique scene count
                },
                'memory_bank': {
                    'active_scenes': len(self.memory_scenes),
                    'class_distribution': dict(self.class_distribution),
                    'class_object_counts': dict(self.class_object_counts),
                    'avg_importance': sum(self.scene_importance.values()) / len(self.scene_importance) if self.scene_importance else 0.0,
                    'importance_range': {
                        'min': min(self.scene_importance.values()) if self.scene_importance else 0.0,
                        'max': max(self.scene_importance.values()) if self.scene_importance else 0.0
                    }
                },
                'selection_strategy': {
                    'strategy': self.selection_strategy,
                    'replacement_strategy': self.replacement_strategy,
                    'scoring_weights': {
                        'class_balance': self.class_balance_weight,
                        'diversity': self.diversity_weight,
                        'recency': self.recency_weight,
                        'density': self.density_weight
                    }
                }
            }
            base_stats.update(budget_stats)
        
        return base_stats
    
    def _convert_numpy_types(self, obj):
        """Recursively convert NumPy types to native Python types for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_state(self, filepath: str):
        """Save memory bank state to file for debugging and visualization.
        
        IMPORTANT: Only saves selected memory bank scenes (memory_scenes), 
        NOT all candidates (scene_snapshots) to maintain incremental learning protocol.
        """
        importance_placeholder = self._importance_placeholder_mode()
        importance_semantics = self._importance_semantics()
        ld_strategy = (str(self.selection_strategy).strip().lower() == 'learning_dynamics')

        if importance_placeholder:
            invalid_importance = []
            for scene_id, scene_data in self.memory_scenes.items():
                for stage_id, stage_info in (scene_data.get('stages', {}) or {}).items():
                    try:
                        imp = float(stage_info.get('importance', 0.0))
                    except Exception:
                        invalid_importance.append(
                            dict(scene_id=str(scene_id), save_stage=int(stage_id), value=repr(stage_info.get('importance', None)))
                        )
                        continue
                    if (not np.isfinite(imp)) or (abs(float(imp) - 1.0) > 1e-6):
                        invalid_importance.append(
                            dict(scene_id=str(scene_id), save_stage=int(stage_id), value=float(imp))
                        )
            if invalid_importance:
                raise RuntimeError(
                    "Importance placeholder mode expects constant importance=1.0, "
                    f"but found {len(invalid_importance)} invalid entries "
                    f"(example: {invalid_importance[0]})."
                )

        def _read_optional_score(stage_info: Dict[str, Any], key: str) -> Optional[float]:
            if key not in stage_info:
                return None
            raw = stage_info.get(key, None)
            if raw is None:
                return None
            try:
                val = float(raw)
            except Exception as e:
                raise RuntimeError(
                    f"Invalid non-numeric seat score '{key}': {raw!r}"
                ) from e
            if not np.isfinite(val):
                raise RuntimeError(f"Invalid non-finite seat score '{key}': {val!r}")
            return float(val)

        # Only save selected memory bank scenes with minimal data
        # All other data can be reconstructed from the dataset using scene_id.
        memory_scenes_data = {}
        seat_scores_data = {}
        for scene_id, scene_data in self.memory_scenes.items():
            stages_data = {}
            scores_data = {}
            for stage_id, stage_info in scene_data.get('stages', {}).items():
                # Do not emit placeholder importance values in stage-ratio modes.
                if importance_placeholder:
                    stages_data[str(stage_id)] = {'selected': True}
                else:
                    imp = _read_optional_score(stage_info, 'importance')
                    if imp is None:
                        raise RuntimeError(
                            "Missing importance in non-placeholder mode for "
                            f"scene_id={scene_id}, save_stage={stage_id}."
                        )
                    stages_data[str(stage_id)] = float(imp)

                ul = _read_optional_score(stage_info, 'underlearning_score')
                fg = _read_optional_score(stage_info, 'forgetness_score')
                ld_f = _read_optional_score(stage_info, 'learning_dynamics_forgetness')
                ld_u = _read_optional_score(stage_info, 'learning_dynamics_replay_priority')
                d1_u = _read_optional_score(stage_info, 'learning_dynamics_design1_unary')
                d1_e1 = _read_optional_score(stage_info, 'learning_dynamics_design1_e1')
                d1_e2 = _read_optional_score(stage_info, 'learning_dynamics_design1_e2')

                if ld_strategy and (ul is not None or fg is not None):
                    raise RuntimeError(
                        "LD memory mode should not carry legacy underlearning/forgetness seat scores. "
                        f"scene_id={scene_id}, save_stage={stage_id}, "
                        f"underlearning_score={ul}, forgetness_score={fg}"
                    )

                stage_scores = {}
                if ul is not None:
                    stage_scores['underlearning_score'] = float(ul)
                if fg is not None:
                    stage_scores['forgetness_score'] = float(fg)
                if ld_f is not None:
                    stage_scores['learning_dynamics_forgetness'] = float(ld_f)
                if ld_u is not None:
                    stage_scores['learning_dynamics_replay_priority'] = float(ld_u)
                if d1_u is not None:
                    stage_scores['learning_dynamics_design1_unary'] = float(d1_u)
                if d1_e1 is not None:
                    stage_scores['learning_dynamics_design1_e1'] = float(d1_e1)
                if d1_e2 is not None:
                    stage_scores['learning_dynamics_design1_e2'] = float(d1_e2)
                if stage_scores:
                    scores_data[str(stage_id)] = stage_scores
            memory_scenes_data[scene_id] = stages_data
            seat_scores_data[scene_id] = scores_data

        if ld_strategy:
            max_save_stage = 0
            for _, scene_data in self.memory_scenes.items():
                for stage_id in (scene_data.get('stages', {}) or {}).keys():
                    try:
                        max_save_stage = max(max_save_stage, int(stage_id))
                    except Exception:
                        continue
            if max_save_stage >= 2:
                missing_ld_scores = []
                for scene_id, scene_data in self.memory_scenes.items():
                    for stage_id, stage_info in (scene_data.get('stages', {}) or {}).items():
                        ld_f = stage_info.get('learning_dynamics_forgetness', None)
                        ld_u = stage_info.get('learning_dynamics_replay_priority', None)
                        if ld_f is None and ld_u is None:
                            missing_ld_scores.append(
                                dict(scene_id=str(scene_id), save_stage=int(stage_id))
                            )
                if missing_ld_scores:
                    raise RuntimeError(
                        "LD state export requires per-seat LD scores for all stored seats "
                        f"after stage>=2, but found {len(missing_ld_scores)} missing entries "
                        f"(example: {missing_ld_scores[0]})."
                    )
        
        # Get current statistics for summary
        stats = self.get_statistics()
        
        # Create human-readable scene-stage list, sorted by best-available per-seat score
        # (highest first). "importance" is a placeholder in the SUNRGBD stage-ratio paths.
        scene_stage_pairs = []
        for scene_id, scene_data in self.memory_scenes.items():
            for stage_id, stage_info in scene_data['stages'].items():
                importance = _read_optional_score(stage_info, 'importance')
                ul = _read_optional_score(stage_info, 'underlearning_score')
                fg = _read_optional_score(stage_info, 'forgetness_score')
                ld_f = _read_optional_score(stage_info, 'learning_dynamics_forgetness')
                ld_u = _read_optional_score(stage_info, 'learning_dynamics_replay_priority')
                d1_u = _read_optional_score(stage_info, 'learning_dynamics_design1_unary')
                d1_e1 = _read_optional_score(stage_info, 'learning_dynamics_design1_e1')
                d1_e2 = _read_optional_score(stage_info, 'learning_dynamics_design1_e2')
                # CRITICAL FIX: Use the actual save_stage from snapshot, not the storage key
                save_stage = stage_info['snapshot']['save_stage']
                sort_score = None
                if d1_u is not None:
                    sort_score = float(d1_u)
                elif ld_u is not None:
                    sort_score = float(ld_u)
                elif ld_f is not None:
                    sort_score = float(ld_f)
                elif ul is not None:
                    sort_score = float(ul)
                elif fg is not None:
                    sort_score = float(fg)
                elif not importance_placeholder:
                    assert importance is not None
                    sort_score = float(importance)
                sort_key = float(sort_score) if sort_score is not None else float('-inf')
                scene_stage_pairs.append((
                    sort_key,
                    str(scene_id),
                    int(save_stage),
                    importance,
                    ul,
                    fg,
                    ld_f,
                    ld_u,
                    d1_u,
                    d1_e1,
                    d1_e2,
                    sort_score,
                ))
        
        scene_stage_pairs.sort(key=lambda x: (x[1], x[2]))
        scene_stage_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Format into strings
        scene_stage_list = []
        for _, scene_id, save_stage, importance, ul, fg, ld_f, ld_u, d1_u, d1_e1, d1_e2, sort_score in scene_stage_pairs:
            if importance_placeholder:
                parts = [f"{scene_id} (stage {save_stage}"]
            else:
                assert importance is not None
                parts = [f"{scene_id} (stage {save_stage}, importance: {float(importance):.3f}"]
            if ul is not None:
                parts.append(f"underlearning_score: {float(ul):.3f}")
            if fg is not None:
                parts.append(f"forgetness_score: {float(fg):.3f}")
            if ld_f is not None:
                parts.append(f"ld_forgetness: {float(ld_f):.3f}")
            if ld_u is not None:
                parts.append(f"ld_replay_priority: {float(ld_u):.3f}")
            if d1_u is not None:
                parts.append(f"ld_design1_unary: {float(d1_u):.3f}")
            if d1_e1 is not None and d1_e2 is not None:
                parts.append(f"ld_design1_e: ({float(d1_e1):.3f}, {float(d1_e2):.3f})")
            if sort_score is not None:
                parts.append(f"sort_score: {float(sort_score):.3f}")
            scene_stage_list.append(", ".join(parts) + ")")
        
        # Add human-readable summary
        if not self.use_legacy_mode:
            budget_info = stats['memory_budget']
            summary = {
                'type': 'Scene Memory Bank State File',
                'mode': 'Global Budget (Scene-Stage Pairs)',
                'budget_usage': f"{budget_info['budget_used']}/{budget_info['total_budget']} scene-stage pairs ({budget_info['budget_utilization']*100:.1f}% used)",
                'unique_scenes': budget_info['unique_scenes'],
                'selection_strategy': self.selection_strategy,
                'importance_semantics': str(importance_semantics),
                'memory_scenes_value_semantics': (
                    'stage_membership_only' if importance_placeholder else 'importance_by_stage'
                ),
                'total_stages_saved': len({
                    int(st)
                    for sdata in self.memory_scenes.values()
                    for st in (sdata.get('stages', {}) or {}).keys()
                }),
                'file_purpose': 'Stores scene references for incremental learning replay (no raw data)',
                'last_updated': 'Generated during training'
            }
        else:
            summary = {
                'type': 'Scene Memory Bank State File',
                'mode': 'Legacy (Per-Class)',
                'scenes_per_class': self.scenes_per_class,
                'file_purpose': 'Stores scene references for incremental learning replay (no raw data)',
                'last_updated': 'Generated during training'
            }

        cfg_export = {
            'mode': 'legacy' if self.use_legacy_mode else 'global_budget',
            'selection_strategy': self.selection_strategy,
            'dedup_strategy': self.dedup_strategy,
        }
        if self.use_legacy_mode:
            cfg_export['scenes_per_class'] = int(self.scenes_per_class)
        else:
            cfg_export['memory_budget'] = int(self.memory_budget)
            cfg_export['memory_budget_ratio'] = float(self.memory_budget_ratio)
            cfg_export['replacement_strategy'] = self.replacement_strategy
            cfg_export['scoring_weights'] = {
                'class_balance': self.class_balance_weight,
                'diversity': self.diversity_weight,
                'recency': self.recency_weight,
                'density': self.density_weight
            }

        state = {
            'summary': summary,
            'scene_stage_list': scene_stage_list,  # Human-readable list for easy inspection
            'memory_scenes': memory_scenes_data,   # Minimal data for code
            'seat_scores': seat_scores_data,       # Per-seat insertion/eviction scores (present keys only)
            'statistics': stats,
            'config': cfg_export,
            # Debugging information (aggregated stats only, no individual scene data)
            'debug_info': {
                'total_memory_scenes': len(self.memory_scenes),
                'memory_budget': self.memory_budget,
                'scenes_per_stage': dict(stats.get('scenes_per_stage', {})),
                'class_distribution': dict(self.class_distribution),
                'deduplication_stats': dict(self.deduplication_stats),
                'class_filtering_stats': dict(self.class_filtering_stats),
                'label_filtering_summary': {
                    'total_scenes_filtered': len(self.label_filtering_stats),
                    'total_objects_before': sum(s['before'] for s in self.label_filtering_stats.values()),
                    'total_objects_after': sum(s['after'] for s in self.label_filtering_stats.values()),
                },
            }
        }
        last_scoring = getattr(self, '_last_scoring_details', None)
        if last_scoring is not None:
            state['debug_info']['last_scoring'] = last_scoring
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert NumPy types to native Python types for JSON serialization
        state_clean = self._convert_numpy_types(state)
        
        from .memory_bank.state_io import atomic_write
        atomic_write(filepath, lambda f: json.dump(state_clean, f, indent=2))
        
        print(f"Scene memory bank state saved to {filepath}")
    
    def save_checkpoint(self, filepath, stage_id, context=None):
        """Save selected global-budget replay snapshots for stage-boundary resume."""
        from .memory_bank.scene_checkpoint import save_checkpoint
        save_checkpoint(self, filepath, stage_id, context)

    @classmethod
    def load_checkpoint(cls, filepath, **kwargs):
        """Restore a trusted checkpoint, checking requested stage/config/protocol."""
        from .memory_bank.scene_checkpoint import load_checkpoint
        return load_checkpoint(cls, filepath, **kwargs)

    def print_summary(self):
        """Print a summary of the memory bank state."""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"SCENE MEMORY BANK SUMMARY")
        print(f"{'='*60}")
        print(f"Mode: {'Legacy (per-class)' if self.use_legacy_mode else 'Global Budget'}")
        
        if not self.use_legacy_mode:
            budget = stats['memory_budget']
            print(f"Memory Budget: {budget['budget_used']}/{budget['total_budget']} scene-stage pairs ({budget['budget_utilization']*100:.1f}% used)")
            print(f"Budget Ratio: {budget['budget_ratio']*100:.1f}% of {self.total_training_scenes} total scenes")
            print(f"Unique Scenes in Memory: {budget['unique_scenes']}")
            
            memory = stats['memory_bank']
            print(f"Active Memory Scenes: {memory['active_scenes']}")
            if self._importance_placeholder_mode():
                print(f"Importance: {self._importance_semantics()}")
            else:
                print(f"Average Importance: {memory['avg_importance']:.3f}")
                print(f"Importance Range: {memory['importance_range']['min']:.3f} - {memory['importance_range']['max']:.3f}")
        
        print(f"Total unique scenes: {stats['total_unique_scenes']}")
        print(f"Total snapshots: {stats['total_snapshots']}")
        print(f"Multi-snapshot scenes: {stats['multi_snapshot_scenes']}")
        print(f"Classes covered: {stats['classes_covered']}")
        print(f"Label filtering impact: {stats['label_filtering_impact']['total_objects_before']} → "
              f"{stats['label_filtering_impact']['total_objects_after']} objects")
        print(f"Scenes per stage: {stats['scenes_per_stage']}")
        print(f"{'='*60}")
