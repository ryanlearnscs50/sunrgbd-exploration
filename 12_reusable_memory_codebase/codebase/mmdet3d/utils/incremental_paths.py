"""
Unified Path Management for Incremental Learning

This module provides centralized path management to eliminate the confusion
between different work_dir concepts that accumulated through development iterations.

Key Design Principles:
- One experiment = one root directory
- Clear subdirectory purposes (checkpoints/, pseudo_labels/, learning_dynamics/, etc.)
- Single source of truth for each path type

NOTE (2026-02-05):
- The incremental pipeline no longer writes a `metrics/` folder under work_dir.
- Backward compatibility for legacy folder layouts is handled by standalone
  analysis / reporting scripts via fallback readers.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Union


class IncrementalPaths:
    """Centralized path management for incremental learning experiments.
    
    This class eliminates the confusion between incremental_cfg.work_dir,
    stage_cfg.work_dir, and dataset work_dir by providing a single,
    clear interface for all file paths in incremental learning.
    """
    
    def __init__(self, experiment_dir: Union[str, Path]):
        """Initialize path manager for an experiment.
        
        Args:
            experiment_dir: Root directory for the entire incremental experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create standard subdirectories
        self._ensure_subdirs()
    
    def _ensure_subdirs(self):
        """Create standard subdirectory structure."""
        # Always-present experiment artifacts (uniform across runs).
        (self.experiment_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'pseudo_labels').mkdir(exist_ok=True)
        (self.experiment_dir / 'learning_dynamics').mkdir(exist_ok=True)

        # Memory bank actions are always present; scores are created lazily
        # only when a run writes score artifacts.
        (self.experiment_dir / 'memory_bank' / 'actions').mkdir(parents=True, exist_ok=True)

        # Reviewing: always create the folder; keep empty if disabled.
        (self.experiment_dir / 'reviewing' / 'weights').mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / 'reviewing' / 'actions').mkdir(parents=True, exist_ok=True)
    
    # === Core Directory Methods ===
    
    def root(self) -> Path:
        """Get experiment root directory."""
        return self.experiment_dir
    
    def checkpoints_dir(self, stage_id: Optional[int] = None) -> Path:
        """Get checkpoints directory, optionally for specific stage."""
        if stage_id is None:
            return self.experiment_dir / 'checkpoints'
        return self.experiment_dir / 'checkpoints' / f'stage_{stage_id}'
    
    def pseudo_labels_dir(self) -> Path:
        """Get pseudo labels directory."""
        return self.experiment_dir / 'pseudo_labels'

    def memory_pseudo_labels_dir(self) -> Path:
        """Get directory for memory-enrichment pseudo labels."""
        d = self.pseudo_labels_dir() / 'memory_enrichment'
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    def learning_dynamics_dir(self) -> Path:
        """Get learning-dynamics tracking directory."""
        return self.experiment_dir / 'learning_dynamics'

    def memory_bank_dir(self) -> Path:
        """Get root directory for memory bank artifacts."""
        return self.experiment_dir / 'memory_bank'

    def memory_bank_scores_dir(self) -> Path:
        """Get directory path for memory bank score artifacts (per-stage).

        NOTE: This accessor returns the path only; it does not create the
        directory. Writers should call `mkdir(..., exist_ok=True)` explicitly
        when score artifacts are actually emitted.
        """
        return self.memory_bank_dir() / 'scores'

    def memory_bank_actions_dir(self) -> Path:
        """Get directory for memory bank action/state artifacts (per-stage)."""
        return self.memory_bank_dir() / 'actions'

    def reviewing_dir(self) -> Path:
        """Get root directory for reviewing artifacts (kept empty when disabled)."""
        return self.experiment_dir / 'reviewing'

    def reviewing_weights_dir(self) -> Path:
        """Get directory for reviewing per-seat weight artifacts."""
        return self.reviewing_dir() / 'weights'

    def reviewing_actions_dir(self) -> Path:
        """Get directory for reviewing resampling action artifacts."""
        return self.reviewing_dir() / 'actions'

    def debug_dir(self) -> Path:
        """Get debug directory (created lazily when requested)."""
        d = self.experiment_dir / 'debug'
        d.mkdir(parents=True, exist_ok=True)
        return d

    # === Retention / Pseudo Consistency Aux Dirs ===
    def pseudo_sets_dir(self) -> Path:
        return self.debug_dir() / 'pseudo_sets'

    def retention_scores_dir(self) -> Path:
        return self.debug_dir() / 'retention_scores'

    def pseudo_set_file(self, stage_id: int, tag: str) -> Path:
        return self.pseudo_sets_dir() / f'stage_{stage_id}_{tag}.jsonl'

    def retention_scores_file(self, stage_id: int, tag: str) -> Path:
        return self.retention_scores_dir() / f'stage_{stage_id}_{tag}.json'
    
    # === Specific File Methods ===
    
    def checkpoint_file(self, stage_id: int, epoch: Optional[int] = None, 
                       best: bool = False) -> Path:
        """Get path to specific checkpoint file."""
        stage_dir = self.checkpoints_dir(stage_id)
        stage_dir.mkdir(exist_ok=True)
        
        if best:
            # Find the best checkpoint pattern
            return stage_dir / 'best_mAP_*.pth'
        elif epoch is None:
            return stage_dir / 'latest.pth'
        else:
            return stage_dir / f'epoch_{epoch}.pth'
    
    def pseudo_label_file(self, stage_id: int) -> Path:
        """Get path to pseudo label file for a stage."""
        return self.pseudo_labels_dir() / f'stage_{stage_id}_pseudo_labels.pkl'

    def memory_pseudo_label_file(self, stage_id: int) -> Path:
        """Get path to memory-enrichment pseudo label file generated at a stage."""
        return self.memory_pseudo_labels_dir() / f'stage_{stage_id}_memory_pseudo_labels.pkl'
    
    def scene_memory_file(self, stage_id: int) -> Path:
        """Get path to scene memory bank file."""
        return self.memory_bank_actions_dir() / f'scene_memory_stage_{stage_id}.json'
    
    def object_memory_manifest(self, stage_id: int) -> Path:
        """Get path to object memory bank manifest."""
        return self.memory_bank_actions_dir() / f'object_memory_stage_{stage_id}.json'
    
    def object_exemplars_dir(self, stage_id: int) -> Path:
        """Get directory for object exemplar files."""
        exemplar_dir = self.debug_dir() / 'object_exemplars' / f'stage_{stage_id}'
        exemplar_dir.mkdir(parents=True, exist_ok=True)
        return exemplar_dir
    
    def stage_metrics_file(self, stage_id: int) -> Path:
        """Get path to stage-specific metrics file (used by memory-bank weighting)."""
        return self.memory_bank_scores_dir() / f'stage_{stage_id}_metrics.json'
    
    def forgetting_metrics_file(self, stage_id: Optional[int] = None) -> Path:
        """Get path to forgetting metrics file (optional diagnostic)."""
        if stage_id is None:
            return self.memory_bank_scores_dir() / 'overall_forgetting_metrics.json'
        return self.memory_bank_scores_dir() / f'forgetting_metrics_stage_{stage_id}.json'
    
    def training_scenes_file(self, stage_id: int) -> Path:
        """Get path to training scenes debug file."""
        return self.debug_dir() / f'training_scenes_stage_{stage_id}.json'
    
    def experiment_log_file(self, timestamp: str) -> Path:
        """Get path to main experiment log."""
        # The pipeline writes the long log at work_dir root; keep logs/ unused.
        return self.root() / f'incremental_training_{timestamp}.log'
    
    def tensorboard_dir(self, stage_id: Optional[int] = None) -> Path:
        """Get tensorboard logs directory."""
        if stage_id is None:
            return self.debug_dir() / 'tensorboard'
        tb_dir = self.debug_dir() / 'tensorboard' / f'stage_{stage_id}'
        tb_dir.mkdir(parents=True, exist_ok=True)
        return tb_dir

    # === Deprecated directory accessors (kept for downstream scripts; pipeline avoids) ===
    def memory_banks_dir(self) -> Path:
        """DEPRECATED: use memory_bank_actions_dir()."""
        warnings.warn(
            "`paths.memory_banks_dir()` is deprecated; use `paths.memory_bank_actions_dir()`.",
            DeprecationWarning,
        )
        return self.memory_bank_actions_dir()

    def metrics_dir(self) -> Path:
        """DEPRECATED: the pipeline no longer writes a `metrics/` folder."""
        raise RuntimeError(
            "`paths.metrics_dir()` is no longer supported. "
            "Use `learning_dynamics_dir()`, `memory_bank_scores_dir()`, "
            "`memory_bank_actions_dir()`, or reviewing dirs instead."
        )

    def logs_dir(self) -> Path:
        """DEPRECATED: long logs are written at work_dir root."""
        warnings.warn(
            "`paths.logs_dir()` is deprecated; long logs are written at work_dir root.",
            DeprecationWarning,
        )
        return self.root() / 'logs'
    
    # === Backward Compatibility Methods ===
    
    def resolve_legacy_pseudo_labels(self, stage_id: int) -> Optional[Path]:
        """Resolve pseudo labels with fallback to legacy locations.
        
        This method handles backward compatibility for experiments that
        might have pseudo labels in various locations from previous
        development iterations.
        
        Returns:
            Path to pseudo label file if found, None otherwise
        """
        # Primary location (new unified structure)
        primary_path = self.pseudo_label_file(stage_id)
        if primary_path.exists():
            return primary_path
        
        # Legacy location 1: stage-specific subdirectory
        legacy_stage_path = (self.experiment_dir / f'stage_{stage_id}' / 
                           'pseudo_labels' / f'stage_{stage_id}_pseudo_labels.pkl')
        if legacy_stage_path.exists():
            warnings.warn(
                f"Found pseudo labels in legacy location: {legacy_stage_path}. "
                f"Consider moving to unified location: {primary_path}",
                UserWarning
            )
            return legacy_stage_path
        
        # Legacy location 2: hardcoded fallback directory
        legacy_fallback = Path('./incremental_logs/pseudo_label_based') / f'stage_{stage_id}_pseudo_labels.pkl'
        if legacy_fallback.exists():
            warnings.warn(
                f"Found pseudo labels in legacy fallback: {legacy_fallback}. "
                f"Consider moving to unified location: {primary_path}",
                UserWarning
            )
            return legacy_fallback
        
        return None
    
    def migrate_from_legacy_structure(self, verbose: bool = True):
        """Migrate files from legacy directory structure to unified structure.
        
        This helps transition existing experiments to the new structure.
        """
        if verbose:
            print(f"🔄 Migrating experiment to unified structure: {self.experiment_dir}")
        
        # Migrate stage directories to new structure
        for stage_dir in self.experiment_dir.glob('stage_*'):
            if not stage_dir.is_dir():
                continue
                
            stage_num = stage_dir.name.split('_')[1]
            if verbose:
                print(f"  Processing {stage_dir.name}...")
            
            # Move checkpoints
            stage_checkpoints = self.checkpoints_dir(int(stage_num))
            for checkpoint in stage_dir.glob('*.pth'):
                target = stage_checkpoints / checkpoint.name
                if not target.exists():
                    checkpoint.rename(target)
                    if verbose:
                        print(f"    Moved {checkpoint.name} to checkpoints/")
            
            # Move metrics files
            for metrics_file in stage_dir.glob('*metrics*.json'):
                target = self.memory_bank_scores_dir() / metrics_file.name
                if not target.exists():
                    metrics_file.rename(target)
                    if verbose:
                        print(f"    Moved {metrics_file.name} to memory_bank/scores/")
            
            # Move debug files
            for debug_file in stage_dir.glob('training_scenes*.json'):
                target = self.debug_dir() / debug_file.name
                if not target.exists():
                    debug_file.rename(target)
                    if verbose:
                        print(f"    Moved {debug_file.name} to debug/")
            
            # Move memory bank files
            for memory_file in stage_dir.glob('*memory_bank*.json'):
                target = self.memory_banks_dir() / memory_file.name
                if not target.exists():
                    memory_file.rename(target)
                    if verbose:
                        print(f"    Moved {memory_file.name} to memory_banks/")
            
            # Move exemplar debug directories
            exemplars_debug = stage_dir / 'exemplars_debug'
            if exemplars_debug.exists():
                target = self.debug_dir() / 'object_exemplars' / f'stage_{stage_num}'
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    exemplars_debug.rename(target)
                    if verbose:
                        print(f"    Moved exemplars_debug to debug/object_exemplars/")
        
        # Move overall metrics to memory_bank/scores directory
        for metrics_file in self.experiment_dir.glob('overall_*.json'):
            target = self.memory_bank_scores_dir() / metrics_file.name
            if not target.exists():
                metrics_file.rename(target)
                if verbose:
                    print(f"  Moved {metrics_file.name} to memory_bank/scores/")
        
        if verbose:
            print("✅ Migration complete!")
    
    def __str__(self) -> str:
        return f"IncrementalPaths({self.experiment_dir})"
    
    def __repr__(self) -> str:
        return f"IncrementalPaths(experiment_dir='{self.experiment_dir}')"


# === Convenience Functions ===

def create_incremental_paths(experiment_dir: Union[str, Path]) -> IncrementalPaths:
    """Create IncrementalPaths instance for an experiment.
    
    Args:
        experiment_dir: Root directory for the experiment
        
    Returns:
        IncrementalPaths instance
    """
    return IncrementalPaths(experiment_dir)


def migrate_experiment_structure(experiment_dir: Union[str, Path], 
                                verbose: bool = True) -> IncrementalPaths:
    """Create IncrementalPaths and migrate legacy structure.
    
    Args:
        experiment_dir: Root directory for the experiment
        verbose: Whether to print migration progress
        
    Returns:
        IncrementalPaths instance
    """
    paths = IncrementalPaths(experiment_dir)
    paths.migrate_from_legacy_structure(verbose=verbose)
    return paths
