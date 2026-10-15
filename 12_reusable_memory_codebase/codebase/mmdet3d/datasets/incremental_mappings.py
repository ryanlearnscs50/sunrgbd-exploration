"""
Incremental Learning Mapping Utilities

Provides utilities for managing class mappings in incremental learning scenarios.
Supports explicit stage definitions with clear NYU40 ID to model index mappings.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any


def _infer_num_classes(stage_definitions: List[Dict]) -> int:
    """Infer total number of classes from stage definitions.

    Assumes class indices are 0-indexed.
    """
    all_indices: List[int] = []
    for stage in stage_definitions:
        all_indices.extend([int(x) for x in stage.get('class_indices', [])])
    if not all_indices:
        raise ValueError("stage_definitions has no class_indices")
    return int(max(all_indices) + 1)


def create_mapping_from_config(stage_definitions: List[Dict]) -> Dict[str, Any]:
    """Create all necessary mappings from explicit stage configuration.
    
    Args:
        stage_definitions: List of stage definition dictionaries with:
            - class_indices: Model class indices (0-indexed)
            - class_names: Human readable class names
            - nyu40_ids: Corresponding NYU40 label IDs (optional; ScanNet only)
            
    Returns:
        Dictionary containing all mapping tables:
        - model_idx_to_name: {0: 'class0', 1: 'class1', ...}
        - model_idx_to_nyu40: {0: 3, 1: 4, ...} (empty for non-ScanNet)
        - nyu40_to_model_idx: {3: 0, 4: 1, ...} (empty for non-ScanNet)
        - class_names: ['cabinet', 'bed', 'chair', ...]
        - valid_nyu40_ids: [3, 4, 5, ...] (empty for non-ScanNet)
    """
    model_idx_to_name: Dict[int, str] = {}
    model_idx_to_nyu40: Dict[int, int] = {}
    nyu40_to_model_idx: Dict[int, int] = {}
    
    # Build mappings from stage definitions
    for stage in stage_definitions:
        indices = [int(x) for x in stage['class_indices']]
        names = list(stage['class_names'])
        nyu40_ids = stage.get('nyu40_ids', None)
        
        for i, model_idx in enumerate(indices):
            name = names[i]
            model_idx_to_name[model_idx] = name

            if nyu40_ids is not None:
                nyu40_id = int(nyu40_ids[i])
                model_idx_to_nyu40[model_idx] = nyu40_id
                nyu40_to_model_idx[nyu40_id] = model_idx
    
    # Create ordered class names list (0..num_classes-1)
    num_classes = _infer_num_classes(stage_definitions)
    class_names = [model_idx_to_name[i] for i in range(num_classes)]
    valid_nyu40_ids = sorted(nyu40_to_model_idx.keys())
    
    # Create reverse name mapping for dataset filtering
    name_to_model_idx = {name: idx for idx, name in model_idx_to_name.items()}
    
    return {
        'model_idx_to_name': model_idx_to_name,
        'name_to_model_idx': name_to_model_idx,
        'model_idx_to_nyu40': model_idx_to_nyu40, 
        'nyu40_to_model_idx': nyu40_to_model_idx,
        'class_names': class_names,
        'valid_nyu40_ids': valid_nyu40_ids,
        'num_classes': num_classes,
    }


def get_seen_classes_mask(stage_definitions: List[Dict], current_stage_id: int) -> torch.Tensor:
    """Get boolean mask for classes seen up to current stage.
    
    Args:
        stage_definitions: List of stage definitions
        current_stage_id: Current stage ID (1, 2, 3, ...)
        
    Returns:
        Boolean tensor of shape (num_classes,) where True indicates class has been seen
    """
    seen_indices = []
    
    for stage in stage_definitions:
        if stage['stage_id'] <= current_stage_id:
            seen_indices.extend(stage['class_indices'])
    
    num_classes = _infer_num_classes(stage_definitions)
    mask = torch.zeros(num_classes, dtype=torch.bool)
    if seen_indices:
        mask[[int(x) for x in seen_indices]] = True
        
    return mask


def get_stage_classes(stage_definitions: List[Dict], stage_id: int) -> List[int]:
    """Get class indices for a specific stage.
    
    Args:
        stage_definitions: List of stage definitions
        stage_id: Stage ID to get classes for
        
    Returns:
        List of class indices for the stage
    """
    for stage in stage_definitions:
        if stage['stage_id'] == stage_id:
            return stage['class_indices']
    
    raise ValueError(f"Stage {stage_id} not found in stage definitions")


def get_all_seen_classes_up_to_stage(stage_definitions: List[Dict], stage_id: int) -> List[int]:
    """Get all class indices seen up to and including given stage.
    
    Args:
        stage_definitions: List of stage definitions
        stage_id: Stage ID (inclusive)
        
    Returns:
        Sorted list of all class indices seen up to stage
    """
    seen_classes = []
    
    for stage in stage_definitions:
        if stage['stage_id'] <= stage_id:
            seen_classes.extend(stage['class_indices'])
    
    return sorted(seen_classes)


def map_targets_to_seen_classes(targets: torch.Tensor, seen_classes_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map target class indices to seen class space for loss computation.
    
    Args:
        targets: Original target indices in model class space (0..num_classes-1 or background).
        seen_classes_mask: Boolean mask of shape (num_classes,) indicating seen classes
        
    Returns:
        Tuple of:
        - mapped_targets: Targets mapped to seen class space (0 to n_seen-1)
        - valid_mask: Boolean mask indicating which targets are valid (in seen classes)
    """
    num_classes = int(seen_classes_mask.numel())
    # Create mapping from num_classes-class space to seen class space
    seen_indices = torch.where(seen_classes_mask)[0]
    n_seen = len(seen_indices)
    
    # Create reverse mapping: full index -> seen class index
    full_to_seen_mapping = torch.full((num_classes,), -1, dtype=torch.long)
    full_to_seen_mapping[seen_indices] = torch.arange(n_seen)
    
    # Map targets
    valid_mask = (targets < num_classes) & seen_classes_mask[targets.clamp(0, num_classes - 1)]
    mapped_targets = torch.where(
        valid_mask,
        full_to_seen_mapping[targets.clamp(0, num_classes - 1)],
        n_seen  # Background class for invalid/unseen targets
    )
    
    return mapped_targets, valid_mask


def validate_incremental_mappings(stage_definitions: List[Dict], verbose: bool = False) -> bool:
    """Validate that incremental learning mappings are consistent.
    
    Args:
        stage_definitions: Stage definitions to validate
        verbose: Whether to print detailed validation info
        
    Returns:
        True if valid, raises AssertionError if invalid
    """
    if verbose:
        print("Validating incremental learning mappings...")
    
    # Create mappings
    mappings = create_mapping_from_config(stage_definitions)
    num_classes = int(mappings['num_classes'])
    
    # Validate mapping completeness
    assert len(mappings['class_names']) == num_classes, \
        f"Expected {num_classes} class names, got {len(mappings['class_names'])}"
    # NYU40 mappings are only required when nyu40_ids are present
    if mappings['model_idx_to_nyu40'] or mappings['nyu40_to_model_idx']:
        assert len(mappings['model_idx_to_nyu40']) == num_classes, \
            f"Expected {num_classes} model->nyu40 mappings, got {len(mappings['model_idx_to_nyu40'])}"
        assert len(mappings['nyu40_to_model_idx']) == num_classes, \
            f"Expected {num_classes} nyu40->model mappings, got {len(mappings['nyu40_to_model_idx'])}"
    
    # Validate bidirectional consistency
    if mappings['model_idx_to_nyu40']:
        for model_idx in range(num_classes):
            nyu40_id = mappings['model_idx_to_nyu40'][model_idx]
            assert mappings['nyu40_to_model_idx'][nyu40_id] == model_idx, \
                f"Bidirectional mapping inconsistency at model_idx {model_idx}"
    
    # Validate stage completeness
    all_stage_classes = []
    for stage in stage_definitions:
        all_stage_classes.extend(stage['class_indices'])
    
    assert len(set(all_stage_classes)) == num_classes, f"Stages don't cover all {num_classes} classes"
    assert set(all_stage_classes) == set(range(num_classes)), \
        f"Stage classes must be exactly 0-{num_classes - 1}"
    
    if verbose:
        print("Incremental learning mappings validated successfully!")
        
        # Print mapping summary
        print(f"   Model classes: 0-{num_classes - 1} ({len(mappings['class_names'])} total)")
        if mappings['valid_nyu40_ids']:
            print(
                f"   NYU40 range: {min(mappings['valid_nyu40_ids'])}-{max(mappings['valid_nyu40_ids'])}"
            )
        print(f"   Stages: {len(stage_definitions)}")
        
        # Test seen classes mask for each stage
        for stage in stage_definitions:
            mask = get_seen_classes_mask(stage_definitions, stage['stage_id'])
            n_seen = mask.sum().item()
            expected_seen = len(get_all_seen_classes_up_to_stage(stage_definitions, stage['stage_id']))
            assert n_seen == expected_seen, f"Stage {stage['stage_id']} mask mismatch"
            print(f"   Stage {stage['stage_id']}: {n_seen} classes seen")
    
    return True


# Example usage and testing
if __name__ == "__main__":
    # Example stage definitions for testing
    test_stage_defs = [
        {
            'stage_id': 1,
            'class_indices': [0, 1, 2],
            'class_names': ['cabinet', 'bed', 'chair'], 
            'nyu40_ids': [3, 4, 5]
        },
        {
            'stage_id': 2, 
            'class_indices': [3, 4],
            'class_names': ['sofa', 'table'],
            'nyu40_ids': [6, 7]
        }
    ]
    
    # Test mapping creation
    mappings = create_mapping_from_config(test_stage_defs)
    print("Test mappings:", mappings)
    
    # Test seen classes mask
    mask_stage1 = get_seen_classes_mask(test_stage_defs, 1) 
    mask_stage2 = get_seen_classes_mask(test_stage_defs, 2)
    print(f"Stage 1 mask (first 10): {mask_stage1[:10]}")
    print(f"Stage 2 mask (first 10): {mask_stage2[:10]}")
    
    print("✅ Mapping utilities test passed!")
