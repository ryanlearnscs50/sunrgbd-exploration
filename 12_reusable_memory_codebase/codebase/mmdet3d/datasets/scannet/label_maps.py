"""
ScanNet Label Mapping Module

This module provides centralized, versioned mappings for ScanNet dataset variants.
It ensures consistent class ordering and mapping across the entire pipeline.

CRITICAL: These mappings MUST match exactly with the data generation and model training.
"""
import numpy as np

# Original NYU40 class names (1-40, as used in ScanNet)
# Index 0 corresponds to NYU40 class 1 (wall), index 39 corresponds to NYU40 class 40 (otherprop)
NYU40_CLASSES = (
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk',
    'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
    'clothes', 'ceiling', 'books', 'refrigerator', 'television', 'paper',
    'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'nightstand',
    'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure',
    'otherfurniture', 'otherprop'
)

# 18-class ScanNet (traditional subset)
SCANNET_18_CLASSES = (
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'showercurtain', 'toilet', 'sink', 'bathtub',
    'garbagebin'
)

# 35-class ScanNet (40 classes minus 5 ignored: wall, floor, ceiling, otherstructure, otherprop)
SCANNET_35_CLASSES = (
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 
    'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
    'clothes', 'books', 'refrigerator', 'television', 'paper',
    'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'nightstand',
    'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherfurniture'
)

# Full 40-class ScanNet (all NYU40 classes)
SCANNET_40_CLASSES = NYU40_CLASSES

# Mapping from NYU40 IDs (1-based) to train class indices (0-based)
# 18-class mapping: NYU40 IDs 1-40 -> 0-17 train indices (matches original TR3D)
SCANNET_18_MAPPING = {
    'valid_cat_ids': (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39),  # 1-based NYU40 IDs
    'max_cat_id': 40,
    'ignored_ids': (1, 2, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 30, 31, 32, 35, 37, 38, 40),
    'description': 'Traditional 18-class ScanNet subset (NYU40 IDs 1-40 to train indices 0-17)'
}

# 35-class mapping: NYU40 IDs (1-based) -> 35-class train indices (0-34)  
# Ignore: wall(NYU40:1), floor(NYU40:2), ceiling(NYU40:22), otherstructure(NYU40:38), otherprop(NYU40:40)
SCANNET_35_MAPPING = {
    'valid_cat_ids': (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39),
    'max_cat_id': 40,
    'ignored_ids': (1, 2, 22, 38, 40),  # wall, floor, ceiling, otherstructure, otherprop (1-based NYU40 IDs)
    'description': '35-class ScanNet (NYU40 IDs 1-40 mapped to 0-34 train indices)'
}

# 40-class mapping: All NYU40 IDs (1-40) -> 0-39 train indices
SCANNET_40_MAPPING = {
    'valid_cat_ids': tuple(range(1, 41)),  # 1-based NYU40 IDs
    'max_cat_id': 40,
    'ignored_ids': (),
    'description': 'Full 40-class ScanNet (NYU40 IDs 1-40 to train indices 0-39)'
}

def get_scannet_classes(variant='18'):
    """Get class names for a ScanNet variant.
    
    Args:
        variant (str): '18', 'dynamic_head', or '40'
        
    Returns:
        tuple: Class names in training order (0-based indices)
    """
    if variant == '18':
        return SCANNET_18_CLASSES
    elif variant == 'dynamic_head':
        # For dynamic_head, we need to import the dynamic head classes
        from scannet_dynamic_head_mapping import SCANNET_DYNAMIC_HEAD_CLASSES
        return SCANNET_DYNAMIC_HEAD_CLASSES
    elif variant == '40':
        return SCANNET_40_CLASSES
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be '18', 'dynamic_head', or '40'")

def get_scannet_mapping(variant='18'):
    """Get NYU40 ID to train class mapping for a ScanNet variant.
    
    Args:
        variant (str): '18', 'dynamic_head', or '40'
        
    Returns:
        dict: Mapping configuration with keys:
            - valid_cat_ids: NYU40 IDs that map to train classes
            - max_cat_id: Maximum possible NYU40 ID
            - ignored_ids: NYU40 IDs that are ignored
            - description: Human-readable description
    """
    if variant == '18':
        return SCANNET_18_MAPPING.copy()
    elif variant == 'dynamic_head':
        # For dynamic_head, return the same as 40-class mapping since it uses all 35 classes + background
        return SCANNET_40_MAPPING.copy()  
    elif variant == '40':
        return SCANNET_40_MAPPING.copy()
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be '18', 'dynamic_head', or '40'")

def validate_mapping(variant='dynamic_head'):
    """Validate that class names and mappings are consistent.
    
    Args:
        variant (str): Variant to validate
        
    Returns:
        bool: True if mapping is valid
        
    Raises:
        AssertionError: If validation fails
    """
    classes = get_scannet_classes(variant)
    mapping = get_scannet_mapping(variant)
    
    # Check that number of classes matches number of valid IDs
    # Skip this check for dynamic_head as it uses incremental learning mapping
    if variant != 'dynamic_head':
        assert len(classes) == len(mapping['valid_cat_ids']), \
            f"Mismatch: {len(classes)} classes vs {len(mapping['valid_cat_ids'])} valid IDs"
    
    # NYU40 category ids are 1-based and run up to max_cat_id inclusive.
    assert all(1 <= id <= mapping['max_cat_id'] for id in mapping['valid_cat_ids']), \
        f"Valid IDs must be 1-{mapping['max_cat_id']}"
    
    # Check no overlap between valid and ignored IDs
    valid_set = set(mapping['valid_cat_ids'])
    ignored_set = set(mapping['ignored_ids'])
    assert valid_set.isdisjoint(ignored_set), \
        f"Overlap between valid and ignored IDs: {valid_set & ignored_set}"
    
    # Check 35-class specific invariants (using 1-based NYU40 IDs)
    if variant == '35':
        assert 1 not in mapping['valid_cat_ids'], "wall (NYU40:1) should be ignored in 35-class"
        assert 2 not in mapping['valid_cat_ids'], "floor (NYU40:2) should be ignored in 35-class"
        assert 22 not in mapping['valid_cat_ids'], "ceiling (NYU40:22) should be ignored in 35-class"
        assert 38 not in mapping['valid_cat_ids'], "otherstructure (NYU40:38) should be ignored in 35-class"
        assert 40 not in mapping['valid_cat_ids'], "otherprop (NYU40:40) should be ignored in 35-class"
    
    return True

# Verify mappings on import
validate_mapping('18')
validate_mapping('dynamic_head')  # 35-class dynamic head variant
validate_mapping('40')