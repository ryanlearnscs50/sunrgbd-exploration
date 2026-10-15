# ScanNet 35-Class Training Configuration - **LEGACY/DEPRECATED**
# 
# ⚠️  WARNING: This is LEGACY mapping for backward compatibility ONLY
# ⚠️  Use scannet_dynamic_head_mapping.py for ALL new development
# ⚠️  This mapping only exists to support old pre-trained models
#
# Uses 40-class data but ignores 5 classes during training
# Maps remaining 35 classes to model indices 0-34 based on NYU40 ID order
# 
# For incremental learning, use the dynamic head mapping which orders
# classes by frequency/stages: 7→14→21→28→35 expansion

from scannet_nyu40_mapping import NYU40_ID_TO_NAME, NYU40_IDS

# 5 classes to ignore during 35-class training (NYU40 IDs)
IGNORED_NYU40_IDS_35CLASS = [1, 2, 22, 38, 40]  # wall, floor, ceiling, otherstructure, otherprop

# 35 valid classes for training (NYU40 IDs)
VALID_NYU40_IDS_35CLASS = [
    nyu40_id for nyu40_id in NYU40_IDS 
    if nyu40_id not in IGNORED_NYU40_IDS_35CLASS
]

# 35 class names (corresponding to valid NYU40 IDs)
SCANNET_35_CLASSES = [
    NYU40_ID_TO_NAME[nyu40_id] for nyu40_id in VALID_NYU40_IDS_35CLASS
]

# Mapping from NYU40 ID to 35-class model index (0-34)
NYU40_TO_35CLASS_MODEL_IDX = {
    nyu40_id: model_idx 
    for model_idx, nyu40_id in enumerate(VALID_NYU40_IDS_35CLASS)
}

# Mapping from 35-class model index to NYU40 ID  
MODEL_IDX_TO_NYU40_35CLASS = {
    model_idx: nyu40_id 
    for nyu40_id, model_idx in NYU40_TO_35CLASS_MODEL_IDX.items()
}

# Mapping from 35-class model index to class name
MODEL_IDX_TO_NAME_35CLASS = {
    model_idx: SCANNET_35_CLASSES[model_idx] 
    for model_idx in range(len(SCANNET_35_CLASSES))
}

# Mapping from class name to 35-class model index
NAME_TO_MODEL_IDX_35CLASS = {
    name: model_idx 
    for model_idx, name in MODEL_IDX_TO_NAME_35CLASS.items()
}

# Names of ignored classes
IGNORED_CLASS_NAMES_35CLASS = [
    NYU40_ID_TO_NAME[nyu40_id] for nyu40_id in IGNORED_NYU40_IDS_35CLASS
]

# Validation
assert len(SCANNET_35_CLASSES) == 35
assert len(VALID_NYU40_IDS_35CLASS) == 35
assert len(IGNORED_NYU40_IDS_35CLASS) == 5
assert len(NYU40_TO_35CLASS_MODEL_IDX) == 35
assert len(MODEL_IDX_TO_NYU40_35CLASS) == 35
assert all(0 <= idx <= 34 for idx in NYU40_TO_35CLASS_MODEL_IDX.values())
assert set(VALID_NYU40_IDS_35CLASS + IGNORED_NYU40_IDS_35CLASS) == set(NYU40_IDS)

# Verify ignored classes
assert IGNORED_CLASS_NAMES_35CLASS == ['wall', 'floor', 'ceiling', 'otherstructure', 'otherprop']