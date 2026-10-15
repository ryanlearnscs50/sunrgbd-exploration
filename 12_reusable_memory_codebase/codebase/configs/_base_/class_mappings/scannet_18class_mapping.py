# ScanNet 18-Class Training Configuration
# Traditional 18-class subset commonly used in 3D object detection research

from scannet_nyu40_mapping import NYU40_ID_TO_NAME

# 18 classes used in traditional ScanNet training
SCANNET_18_CLASSES = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin'
]

# NYU40 IDs corresponding to the 18 classes
# Note: 'showercurtrain' maps to 'shower_curtain' (NYU40 ID 28)
# Note: 'garbagebin' is a custom mapping - need to verify actual NYU40 correspondence
SCANNET_18_NYU40_IDS = [
    3,   # cabinet
    4,   # bed  
    5,   # chair
    6,   # sofa
    7,   # table
    8,   # door
    9,   # window
    10,  # bookshelf
    11,  # picture
    12,  # counter
    14,  # desk
    16,  # curtain
    24,  # refrigerator
    28,  # shower_curtain (showercurtrain)
    33,  # toilet
    34,  # sink
    36,  # bathtub
    # Note: garbagebin needs proper NYU40 mapping - using otherprop as placeholder
    40   # otherprop (placeholder for garbagebin)
]

# Mapping from NYU40 ID to 18-class model index (0-17)
NYU40_TO_18CLASS_MODEL_IDX = {
    nyu40_id: model_idx 
    for model_idx, nyu40_id in enumerate(SCANNET_18_NYU40_IDS)
}

# Mapping from 18-class model index to NYU40 ID
MODEL_IDX_TO_NYU40_18CLASS = {
    model_idx: nyu40_id 
    for nyu40_id, model_idx in NYU40_TO_18CLASS_MODEL_IDX.items()
}

# Mapping from 18-class model index to class name
MODEL_IDX_TO_NAME_18CLASS = {
    model_idx: SCANNET_18_CLASSES[model_idx] 
    for model_idx in range(len(SCANNET_18_CLASSES))
}

# All other NYU40 classes are ignored for 18-class training
IGNORED_NYU40_IDS_18CLASS = [
    nyu40_id for nyu40_id in range(1, 41) 
    if nyu40_id not in SCANNET_18_NYU40_IDS
]

# Validation
assert len(SCANNET_18_CLASSES) == 18
assert len(SCANNET_18_NYU40_IDS) == 18
assert len(NYU40_TO_18CLASS_MODEL_IDX) == 18
assert len(IGNORED_NYU40_IDS_18CLASS) == 22  # 40 - 18 = 22 ignored classes
assert all(0 <= idx <= 17 for idx in NYU40_TO_18CLASS_MODEL_IDX.values())