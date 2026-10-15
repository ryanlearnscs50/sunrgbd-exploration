# ScanNet NYU40 Complete Class Mapping Configuration
# This file defines the complete mapping between NYU40 IDs (1-40) and class names
# Used as the foundation for all ScanNet training configurations

# Complete NYU40 class names (indices 0-39 correspond to NYU40 IDs 1-40)
NYU40_CLASSES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk',
    'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
    'clothes', 'ceiling', 'books', 'refrigerator', 'television', 'paper',
    'towel', 'shower_curtain', 'box', 'whiteboard', 'person', 'nightstand',
    'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure',
    'otherfurniture', 'otherprop'
]

# NYU40 IDs (1-40) - the original ScanNet annotation IDs
NYU40_IDS = list(range(1, 41))

# Mapping from NYU40 ID to class name
NYU40_ID_TO_NAME = {nyu40_id: NYU40_CLASSES[nyu40_id - 1] for nyu40_id in NYU40_IDS}

# Mapping from class name to NYU40 ID
NYU40_NAME_TO_ID = {name: nyu40_id for nyu40_id, name in NYU40_ID_TO_NAME.items()}

# Validation
assert len(NYU40_CLASSES) == 40
assert len(NYU40_ID_TO_NAME) == 40
assert len(NYU40_NAME_TO_ID) == 40
assert all(nyu40_id in range(1, 41) for nyu40_id in NYU40_IDS)