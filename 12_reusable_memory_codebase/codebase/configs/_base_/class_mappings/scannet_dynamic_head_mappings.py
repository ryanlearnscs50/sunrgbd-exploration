"""ScanNet dynamic-head class mappings with configurable stage settings.

Supported ordering strategies:
- `frequency` (recommended)
- `alphabetical`

Supported stage settings (35-class ScanNet):
- `scannet35_s5_freqorder` (7-7-7-7-7)
- `scannet35_s3_freqorder_15_10_10` (15-10-10)
- `scannet35_s10_freqorder_4444433333` (4-4-4-4-4-3-3-3-3-3)

Backward compatibility:
- `get_stage_definitions('frequency')` keeps the historical 5-stage behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


CLASS_STATISTICS: Dict[str, Dict[str, int]] = {
    'chair': {'nyu40_id': 5, 'objects': 4357, 'scenes': 798, 'freq_rank': 1},
    'door': {'nyu40_id': 8, 'objects': 2026, 'scenes': 874, 'freq_rank': 2},
    'otherfurniture': {'nyu40_id': 39, 'objects': 1985, 'scenes': 890, 'freq_rank': 3},
    'books': {'nyu40_id': 23, 'objects': 1554, 'scenes': 197, 'freq_rank': 4},
    'cabinet': {'nyu40_id': 3, 'objects': 1427, 'scenes': 564, 'freq_rank': 5},
    'table': {'nyu40_id': 7, 'objects': 1271, 'scenes': 623, 'freq_rank': 6},
    'window': {'nyu40_id': 9, 'objects': 928, 'scenes': 609, 'freq_rank': 7},
    'pillow': {'nyu40_id': 18, 'objects': 745, 'scenes': 248, 'freq_rank': 8},
    'picture': {'nyu40_id': 11, 'objects': 661, 'scenes': 321, 'freq_rank': 9},
    'box': {'nyu40_id': 29, 'objects': 657, 'scenes': 248, 'freq_rank': 10},
    'desk': {'nyu40_id': 14, 'objects': 551, 'scenes': 327, 'freq_rank': 11},
    'shelves': {'nyu40_id': 15, 'objects': 486, 'scenes': 321, 'freq_rank': 12},
    'towel': {'nyu40_id': 27, 'objects': 481, 'scenes': 171, 'freq_rank': 13},
    'sofa': {'nyu40_id': 6, 'objects': 406, 'scenes': 279, 'freq_rank': 14},
    'sink': {'nyu40_id': 34, 'objects': 390, 'scenes': 322, 'freq_rank': 15},
    'clothes': {'nyu40_id': 21, 'objects': 386, 'scenes': 170, 'freq_rank': 16},
    'lamp': {'nyu40_id': 35, 'objects': 364, 'scenes': 205, 'freq_rank': 17},
    'bed': {'nyu40_id': 4, 'objects': 307, 'scenes': 245, 'freq_rank': 18},
    'bookshelf': {'nyu40_id': 10, 'objects': 300, 'scenes': 172, 'freq_rank': 19},
    'curtain': {'nyu40_id': 16, 'objects': 292, 'scenes': 193, 'freq_rank': 20},
    'mirror': {'nyu40_id': 19, 'objects': 279, 'scenes': 235, 'freq_rank': 21},
    'bag': {'nyu40_id': 37, 'objects': 253, 'scenes': 158, 'freq_rank': 22},
    'whiteboard': {'nyu40_id': 30, 'objects': 251, 'scenes': 203, 'freq_rank': 23},
    'counter': {'nyu40_id': 12, 'objects': 216, 'scenes': 179, 'freq_rank': 24},
    'toilet': {'nyu40_id': 33, 'objects': 201, 'scenes': 189, 'freq_rank': 25},
    'nightstand': {'nyu40_id': 32, 'objects': 190, 'scenes': 148, 'freq_rank': 26},
    'refrigerator': {'nyu40_id': 24, 'objects': 186, 'scenes': 177, 'freq_rank': 27},
    'television': {'nyu40_id': 25, 'objects': 177, 'scenes': 172, 'freq_rank': 28},
    'dresser': {'nyu40_id': 17, 'objects': 170, 'scenes': 140, 'freq_rank': 29},
    'shower_curtain': {'nyu40_id': 28, 'objects': 116, 'scenes': 106, 'freq_rank': 30},
    'bathtub': {'nyu40_id': 36, 'objects': 113, 'scenes': 113, 'freq_rank': 31},
    'paper': {'nyu40_id': 26, 'objects': 52, 'scenes': 29, 'freq_rank': 32},
    'person': {'nyu40_id': 31, 'objects': 39, 'scenes': 29, 'freq_rank': 33},
    'floor_mat': {'nyu40_id': 20, 'objects': 32, 'scenes': 26, 'freq_rank': 34},
    'blinds': {'nyu40_id': 13, 'objects': 22, 'scenes': 18, 'freq_rank': 35},
}


CLASS_ORDERING_STRATEGIES: Dict[str, Dict[str, Any]] = {
    'frequency': {
        'description': 'Classes ordered by occurrence frequency (recommended)',
        'class_order': [
            'chair', 'door', 'otherfurniture', 'books', 'cabinet', 'table', 'window',
            'pillow', 'picture', 'box', 'desk', 'shelves', 'towel', 'sofa',
            'sink', 'clothes', 'lamp', 'bed', 'bookshelf', 'curtain', 'mirror',
            'bag', 'whiteboard', 'counter', 'toilet', 'nightstand', 'refrigerator',
            'television', 'dresser', 'shower_curtain', 'bathtub', 'paper', 'person',
            'floor_mat', 'blinds',
        ],
    },
    'alphabetical': {
        'description': 'Classes ordered alphabetically',
        'class_order': [
            'bag', 'bathtub', 'bed', 'blinds', 'books', 'bookshelf', 'box',
            'cabinet', 'chair', 'clothes', 'counter', 'curtain', 'desk', 'door',
            'dresser', 'floor_mat', 'lamp', 'mirror', 'nightstand', 'otherfurniture',
            'paper', 'person', 'picture', 'pillow', 'refrigerator', 'shelves',
            'shower_curtain', 'sink', 'sofa', 'table', 'television', 'toilet',
            'towel', 'whiteboard', 'window',
        ],
    },
}


SCANNET_35_STAGE_SETTINGS: Dict[str, Dict[str, Any]] = {
    'scannet35_s5_freqorder': {
        'description': 'Default 5-stage split (7-7-7-7-7)',
        'stage_sizes': [7, 7, 7, 7, 7],
    },
    'scannet35_s3_freqorder_15_10_10': {
        'description': '3-stage split (15-10-10)',
        'stage_sizes': [15, 10, 10],
    },
    'scannet35_s10_freqorder_4444433333': {
        'description': '10-stage split (4-4-4-4-4-3-3-3-3-3)',
        'stage_sizes': [4, 4, 4, 4, 4, 3, 3, 3, 3, 3],
    },
}

SCANNET_35_DEFAULT_STAGE_SETTING = 'scannet35_s5_freqorder'
_SCANNET_35_NUM_CLASSES = 35


def _resolve_strategy_and_stage_setting(
    strategy: str,
    stage_setting: Optional[str],
) -> Tuple[str, str]:
    """Resolve and validate strategy/stage_setting with compatibility behavior."""
    strategy_name = str(strategy)

    # Compatibility: allow old positional misuse: get_stage_definitions('scannet35_*').
    if strategy_name in SCANNET_35_STAGE_SETTINGS and stage_setting is None:
        return 'frequency', strategy_name

    stage_setting_name = (
        SCANNET_35_DEFAULT_STAGE_SETTING
        if stage_setting is None
        else str(stage_setting)
    )

    if strategy_name not in CLASS_ORDERING_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {sorted(CLASS_ORDERING_STRATEGIES.keys())}"
        )
    if stage_setting_name not in SCANNET_35_STAGE_SETTINGS:
        raise ValueError(
            f"Unknown stage_setting '{stage_setting_name}'. "
            f"Available: {sorted(SCANNET_35_STAGE_SETTINGS.keys())}"
        )

    return strategy_name, stage_setting_name


def _compute_stage_statistics(stage_classes: List[str]) -> Dict[str, Any]:
    objects = [int(CLASS_STATISTICS[name]['objects']) for name in stage_classes]
    scenes = [int(CLASS_STATISTICS[name]['scenes']) for name in stage_classes]
    ranks = [int(CLASS_STATISTICS[name]['freq_rank']) for name in stage_classes]
    return {
        'total_objects': int(sum(objects)),
        'max_scenes': int(max(scenes)) if scenes else 0,
        'rank_range': f"{min(ranks)}-{max(ranks)}" if ranks else 'n/a',
        'avg_rank': float(sum(ranks) / len(ranks)) if ranks else 0.0,
    }


def _validate_stage_definitions(
    *,
    strategy: str,
    stage_setting: str,
    stage_definitions: List[Dict[str, Any]],
) -> None:
    expected_sizes = list(SCANNET_35_STAGE_SETTINGS[stage_setting]['stage_sizes'])

    assert len(stage_definitions) == len(expected_sizes), (
        stage_setting,
        len(stage_definitions),
        len(expected_sizes),
    )

    stage_ids = [int(sd.get('stage_id', -1)) for sd in stage_definitions]
    assert stage_ids == list(range(1, len(expected_sizes) + 1)), stage_ids

    all_indices: List[int] = []
    all_names: List[str] = []
    all_nyu40: List[int] = []

    for idx, sd in enumerate(stage_definitions):
        indices = [int(x) for x in sd.get('class_indices', [])]
        names = list(sd.get('class_names', []))
        nyu40_ids = [int(x) for x in sd.get('nyu40_ids', [])]

        assert len(indices) == expected_sizes[idx], (idx + 1, len(indices), expected_sizes[idx])
        assert len(names) == expected_sizes[idx], (idx + 1, len(names), expected_sizes[idx])
        assert len(nyu40_ids) == expected_sizes[idx], (idx + 1, len(nyu40_ids), expected_sizes[idx])

        all_indices.extend(indices)
        all_names.extend(names)
        all_nyu40.extend(nyu40_ids)

    assert all_indices == list(range(_SCANNET_35_NUM_CLASSES)), all_indices
    assert len(set(all_names)) == _SCANNET_35_NUM_CLASSES, len(set(all_names))
    assert len(set(all_nyu40)) == _SCANNET_35_NUM_CLASSES, len(set(all_nyu40))

    expected_order = CLASS_ORDERING_STRATEGIES[strategy]['class_order']
    assert all_names == expected_order, (strategy, stage_setting)


def get_stage_definitions(
    strategy: str = 'frequency',
    stage_setting: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build stage definitions for a given ordering strategy and stage setting."""
    strategy_name, stage_setting_name = _resolve_strategy_and_stage_setting(
        strategy=strategy,
        stage_setting=stage_setting,
    )

    class_order = list(CLASS_ORDERING_STRATEGIES[strategy_name]['class_order'])
    if len(class_order) != _SCANNET_35_NUM_CLASSES:
        raise ValueError(
            f"Strategy '{strategy_name}' has {len(class_order)} classes, "
            f"expected {_SCANNET_35_NUM_CLASSES}."
        )

    for class_name in class_order:
        if class_name not in CLASS_STATISTICS:
            raise ValueError(
                f"Class '{class_name}' is missing statistics in CLASS_STATISTICS."
            )

    stage_sizes = list(SCANNET_35_STAGE_SETTINGS[stage_setting_name]['stage_sizes'])
    if sum(stage_sizes) != _SCANNET_35_NUM_CLASSES:
        raise ValueError(
            f"Stage setting '{stage_setting_name}' sums to {sum(stage_sizes)} classes, "
            f"expected {_SCANNET_35_NUM_CLASSES}."
        )

    stage_definitions: List[Dict[str, Any]] = []
    cursor = 0
    for stage_id, stage_size in enumerate(stage_sizes, start=1):
        end = cursor + int(stage_size)
        stage_classes = class_order[cursor:end]
        class_indices = list(range(cursor, end))
        nyu40_ids = [int(CLASS_STATISTICS[name]['nyu40_id']) for name in stage_classes]

        stage_definitions.append({
            'stage_id': int(stage_id),
            'stage_name': (
                f"Stage {stage_id} - {strategy_name.title()} Ordering "
                f"({stage_setting_name})"
            ),
            'class_indices': class_indices,
            'class_names': stage_classes,
            'nyu40_ids': nyu40_ids,
            'strategy': strategy_name,
            'stage_setting': stage_setting_name,
            'statistics': _compute_stage_statistics(stage_classes),
        })
        cursor = end

    _validate_stage_definitions(
        strategy=strategy_name,
        stage_setting=stage_setting_name,
        stage_definitions=stage_definitions,
    )
    return stage_definitions


def validate_scannet_35class_mapping(
    strategy: str = 'frequency',
    stage_setting: str = SCANNET_35_DEFAULT_STAGE_SETTING,
    verbose: bool = False,
) -> bool:
    """Validate stage definitions and class mappings for ScanNet 35-class setup."""
    stage_definitions = get_stage_definitions(
        strategy=strategy,
        stage_setting=stage_setting,
    )
    _validate_stage_definitions(
        strategy=strategy,
        stage_setting=stage_setting,
        stage_definitions=stage_definitions,
    )

    if verbose:
        print(
            f"Validated ScanNet mapping: strategy={strategy}, "
            f"stage_setting={stage_setting}, stages={len(stage_definitions)}"
        )

    return True


def get_class_statistics(class_name: str) -> Dict[str, int]:
    """Get statistics for a specific class."""
    if class_name not in CLASS_STATISTICS:
        raise ValueError(f"Unknown class '{class_name}'.")
    return dict(CLASS_STATISTICS[class_name])


def get_all_strategies() -> List[str]:
    """List available ordering strategies."""
    return sorted(CLASS_ORDERING_STRATEGIES.keys())


def get_all_stage_settings() -> List[str]:
    """List available ScanNet 35-class stage settings."""
    return sorted(SCANNET_35_STAGE_SETTINGS.keys())


def get_strategy_info(
    strategy: str,
    stage_setting: Optional[str] = None,
) -> Dict[str, Any]:
    """Get strategy metadata with stage-setting context."""
    strategy_name, stage_setting_name = _resolve_strategy_and_stage_setting(
        strategy=strategy,
        stage_setting=stage_setting,
    )
    stage_sizes = list(SCANNET_35_STAGE_SETTINGS[stage_setting_name]['stage_sizes'])
    return {
        'strategy': strategy_name,
        'description': CLASS_ORDERING_STRATEGIES[strategy_name]['description'],
        'stage_setting': stage_setting_name,
        'stage_sizes': stage_sizes,
        'num_stages': len(stage_sizes),
        'total_classes': _SCANNET_35_NUM_CLASSES,
    }


def validate_strategy_quadruples(
    strategy: str,
    stage_setting: Optional[str] = None,
) -> List[Tuple[int, int, str, int]]:
    """Get validation tuples: (stage_id, class_index, class_name, nyu40_id)."""
    stage_defs = get_stage_definitions(strategy=strategy, stage_setting=stage_setting)
    out: List[Tuple[int, int, str, int]] = []
    for stage_def in stage_defs:
        stage_id = int(stage_def['stage_id'])
        class_names = list(stage_def['class_names'])
        class_indices = [int(x) for x in stage_def['class_indices']]
        nyu40_ids = [int(x) for x in stage_def['nyu40_ids']]
        for name, idx, nyu in zip(class_names, class_indices, nyu40_ids):
            out.append((stage_id, idx, name, nyu))
    return out


# Legacy compatibility globals: default frequency + 5-stage setting.
_DEFAULT_STRATEGY = 'frequency'
_DEFAULT_STAGE_SETTING = SCANNET_35_DEFAULT_STAGE_SETTING
_DEFAULT_STAGE_DEFS = get_stage_definitions(
    strategy=_DEFAULT_STRATEGY,
    stage_setting=_DEFAULT_STAGE_SETTING,
)

SCANNET_DYNAMIC_HEAD_CLASSES: List[str] = []
NYU40_IDS_DYNAMIC_HEAD: List[int] = []
DYNAMIC_HEAD_STAGE_DEFINITIONS: List[Dict[str, Any]] = []

for _stage_def in _DEFAULT_STAGE_DEFS:
    SCANNET_DYNAMIC_HEAD_CLASSES.extend(_stage_def['class_names'])
    NYU40_IDS_DYNAMIC_HEAD.extend(_stage_def['nyu40_ids'])
    DYNAMIC_HEAD_STAGE_DEFINITIONS.append(_stage_def)

VALID_NYU40_IDS_DYNAMIC_HEAD = list(NYU40_IDS_DYNAMIC_HEAD)
IGNORED_NYU40_IDS_DYNAMIC_HEAD = [1, 2, 22, 38, 40]

NYU40_TO_DYNAMIC_HEAD_GCI = {
    int(nyu40_id): int(gci)
    for gci, nyu40_id in enumerate(NYU40_IDS_DYNAMIC_HEAD)
}

DYNAMIC_HEAD_GCI_TO_NYU40 = {
    int(gci): int(nyu40_id)
    for nyu40_id, gci in NYU40_TO_DYNAMIC_HEAD_GCI.items()
}

DYNAMIC_HEAD_GCI_TO_NAME = {
    int(gci): SCANNET_DYNAMIC_HEAD_CLASSES[int(gci)]
    for gci in range(len(SCANNET_DYNAMIC_HEAD_CLASSES))
}

NAME_TO_DYNAMIC_HEAD_GCI = {
    str(name): int(gci)
    for gci, name in DYNAMIC_HEAD_GCI_TO_NAME.items()
}

assert len(SCANNET_DYNAMIC_HEAD_CLASSES) == _SCANNET_35_NUM_CLASSES
assert len(NYU40_IDS_DYNAMIC_HEAD) == _SCANNET_35_NUM_CLASSES
assert len(DYNAMIC_HEAD_STAGE_DEFINITIONS) == 5


if __name__ == '__main__':
    validate_scannet_35class_mapping(
        strategy='frequency',
        stage_setting='scannet35_s5_freqorder',
        verbose=True,
    )
    validate_scannet_35class_mapping(
        strategy='frequency',
        stage_setting='scannet35_s3_freqorder_15_10_10',
        verbose=True,
    )
    validate_scannet_35class_mapping(
        strategy='frequency',
        stage_setting='scannet35_s10_freqorder_4444433333',
        verbose=True,
    )
