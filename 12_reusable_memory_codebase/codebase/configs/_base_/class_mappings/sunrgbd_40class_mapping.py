"""
SUN RGB-D 40-class (raw top-40) mapping utilities.

This file defines the SUN RGB-D class order for:
- Fully supervised 40-class training (upper bound)
- Incremental learning stage settings that preserve the same global class order:
  - 8-class × 5-stage (`sunrgbd40_s5_freqorder`)
  - 4-class × 10-stage (`sunrgbd40_s10_freqorder_split`)
  - 20/10/10 × 3-stage (`sunrgbd40_s3_20_10_10_freqorder`)

IMPORTANT:
- The class list is derived from the SUN RGB-D *train split only* to avoid
  validation/test leakage.
- The class names are treated as raw tokens (lowercased/stripped) from
  `sunrgbd_trainval/label/*.txt` first-column entries.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Top-40 raw classes in descending order of *train instance count*.
SUNRGBD_40_RAW_TOP40_CLASSES: List[str] = [
    # Stage 1 (0-7)
    'chair',
    'table',
    'pillow',
    'sofa_chair',
    'desk',
    'bed',
    'sofa',
    'computer',
    # Stage 2 (8-15)
    'lamp',
    'box',
    'garbage_bin',
    'cabinet',
    'shelf',
    'drawer',
    'night_stand',
    'endtable',
    # Stage 3 (16-23)
    'sink',
    'picture',
    'stool',
    'coffee_table',
    'bookshelf',
    'painting',
    'keyboard',
    'dresser',
    # Stage 4 (24-31)
    'tv',
    'whiteboard',
    'cpu',
    'toilet',
    'paper',
    'ottoman',
    'bench',
    'recycle_bin',
    # Stage 5 (32-39)
    'monitor',
    'printer',
    'plant',
    'door',
    'book',
    'mirror',
    'laptop',
    'towel',
]

SUNRGBD_40_STAGE_SETTINGS: Dict[str, Dict[str, Any]] = {
    # Existing default (kept for backward compatibility).
    'sunrgbd40_s5_freqorder': {'num_stages': 5, 'stage_size': 8},
    # New: split each 8-class stage into two 4-class sub-stages.
    'sunrgbd40_s10_freqorder_split': {'num_stages': 10, 'stage_size': 4},
    # New: 3-stage split with non-uniform stage sizes (20/10/10).
    'sunrgbd40_s3_20_10_10_freqorder': {'num_stages': 3, 'stage_sizes': [20, 10, 10]},
}

SUNRGBD_40_DEFAULT_STAGE_SETTING = 'sunrgbd40_s5_freqorder'


# Train-only statistics (instances + samples), aligned with the class order above.
SUNRGBD_40_RAW_TOP40_STATISTICS: Dict[str, Dict[str, int]] = {
    'chair': {'instances_train': 9278, 'samples_train': 2429, 'freq_rank': 1},
    'table': {'instances_train': 2539, 'samples_train': 1678, 'freq_rank': 2},
    'pillow': {'instances_train': 1564, 'samples_train': 518, 'freq_rank': 3},
    'sofa_chair': {'instances_train': 1056, 'samples_train': 511, 'freq_rank': 4},
    'desk': {'instances_train': 933, 'samples_train': 611, 'freq_rank': 5},
    'bed': {'instances_train': 771, 'samples_train': 622, 'freq_rank': 6},
    'sofa': {'instances_train': 706, 'samples_train': 499, 'freq_rank': 7},
    'computer': {'instances_train': 682, 'samples_train': 383, 'freq_rank': 8},
    'lamp': {'instances_train': 519, 'samples_train': 387, 'freq_rank': 9},
    'box': {'instances_train': 487, 'samples_train': 299, 'freq_rank': 10},
    'garbage_bin': {'instances_train': 470, 'samples_train': 396, 'freq_rank': 11},
    'cabinet': {'instances_train': 400, 'samples_train': 330, 'freq_rank': 12},
    'shelf': {'instances_train': 342, 'samples_train': 269, 'freq_rank': 13},
    'drawer': {'instances_train': 297, 'samples_train': 246, 'freq_rank': 14},
    'night_stand': {'instances_train': 293, 'samples_train': 247, 'freq_rank': 15},
    'endtable': {'instances_train': 260, 'samples_train': 218, 'freq_rank': 16},
    'sink': {'instances_train': 222, 'samples_train': 193, 'freq_rank': 17},
    'picture': {'instances_train': 217, 'samples_train': 135, 'freq_rank': 18},
    'stool': {'instances_train': 214, 'samples_train': 94, 'freq_rank': 19},
    'coffee_table': {'instances_train': 212, 'samples_train': 199, 'freq_rank': 20},
    'bookshelf': {'instances_train': 204, 'samples_train': 157, 'freq_rank': 21},
    'painting': {'instances_train': 195, 'samples_train': 141, 'freq_rank': 22},
    'keyboard': {'instances_train': 185, 'samples_train': 150, 'freq_rank': 23},
    'dresser': {'instances_train': 182, 'samples_train': 147, 'freq_rank': 24},
    'tv': {'instances_train': 181, 'samples_train': 178, 'freq_rank': 25},
    'whiteboard': {'instances_train': 180, 'samples_train': 168, 'freq_rank': 26},
    'cpu': {'instances_train': 180, 'samples_train': 114, 'freq_rank': 27},
    'toilet': {'instances_train': 171, 'samples_train': 168, 'freq_rank': 28},
    'paper': {'instances_train': 155, 'samples_train': 106, 'freq_rank': 29},
    'ottoman': {'instances_train': 151, 'samples_train': 121, 'freq_rank': 30},
    'bench': {'instances_train': 138, 'samples_train': 109, 'freq_rank': 31},
    'recycle_bin': {'instances_train': 131, 'samples_train': 123, 'freq_rank': 32},
    'monitor': {'instances_train': 129, 'samples_train': 83, 'freq_rank': 33},
    'printer': {'instances_train': 120, 'samples_train': 103, 'freq_rank': 34},
    'plant': {'instances_train': 112, 'samples_train': 85, 'freq_rank': 35},
    'door': {'instances_train': 111, 'samples_train': 100, 'freq_rank': 36},
    'book': {'instances_train': 108, 'samples_train': 84, 'freq_rank': 37},
    'mirror': {'instances_train': 98, 'samples_train': 87, 'freq_rank': 38},
    'laptop': {'instances_train': 93, 'samples_train': 92, 'freq_rank': 39},
    'towel': {'instances_train': 91, 'samples_train': 68, 'freq_rank': 40},
}


def _validate_sunrgbd_stage_definitions(*,
                                       stage_setting: str,
                                       stage_definitions: List[Dict[str, Any]],
                                       verbose: bool = False) -> None:
    """Fail-fast validation for SUNRGBD stage definitions derived from this module."""
    assert stage_setting in SUNRGBD_40_STAGE_SETTINGS, stage_setting
    cfg = SUNRGBD_40_STAGE_SETTINGS[stage_setting]
    expected_num_stages = int(cfg['num_stages'])
    if cfg.get('stage_sizes', None) is not None:
        expected_stage_sizes = [int(x) for x in list(cfg['stage_sizes'])]
        assert len(expected_stage_sizes) == expected_num_stages, (
            stage_setting, expected_stage_sizes, expected_num_stages
        )
    else:
        expected_stage_sizes = [int(cfg['stage_size'])] * expected_num_stages

    assert len(SUNRGBD_40_RAW_TOP40_CLASSES) == 40
    assert len(set(SUNRGBD_40_RAW_TOP40_CLASSES)) == 40

    assert len(stage_definitions) == expected_num_stages, (
        stage_setting, len(stage_definitions), expected_num_stages
    )

    expected_stage_ids = list(range(1, expected_num_stages + 1))
    stage_ids = [int(sd.get('stage_id', -1)) for sd in stage_definitions]
    assert stage_ids == expected_stage_ids, (stage_setting, stage_ids, expected_stage_ids)

    all_indices: List[int] = []
    all_names: List[str] = []
    for idx, sd in enumerate(stage_definitions):
        expected_stage_size = int(expected_stage_sizes[idx])
        indices = [int(x) for x in sd.get('class_indices', [])]
        names = list(sd.get('class_names', []))
        assert len(indices) == expected_stage_size, (
            stage_setting, int(sd.get('stage_id', -1)), len(indices), expected_stage_size
        )
        assert len(names) == expected_stage_size, (
            stage_setting, int(sd.get('stage_id', -1)), len(names), expected_stage_size
        )
        all_indices.extend(indices)
        all_names.extend(names)

    assert len(all_indices) == 40
    assert len(all_names) == 40
    assert len(set(all_indices)) == 40
    assert len(set(all_names)) == 40
    assert all_indices == list(range(40))
    assert all_names == SUNRGBD_40_RAW_TOP40_CLASSES

    if verbose:
        print(f"✅ SUNRGBD stage definitions validated: {stage_setting}")


def get_stage_definitions(stage_setting: str = SUNRGBD_40_DEFAULT_STAGE_SETTING
                          ) -> List[Dict[str, Any]]:
    """Return stage definitions (frequency-ordered) for SUN RGB-D top-40."""
    if len(SUNRGBD_40_RAW_TOP40_CLASSES) != 40:
        raise ValueError(
            f"Expected 40 classes, got {len(SUNRGBD_40_RAW_TOP40_CLASSES)}"
        )

    if stage_setting not in SUNRGBD_40_STAGE_SETTINGS:
        raise ValueError(
            f"Unknown stage_setting={stage_setting!r}. "
            f"Supported: {sorted(SUNRGBD_40_STAGE_SETTINGS.keys())}"
        )
    cfg = SUNRGBD_40_STAGE_SETTINGS[stage_setting]
    num_stages = int(cfg['num_stages'])
    if cfg.get('stage_sizes', None) is not None:
        stage_sizes = [int(x) for x in list(cfg['stage_sizes'])]
        if len(stage_sizes) != num_stages:
            raise ValueError(
                f"Invalid stage_sizes for {stage_setting!r}: {stage_sizes}"
            )
    else:
        stage_sizes = [int(cfg['stage_size'])] * num_stages

    stage_definitions: List[Dict[str, Any]] = []
    start_idx = 0
    for stage_id in range(1, num_stages + 1):
        stage_size = int(stage_sizes[stage_id - 1])
        end_idx = start_idx + stage_size

        class_names = SUNRGBD_40_RAW_TOP40_CLASSES[start_idx:end_idx]
        class_indices = list(range(start_idx, end_idx))  # 0-39
        total_instances = int(
            sum(SUNRGBD_40_RAW_TOP40_STATISTICS[n]['instances_train']
                for n in class_names)
        )

        stage_definitions.append(
            dict(
                stage_id=stage_id,
                # Keep the legacy 8×5 stage_name stable for backward compatibility.
                stage_name=(
                    f"Stage {stage_id} - Frequency Ordering (Top-40 Raw)"
                    if stage_setting == 'sunrgbd40_s5_freqorder' else
                    (
                        f"Stage {stage_id} - Frequency Ordering (Top-40 Raw) [4×10 split]"
                        if stage_setting == 'sunrgbd40_s10_freqorder_split' else
                        (
                            f"Stage {stage_id} - Frequency Ordering (Top-40 Raw) [20/10/10 split]"
                            if stage_setting == 'sunrgbd40_s3_20_10_10_freqorder' else
                            f"Stage {stage_id} - Frequency Ordering (Top-40 Raw) [{stage_setting}]"
                        )
                    )
                ),
                class_indices=class_indices,
                class_names=class_names,
                statistics=dict(total_instances_train=total_instances),
            )
        )
        start_idx = end_idx

    _validate_sunrgbd_stage_definitions(
        stage_setting=stage_setting,
        stage_definitions=stage_definitions,
        verbose=False,
    )
    return stage_definitions


def validate_sunrgbd_40class_mapping(
        stage_setting: str = SUNRGBD_40_DEFAULT_STAGE_SETTING,
        verbose: bool = False) -> bool:
    """Validate that the mapping and stage split are internally consistent."""
    assert len(SUNRGBD_40_RAW_TOP40_CLASSES) == 40
    assert len(set(SUNRGBD_40_RAW_TOP40_CLASSES)) == 40
    for cls in SUNRGBD_40_RAW_TOP40_CLASSES:
        assert cls in SUNRGBD_40_RAW_TOP40_STATISTICS

    stages = get_stage_definitions(stage_setting=stage_setting)
    _validate_sunrgbd_stage_definitions(
        stage_setting=stage_setting,
        stage_definitions=stages,
        verbose=verbose,
    )
    all_indices = []
    all_names = []
    for stage in stages:
        all_indices.extend(stage['class_indices'])
        all_names.extend(stage['class_names'])
    assert all_indices == list(range(40))
    assert all_names == SUNRGBD_40_RAW_TOP40_CLASSES

    if verbose:
        print(f"✅ SUNRGBD 40-class mapping validated ({stage_setting})")
        for stage in stages:
            print(
                f"Stage {stage['stage_id']}: "
                f"{stage['class_indices'][0]}-{stage['class_indices'][-1]} "
                f"({stage['statistics']['total_instances_train']} instances)"
            )

    return True


if __name__ == '__main__':
    validate_sunrgbd_40class_mapping(stage_setting='sunrgbd40_s5_freqorder', verbose=True)
    validate_sunrgbd_40class_mapping(stage_setting='sunrgbd40_s10_freqorder_split', verbose=True)
    validate_sunrgbd_40class_mapping(stage_setting='sunrgbd40_s3_20_10_10_freqorder', verbose=True)
