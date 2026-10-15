import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING_DIR = REPO_ROOT / 'configs' / '_base_' / 'class_mappings'
if str(MAPPING_DIR) not in sys.path:
    sys.path.append(str(MAPPING_DIR))

from sunrgbd_40class_mapping import (  # type: ignore  # noqa: E402
    SUNRGBD_40_RAW_TOP40_CLASSES,
    get_stage_definitions,
    validate_sunrgbd_40class_mapping,
)


def _concat(stage_definitions, key):
    out = []
    for sd in stage_definitions:
        out.extend(list(sd[key]))
    return out


def test_sunrgbd_stage_setting_s5_freqorder():
    stages = get_stage_definitions(stage_setting='sunrgbd40_s5_freqorder')
    assert validate_sunrgbd_40class_mapping(stage_setting='sunrgbd40_s5_freqorder') is True

    assert len(stages) == 5
    assert [int(sd['stage_id']) for sd in stages] == [1, 2, 3, 4, 5]
    assert [len(sd['class_indices']) for sd in stages] == [8, 8, 8, 8, 8]

    indices = _concat(stages, 'class_indices')
    names = _concat(stages, 'class_names')
    assert indices == list(range(40))
    assert names == list(SUNRGBD_40_RAW_TOP40_CLASSES)

    assert len(set(indices)) == 40
    assert len(set(names)) == 40


def test_sunrgbd_stage_setting_s10_freqorder_split():
    stages_s5 = get_stage_definitions(stage_setting='sunrgbd40_s5_freqorder')
    stages_s10 = get_stage_definitions(stage_setting='sunrgbd40_s10_freqorder_split')
    assert validate_sunrgbd_40class_mapping(stage_setting='sunrgbd40_s10_freqorder_split') is True

    assert len(stages_s10) == 10
    assert [int(sd['stage_id']) for sd in stages_s10] == list(range(1, 11))
    assert [len(sd['class_indices']) for sd in stages_s10] == [4] * 10

    indices = _concat(stages_s10, 'class_indices')
    names = _concat(stages_s10, 'class_names')
    assert indices == list(range(40))
    assert names == list(SUNRGBD_40_RAW_TOP40_CLASSES)

    assert len(set(indices)) == 40
    assert len(set(names)) == 40

    # Pairwise-merge 4×10 stages reproduces the original 8×5 grouping.
    for i in range(5):
        s10_pair_indices = list(stages_s10[i * 2]['class_indices']) + list(stages_s10[i * 2 + 1]['class_indices'])
        s10_pair_names = list(stages_s10[i * 2]['class_names']) + list(stages_s10[i * 2 + 1]['class_names'])
        assert s10_pair_indices == list(stages_s5[i]['class_indices'])
        assert s10_pair_names == list(stages_s5[i]['class_names'])

