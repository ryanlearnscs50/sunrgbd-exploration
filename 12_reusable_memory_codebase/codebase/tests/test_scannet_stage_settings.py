import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING_DIR = REPO_ROOT / 'configs' / '_base_' / 'class_mappings'
if str(MAPPING_DIR) not in sys.path:
    sys.path.append(str(MAPPING_DIR))

from scannet_dynamic_head_mappings import (  # type: ignore  # noqa: E402
    SCANNET_35_DEFAULT_STAGE_SETTING,
    get_stage_definitions,
    validate_scannet_35class_mapping,
)


def _concat(stage_definitions, key):
    out = []
    for sd in stage_definitions:
        out.extend(list(sd[key]))
    return out


def test_scannet_stage_setting_s5_backward_compatible():
    stages = get_stage_definitions('frequency')
    stages_explicit = get_stage_definitions(
        strategy='frequency',
        stage_setting='scannet35_s5_freqorder',
    )

    assert validate_scannet_35class_mapping(
        strategy='frequency',
        stage_setting='scannet35_s5_freqorder',
    ) is True

    assert len(stages) == 5
    assert [int(sd['stage_id']) for sd in stages] == [1, 2, 3, 4, 5]
    assert [len(sd['class_indices']) for sd in stages] == [7, 7, 7, 7, 7]

    assert _concat(stages, 'class_indices') == list(range(35))
    assert _concat(stages, 'class_names') == _concat(stages_explicit, 'class_names')
    assert _concat(stages, 'nyu40_ids') == _concat(stages_explicit, 'nyu40_ids')



def test_scannet_stage_setting_s3_15_10_10():
    stages_s3 = get_stage_definitions(
        strategy='frequency',
        stage_setting='scannet35_s3_freqorder_15_10_10',
    )
    stages_s5 = get_stage_definitions(
        strategy='frequency',
        stage_setting='scannet35_s5_freqorder',
    )

    assert validate_scannet_35class_mapping(
        strategy='frequency',
        stage_setting='scannet35_s3_freqorder_15_10_10',
    ) is True

    assert len(stages_s3) == 3
    assert [int(sd['stage_id']) for sd in stages_s3] == [1, 2, 3]
    assert [len(sd['class_indices']) for sd in stages_s3] == [15, 10, 10]

    assert _concat(stages_s3, 'class_indices') == list(range(35))
    assert _concat(stages_s3, 'class_names') == _concat(stages_s5, 'class_names')
    assert _concat(stages_s3, 'nyu40_ids') == _concat(stages_s5, 'nyu40_ids')



def test_scannet_stage_setting_s10_4444433333():
    stages_s10 = get_stage_definitions(
        strategy='frequency',
        stage_setting='scannet35_s10_freqorder_4444433333',
    )
    stages_s5 = get_stage_definitions(
        strategy='frequency',
        stage_setting='scannet35_s5_freqorder',
    )

    assert validate_scannet_35class_mapping(
        strategy='frequency',
        stage_setting='scannet35_s10_freqorder_4444433333',
    ) is True

    assert len(stages_s10) == 10
    assert [int(sd['stage_id']) for sd in stages_s10] == list(range(1, 11))
    assert [len(sd['class_indices']) for sd in stages_s10] == [4, 4, 4, 4, 4, 3, 3, 3, 3, 3]

    assert _concat(stages_s10, 'class_indices') == list(range(35))
    assert _concat(stages_s10, 'class_names') == _concat(stages_s5, 'class_names')
    assert _concat(stages_s10, 'nyu40_ids') == _concat(stages_s5, 'nyu40_ids')



def test_stage_setting_positional_compatibility():
    default_setting = SCANNET_35_DEFAULT_STAGE_SETTING
    by_setting_positional = get_stage_definitions('scannet35_s3_freqorder_15_10_10')
    by_keyword = get_stage_definitions(
        strategy='frequency',
        stage_setting='scannet35_s3_freqorder_15_10_10',
    )
    by_default = get_stage_definitions(strategy='frequency', stage_setting=default_setting)

    assert _concat(by_setting_positional, 'class_names') == _concat(by_keyword, 'class_names')
    assert len(by_default) == 5
