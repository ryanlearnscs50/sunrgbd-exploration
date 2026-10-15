"""ScanNet dynamic-head compatibility wrapper.

This module re-exports symbols from `scannet_dynamic_head_mappings.py`.
It preserves legacy imports while exposing new stage-setting APIs.
"""

from __future__ import annotations

import warnings

from scannet_nyu40_mapping import NYU40_ID_TO_NAME

try:
    from scannet_dynamic_head_mappings import (
        SCANNET_DYNAMIC_HEAD_CLASSES,
        NYU40_IDS_DYNAMIC_HEAD,
        DYNAMIC_HEAD_STAGE_DEFINITIONS,
        NYU40_TO_DYNAMIC_HEAD_GCI,
        DYNAMIC_HEAD_GCI_TO_NYU40,
        DYNAMIC_HEAD_GCI_TO_NAME,
        NAME_TO_DYNAMIC_HEAD_GCI,
        VALID_NYU40_IDS_DYNAMIC_HEAD,
        IGNORED_NYU40_IDS_DYNAMIC_HEAD,
        SCANNET_35_DEFAULT_STAGE_SETTING,
        SCANNET_35_STAGE_SETTINGS,
        get_stage_definitions,
        get_all_stage_settings,
        validate_scannet_35class_mapping,
    )
except ImportError as e:
    raise ImportError(f"Failed to import scannet_dynamic_head_mappings: {e}")


warnings.warn(
    "ScanNet class mapping wrapper in use. Prefer importing "
    "scannet_dynamic_head_mappings directly for new code.",
    UserWarning,
    stacklevel=2,
)


IGNORED_CLASS_NAMES_DYNAMIC_HEAD = [
    NYU40_ID_TO_NAME[nyu40_id] for nyu40_id in IGNORED_NYU40_IDS_DYNAMIC_HEAD
]


assert len(SCANNET_DYNAMIC_HEAD_CLASSES) == 35
assert len(NYU40_IDS_DYNAMIC_HEAD) == 35
assert len(IGNORED_NYU40_IDS_DYNAMIC_HEAD) == 5
assert len(NYU40_TO_DYNAMIC_HEAD_GCI) == 35
assert len(DYNAMIC_HEAD_GCI_TO_NYU40) == 35
assert all(0 <= gci <= 34 for gci in NYU40_TO_DYNAMIC_HEAD_GCI.values())
assert len(set(NYU40_IDS_DYNAMIC_HEAD)) == 35
assert len(set(NYU40_IDS_DYNAMIC_HEAD + IGNORED_NYU40_IDS_DYNAMIC_HEAD)) == 40


def get_frequency_ordering(stage_setting: str = SCANNET_35_DEFAULT_STAGE_SETTING):
    """Get frequency ordering for the requested stage setting."""
    return get_stage_definitions(strategy='frequency', stage_setting=stage_setting)


def get_alphabetical_ordering(stage_setting: str = SCANNET_35_DEFAULT_STAGE_SETTING):
    """Get alphabetical ordering for the requested stage setting."""
    return get_stage_definitions(strategy='alphabetical', stage_setting=stage_setting)


def get_supported_stage_settings():
    """List available stage-setting identifiers."""
    return get_all_stage_settings()
