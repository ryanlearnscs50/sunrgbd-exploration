"""ScanNet in-memory dataset for reviewing/evaluation subsets."""

from __future__ import annotations

from typing import Any, Dict, List

from .builder import DATASETS
from .scannet_dataset import ScanNetDataset


@DATASETS.register_module()
class ScanNetMemoryDataset(ScanNetDataset):
    """A ScanNet dataset backed by provided `data_infos` list.

    This mirrors `SUNRGBDMemoryDataset` usage for reviewing or focused
    subset evaluations without reading a pkl file from disk.
    """

    def __init__(self,
                 data_infos: List[Dict[str, Any]],
                 data_root: str,
                 pipeline=None,
                 classes=None,
                 box_type_3d: str = 'Depth',
                 variant: str = 'dynamic_head',
                 filter_empty_gt: bool = False,
                 test_mode: bool = True,
                 **kwargs):
        self._provided_data_infos = list(data_infos) if data_infos is not None else []
        super().__init__(
            data_root=data_root,
            ann_file='__memory__',
            pipeline=pipeline,
            classes=classes,
            box_type_3d=box_type_3d,
            variant=variant,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs,
        )

    def load_annotations(self, ann_file: str) -> List[Dict[str, Any]]:
        # Ignore ann_file; return provided data infos.
        return list(self._provided_data_infos)

    def set_data_infos(self, data_infos: List[Dict[str, Any]]) -> None:
        self.data_infos = list(data_infos) if data_infos is not None else []
        if hasattr(self, 'flag'):
            import numpy as np
            self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
