"""SUN RGB-D in-memory dataset for reviewing/evaluation.

This dataset is used to evaluate models on a *subset* of SUNRGBD scenes
represented directly by `data_infos` dicts (e.g. SceneMemoryBank seats),
without reading from an annotation pkl on disk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .builder import DATASETS
from .sunrgbd_dataset import SUNRGBDDataset


@DATASETS.register_module()
class SUNRGBDMemoryDataset(SUNRGBDDataset):
    """A SUNRGBD dataset backed by a provided `data_infos` list."""

    def __init__(self,
                 data_infos: List[Dict[str, Any]],
                 data_root: str,
                 pipeline=None,
                 classes=None,
                 modality=dict(use_camera=False, use_lidar=True),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = False,
                 test_mode: bool = True,
                 **kwargs):
        self._provided_data_infos = list(data_infos) if data_infos is not None else []
        # `ann_file` is unused because we override `load_annotations`, but must be
        # present for the parent class init path.
        super().__init__(
            data_root=data_root,
            ann_file='__memory__',
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs,
        )

    def load_annotations(self, ann_file: str) -> List[Dict[str, Any]]:
        # Ignore ann_file; return the provided dicts.
        return list(self._provided_data_infos)

    def set_data_infos(self, data_infos: List[Dict[str, Any]]) -> None:
        """Replace data_infos in-place (useful for iterative reviewing)."""
        self.data_infos = list(data_infos) if data_infos is not None else []
        if hasattr(self, 'flag'):
            # Match other datasets that reset flags on mutation.
            import numpy as np
            self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

