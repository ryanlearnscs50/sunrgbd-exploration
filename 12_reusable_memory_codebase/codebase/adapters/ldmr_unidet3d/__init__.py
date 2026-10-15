"""LDMR protocol adapters; importing this package never imports either host.

Add repo/adapters, rather than repo, to PYTHONPATH in the UniDet3D environment.
Host registry integration is explicit via ldmr_unidet3d.transforms.
"""
from .protocol import ScanNet35Protocol, remap_classifier
from .targets import stage_targets

__all__ = ['ScanNet35Protocol', 'remap_classifier', 'stage_targets']
