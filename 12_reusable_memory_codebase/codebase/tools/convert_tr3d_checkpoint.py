#!/usr/bin/env python3
"""Convert published OpenMMLab TR3D weights for strict legacy-model inference.

Only the known head-name/bias schema is supported. Preserve the original file;
the output omits optimizer state and is not a training-resume checkpoint.
"""
import argparse
import copy
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refusing to overwrite')
    import torch
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmdet3d.utils.tr3d_checkpoint import convert_openmmlab_tr3d_state
    from tools.evaluation_artifacts import file_identity

    cfg = Config.fromfile(str(args.config))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    classes = tuple(checkpoint.get('meta', {}).get('CLASSES', ()))
    if not classes or classes != tuple(cfg.data.test.classes):
        raise ValueError('checkpoint and evaluation config must declare identical ordered classes')
    model = build_model(cfg.model)
    converted = convert_openmmlab_tr3d_state(checkpoint['state_dict'], model.state_dict())
    model.load_state_dict(converted, strict=True)
    meta = copy.deepcopy(checkpoint.get('meta', {}))
    meta['ldmr_inference_conversion'] = dict(source=file_identity(args.checkpoint),
        config=file_identity(args.config), head_names='conv_reg -> bbox_conv; conv_cls -> cls_conv',
        classifier_bias='zero; source classifier has no bias', optimizer='not transferred')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(dict(state_dict=converted, meta=meta), stream)
        os.link(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(args.output)


if __name__ == '__main__':
    main()
