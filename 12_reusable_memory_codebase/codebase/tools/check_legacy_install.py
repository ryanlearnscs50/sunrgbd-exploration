"""Verify the documented legacy environment, including real CUDA operations.

Run with the legacy interpreter after installation, from any directory. This
does not load a dataset or checkpoint and does not validate detector accuracy.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def check():
    expected = {
        'torch': '1.12.1+cu113', 'torchvision': '0.13.1+cu113',
        'mmcv-full': '1.6.0', 'mmdet': '2.24.1',
        'mmsegmentation': '0.24.1', 'MinkowskiEngine': '0.5.4',
        'numpy': '1.23.5', 'numba': '0.56.4', 'networkx': '2.8.8',
        'opencv-python': '4.8.1.78', 'yapf': '0.40.1',
    }
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError('the documented legacy installation requires Python 3.9')
    versions = {name: importlib.metadata.version(name) for name in expected}
    mismatches = {name: (versions[name], wanted) for name, wanted in expected.items()
                  if versions[name] != wanted}
    if mismatches:
        raise RuntimeError(f'packages differ from the legacy profile: {mismatches}')
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)

    import cv2
    import mmcv
    import mmdet3d
    import MinkowskiEngine as ME
    import numpy as np
    import torch
    from mmcv.ops import nms3d

    if not torch.cuda.is_available():
        raise RuntimeError('a CUDA GPU is required for native installation validation')
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                               dtype=torch.int32)
    features = torch.randn(3, 4, device='cuda', requires_grad=True)
    sparse = ME.SparseTensor(features=features, coordinates=coordinates, device='cuda')
    conv = ME.MinkowskiConvolution(4, 2, kernel_size=3, dimension=3).cuda()
    output = conv(sparse)
    output.F.square().sum().backward()
    for name, value in [('output', output.F), ('input gradient', features.grad),
                        ('kernel gradient', conv.kernel.grad)]:
        if value is None or not torch.isfinite(value).all():
            raise RuntimeError(f'sparse CUDA {name} is missing or nonfinite')
    boxes = torch.tensor([[0., 0., 0., 1., 1., 1., 0.]], device='cuda')
    if nms3d(boxes, torch.ones(1, device='cuda'), 0.5).tolist() != [0]:
        raise RuntimeError('MMCV CUDA NMS returned an unexpected result')
    if cv2.resize(np.zeros((4, 4, 3), dtype=np.uint8), (2, 2)).shape != (2, 2, 3):
        raise RuntimeError('OpenCV resize returned an unexpected result')
    with tempfile.TemporaryDirectory() as directory:
        config_path = str(Path(directory) / 'config.py')
        mmcv.Config(dict(seed=200)).dump(config_path)
        if mmcv.Config.fromfile(config_path).seed != 200:
            raise RuntimeError('MMCV config serialization failed to round trip')
    torch.cuda.synchronize()
    return dict(status='passed', python=sys.version, executable=sys.executable,
                packages=versions, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), dependency_check=result.stdout.strip(),
                modules={module.__name__: module.__file__
                         for module in (torch, ME, mmcv, mmdet3d, cv2)},
                checks=['dependency_metadata', 'sparse_cuda_forward_backward',
                        'mmcv_cuda_nms', 'opencv_resize', 'mmcv_config_roundtrip'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='exclusive JSON evidence file')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('output already exists')
    report = check()
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        with args.output.open('x') as stream:
            stream.write(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
