# Isolated UniDet3D runtime

Keep this environment separate from LDMR's MMDetection3D 0.x runtime. Both the
venv and native-extension build directories must physically reside on an
executable filesystem. A workspace symlink to that location is fine. A
`noexec` mount causes PyTorch shared-library mapping and CMake execution to
fail even when the Python interpreter itself starts.

The current installation uses Python 3.9, PyTorch 2.1.2+cu121,
torchvision 0.16.2+cu121, NumPy 1.24.1, SciPy 1.10.1, MMEngine 0.9.0,
MMCV 2.1.0, MMDetection 3.3.0, MMDetection3D 1.4.0, spconv-cu120 2.3.6,
torch-scatter 2.1.2+pt21cu121, plyfile 1.0.2, CMake 3.28.4,
ninja 1.11.1, and pybind11 2.10.4. Install matching PyTorch/torchvision CUDA
wheels from https://download.pytorch.org/whl/cu121, MMCV from
https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html, and
scatter from https://data.pyg.org/whl/torch-2.1.0+cu121.html.

Native source pins:

- UniDet3D: https://github.com/filapro/unidet3d at
  `940a730a09711b0bf266fd972504da29a83b91f6`.
- Segmentator: https://github.com/Karbo123/segmentator at
  `76efe46d03dd27afa78df972b17d07f2c6cfb696`.
  Set its CMake C++ standard to 17 for PyTorch 2.1. Configure Torch's
  `torch.utils.cmake_prefix_path`, the matching Python development include
  directory and shared library, and install prefix from
  `sysconfig.get_path("purelib")`. Its install creates a symlink to the source,
  so keep that source/build directory on an executable filesystem.
- MinkowskiEngine: https://github.com/daizhirui/MinkowskiEngine at
  `ce930eeb403a8e3f99693662ec5ce329a0ab3528` (already includes CUDA 12 Thrust
  include fixes). Build with OpenBLAS, matching Python development headers,
  `CUDA_HOME`, and the target GPU architecture (`8.6` for this host's RTX 3090).

The local compiler is GCC 9.4 with CUDA toolkit 12.6; PyTorch's CUDA version is
12.1. This minor-version mismatch emits a build warning. The supplied
[minkowski_cuda126_gcc9.patch](minkowski_cuda126_gcc9.patch) disables conflicting
NVTX profiling headers and avoids GCC 9's ambiguous unique-to-shared pointer
conversion while retaining the factory's custom GPU deleter. It applies only
to the stated source pin; preserve the original source notices. NVTX motivation:
https://github.com/NVIDIA/MinkowskiEngine/issues/614. This is a local build
compatibility patch, not a claimed upstream release.

After installing, run `python -m pip check`, import every native module, and
run the adapter host tests on CUDA. Installation alone does not validate the
model. Do not put the LDMR repository root on host `PYTHONPATH`; use only its
`adapters` directory and the pinned UniDet3D root. The legacy LDMR interpreter
is still required by `export_memory.py`; `run_sequence.py` keeps these runtimes
separate automatically.


## Moving the code to another researcher workspace

Set `UNIDET3D_ROOT` to the pinned upstream checkout when running either test
suite; it need not be a sibling of the maintained repository. The legacy suite
uses the maintained root on `PYTHONPATH`. The host suite uses only `adapters`
and the upstream checkout, as shown in the [handoff](../../docs/researcher_handoff.md).
Keep both Python environments separate after moving the sources.

`host-tested-freeze.txt` records the tested installation; it is not a portable
installation lockfile. Its MinkowskiEngine and MMCV entries contain local
`file://` wheel locations from this server. Rebuild or obtain the matching
artifacts using the versions, source pins and patch above, then record the new
wheel hashes. Do not expect those original absolute wheel paths to exist on a
teammate's machine. A relocated-checkout test using existing environments does
not validate a fresh-machine build or installation.

### Running the host regression suite

Run host tests from outside the legacy repository so its vendored `mmdet3d`
directory does not shadow the modern installed framework. Use absolute paths:

```bash
cd /tmp
UNIDET3D_ROOT=/absolute/path/to/UniDet3D \
PYTHONPATH=/absolute/path/to/repo/adapters:/absolute/path/to/UniDet3D \
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
/path/to/host/bin/python -m pytest -q /absolute/path/to/repo/adapters/tests
```

A collection error importing legacy `mmdet3d/__init__.py` indicates path
contamination; keep the host and legacy dependency environments separate.
