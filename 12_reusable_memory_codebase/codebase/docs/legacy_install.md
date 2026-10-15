# Legacy LDMR installation

Target: Linux x86_64, Python 3.9, PyTorch 1.12.1+cu113, MMCV 1.6.0,
MMDetection 2.24.1 and MinkowskiEngine 0.5.4. Keep this environment separate
from the [UniDet3D environment](../adapters/environment/README.md).

The October 14 fresh isolated environment passed dependency checks, sparse CUDA
forward/backward, MMCV CUDA NMS, OpenCV resize, config serialization, and all
389 legacy tests (16 dependency warnings, no skips). The original working
environments remain preserved. This validates installation on the existing
Ubuntu 20.04 / RTX 3090 server, not a fresh-machine installation.
Detailed installation logs and wheel identities are retained locally.

## Dependencies

Use a new virtual environment on an executable filesystem. A virtual environment
on a `noexec` data mount cannot load native shared libraries. Start in the LDMR
checkout and substitute your environment path:

```bash
python3.9 -m venv /path/on/executable/filesystem/ldmr
source /path/on/executable/filesystem/ldmr/bin/activate
python -m pip install pip==25.2 setuptools==59.5.0 wheel==0.45.1
python -m pip install -r requirements/legacy-cu113.txt -c requirements/legacy-cu113-tested.txt
```

The specification preserves the legacy detector framework versions. NumPy
1.23.5, Numba 0.56.4 and NetworkX 2.8.8 match the previously tested environment.
Numba's [version compatibility table](https://numba.readthedocs.io/en/0.56.4/user/installing.html)
supports this Python/NumPy pairing. OpenCV 4.8.1.78 avoids OpenCV 5's NumPy 2
requirement. YAPF 0.40.1 retains the API used by MMCV's config serializer;
protobuf 3.20.3 retains the API used by TensorBoard 2.11.2.

NuScenes and Lyft remain required because the dataset registry imports their
modules eagerly. Their transitive dependencies, including Jupyter, must be
installed by the resolver. Do not hide the NuScenes requirement with `--no-deps`
or silently make those registered datasets unavailable. Optionalizing these
datasets would be a separate code change.

## Native sparse engine

Build MinkowskiEngine from pinned source on a compatible machine:

```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git /path/to/native-build
cd /path/to/native-build
git checkout 405b39cb7e68c2ec1b4ac8c4b4ca9fa6fcfcb2fc
export CUDA_HOME=/usr/local/cuda-11.3
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
python setup.py bdist_wheel --blas=openblas --force_cuda
python -m pip install dist/MinkowskiEngine-0.5.4-*.whl
```

This build needs GCC 9, OpenBLAS development files, CUDA 11.3 development files,
and Python 3.9 development headers. Set the GPU architecture for your hardware;
8.6 is the tested RTX 3090 target. If Python headers are extracted outside the
system include tree, add both `include/python3.9` and its parent `include` to
`CPATH`. Keep source/build directories on an executable filesystem.

Return to the LDMR checkout and install its metadata with dependency resolution:

```bash
python -m pip install --no-build-isolation -e .
python -m pip check
OMP_NUM_THREADS=2 python tools/check_legacy_install.py --output /path/to/new-install-check.json
OMP_NUM_THREADS=2 UNIDET3D_ROOT=/path/to/pinned/UniDet3D python -m pytest tests -q
```

See the [handoff](researcher_handoff.md) for the upstream UniDet3D source pin used
by host-config contract tests. These legacy tests do not require the host runtime
to be installed in this environment. Native CUDA forward/backward and MMCV NMS
must also pass before accepting an installation. Package import alone is weaker
evidence than those checks.

The local trial records commands, logs, immutable source hashes, downloaded
artifacts and wheel hashes under `week5_audit/install_20261014/`. A resolver
specification is not a transitive lockfile; preserve the final resolved package
inventory and artifact hashes when reproducing the trial elsewhere.

## Reproducing the resolved environment

[legacy-cu113-tested.txt](../requirements/legacy-cu113-tested.txt) records the
resolved transitive package versions as portable constraints, excluding the
editable repository and separately built native engine. Wheel SHA256 values are
in the acceptance evidence. Rebuild native extensions for a different platform;
the local MinkowskiEngine wheel targets Python 3.9 and RTX 3090 architecture 8.6.
Keep the activated environment first on `PATH`: the pinned native setup invokes
`pip` by name. No system or existing working environment should receive installs.

The trial completed offline after downloading its artifacts. For offline reuse,
remove remote `--extra-index-url` and `--find-links` directives from a copied,
flattened requirements file, then use `--no-index --find-links /path/to/wheels`.
`--no-index` alone does not disable remote `--find-links` downloads. The local
wheel directory and original logs are retained alongside the source snapshot.

The final audit verified 35,162 RECORD-hashed installed files across 179
distributions with no failures (16,393 unhashed entries are excluded). An offline
resolver dry run ignoring installed packages reproduces every resolved dependency
version under the tested constraints. Include both the original wheel directory
and `bootstrap_wheels` as local `--find-links` locations: the bootstrap pins
were initially absent from the original wheel bundle. Their separate hashes are
in the acceptance report.
