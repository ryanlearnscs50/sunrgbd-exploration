# MinkowskiEngine v0.5.4 Installation Guide

This guide documents the successful installation of MinkowskiEngine v0.5.4 on Ubuntu 22.04 without docker or sudo access, using local CUDA 11.3 and avoiding conda contamination.

## Prerequisites

- Ubuntu 22.04 LTS
- Python 3.9
- No sudo access required
- CUDA 11.3 toolkit installed at `$HOME/local/cuda-11.3`
- OpenBLAS installed at `$HOME/local/openblas`

## Installation Overview

The installation process involves:
1. Setting up a clean Python virtual environment (no conda)
2. Installing local GCC-9 compiler (compatible with CUDA 11.3)
3. Building MinkowskiEngine from source with proper environment isolation
4. Installing and testing the built wheel

## Step 1: Environment Setup

Create a clean working directory and Python virtual environment:

```bash
# Set up directories
export H="$HOME"
export WORK="$H/me_sysbuild"
export VENV="$WORK/venv"
mkdir -p "$WORK"

# Create pure Python venv (not conda-based)
python3 -m venv "$VENV"
source "$VENV/bin/activate"

# Upgrade build tools
python -m pip install -U pip setuptools wheel cmake ninja
```

## Step 2: CUDA 11.3 Configuration

```bash
export CUDA_HOME="$H/local/cuda-11.3"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Verify CUDA installation
nvcc --version
```

Expected output:
```
Cuda compilation tools, release 11.3, V11.3.58
```

## Step 3: Install PyTorch 1.12.1+cu113

```bash
pip install "torch==1.12.1+cu113" "torchvision==0.13.1+cu113" \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

## Step 4: OpenBLAS Configuration

```bash
export OPENBLAS_PREFIX="$H/local/openblas"
export BLAS_PREFIX="$OPENBLAS_PREFIX"
export CPATH="$BLAS_PREFIX/include"
export LIBRARY_PATH="$BLAS_PREFIX/lib"
export LD_LIBRARY_PATH="$BLAS_PREFIX/lib:$LD_LIBRARY_PATH"

# Verify OpenBLAS
ls -la "$BLAS_PREFIX/include/cblas.h"
```

## Step 5: Install Local GCC-9 (No Sudo Required)

CUDA 11.3 requires GCC ≤10.0. We use GCC-9 for maximum compatibility:

```bash
export OPT_GCC9_ROOT="$H/opt/gcc-9"
mkdir -p "$OPT_GCC9_ROOT/DEBS" "$OPT_GCC9_ROOT/ROOT"
cd "$OPT_GCC9_ROOT/DEBS"

# Download GCC-9 packages using apt (no sudo needed)
apt download gcc-9 g++-9 libstdc++-9-dev libgcc-9-dev

# Extract packages
for deb in *.deb; do
    dpkg-deb -x "$deb" "$OPT_GCC9_ROOT/ROOT"
done

# Configure GCC-9 environment
export USE_GCC9="$OPT_GCC9_ROOT/ROOT/usr/bin"
export LD_LIBRARY_PATH="$OPT_GCC9_ROOT/ROOT/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export CC="$USE_GCC9/gcc-9"
export CXX="$USE_GCC9/g++-9"

# Verify GCC-9 installation
$CC --version
```

Expected output:
```
gcc-9 (Ubuntu 9.5.0-1ubuntu1~22.04) 9.5.0
```

## Step 6: Clone and Patch MinkowskiEngine

```bash
cd "$WORK"
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
git checkout v0.5.4
```

### Critical Fix: Patch spmm.cu Header Order

The original code has header inclusion issues causing `at::Tensor` incomplete type errors. Apply this fix:

```bash
# Edit src/spmm.cu to reorder includes (lines 32-36)
# Change FROM:
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include <torch/script.h>

# TO:
#include <torch/extension.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
```

## Step 7: Build Configuration

```bash
# Build settings
export TORCH_CUDA_ARCH_LIST="8.6"  # Adjust for your GPU architecture
export MAX_JOBS=4
export FORCE_CUDA=1

# Clean any previous builds
rm -rf build dist *.egg-info
python setup.py clean
```

## Step 8: Build MinkowskiEngine Wheel

```bash
# Complete environment setup script
source "$VENV/bin/activate"
export H="$HOME"
export CUDA_HOME="$H/local/cuda-11.3"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OPENBLAS_PREFIX="$H/local/openblas"
export BLAS_PREFIX="$OPENBLAS_PREFIX"
export CPATH="$BLAS_PREFIX/include"
export LIBRARY_PATH="$BLAS_PREFIX/lib"
export LD_LIBRARY_PATH="$BLAS_PREFIX/lib:$LD_LIBRARY_PATH"
export OPT_GCC9_ROOT="$H/opt/gcc-9"
export USE_GCC9="$OPT_GCC9_ROOT/ROOT/usr/bin"
export LD_LIBRARY_PATH="$OPT_GCC9_ROOT/ROOT/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export CC="$USE_GCC9/gcc-9"
export CXX="$USE_GCC9/g++-9"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4
export FORCE_CUDA=1

# Build the wheel
python setup.py bdist_wheel --blas=openblas --force_cuda
```

## Step 9: Install and Test

```bash
# Install the built wheel
WHEEL=$(ls -1 dist/MinkowskiEngine-*.whl | head -n1)
pip install "$WHEEL" --force-reinstall

# Fix NumPy compatibility
pip install "numpy<2.0"

# Set OpenMP threads
export OMP_NUM_THREADS=4
```

## Step 10: CUDA Smoke Test

```bash
python - << 'EOF'
import torch
import MinkowskiEngine as ME
print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("MinkowskiEngine:", ME.__version__)

# CUDA smoke test
if torch.cuda.is_available():
    # Create 3D coordinates as int32
    coords = torch.randint(0, 16, (10, 4), dtype=torch.int32)  # [batch_idx, x, y, z]
    features = torch.randn(10, 4, dtype=torch.float32)
    
    # Move to CUDA
    coords = coords.to('cuda')
    features = features.to('cuda')
    
    x = ME.SparseTensor(
        features=features,
        coordinates=coords,
    )
    layer = ME.MinkowskiConvolution(4, 8, kernel_size=3, dimension=3).to('cuda')
    y = layer(x)
    print("CUDA test PASSED. Output shape:", y.F.shape)
    print("SUCCESS: MinkowskiEngine v0.5.4 working with CUDA!")
else:
    print("CUDA not available for testing")
EOF
```

Expected output:
```
PyTorch: 1.12.1+cu113 CUDA available: True
MinkowskiEngine: 0.5.4
CUDA test PASSED. Output shape: torch.Size([10, 8])
SUCCESS: MinkowskiEngine v0.5.4 working with CUDA!
```

## Troubleshooting

### Common Issues and Solutions

1. **`__int128` errors from conda sysroot**
   - Solution: Use pure Python venv, completely avoid conda during compilation

2. **GCC version too new for CUDA 11.3**
   - Error: `version of g++ (10.5.0) is greater than maximum required by CUDA 11.3 (10.0.0)`
   - Solution: Use GCC-9 as documented above

3. **`at::Tensor` incomplete type errors in spmm.cu**
   - Solution: Apply the header reordering patch in Step 6

4. **NumPy 2.x compatibility issues**
   - Error: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2`
   - Solution: `pip install "numpy<2.0"`

5. **`"floor_cuda" not implemented for 'Long'`**
   - Solution: Use `torch.int32` for coordinates instead of default `torch.int64`

## Environment Variables Quick Reference

```bash
# Essential environment variables for MinkowskiEngine
export H="$HOME"
export CUDA_HOME="$H/local/cuda-11.3"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export BLAS_PREFIX="$H/local/openblas"
export CPATH="$BLAS_PREFIX/include"
export LIBRARY_PATH="$BLAS_PREFIX/lib"
export LD_LIBRARY_PATH="$BLAS_PREFIX/lib:$LD_LIBRARY_PATH"
export OPT_GCC9_ROOT="$H/opt/gcc-9"
export USE_GCC9="$OPT_GCC9_ROOT/ROOT/usr/bin"
export LD_LIBRARY_PATH="$OPT_GCC9_ROOT/ROOT/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export CC="$USE_GCC9/gcc-9"
export CXX="$USE_GCC9/g++-9"
export TORCH_CUDA_ARCH_LIST="8.6"
export OMP_NUM_THREADS=4
```

## Final File Locations

- **Built wheel**: `$HOME/me_sysbuild/MinkowskiEngine/dist/MinkowskiEngine-0.5.4-cp39-cp39-linux_x86_64.whl`
- **Virtual environment**: `$HOME/me_sysbuild/venv/`
- **GCC-9 installation**: `$HOME/opt/gcc-9/ROOT/usr/bin/`
- **Source code**: `$HOME/me_sysbuild/MinkowskiEngine/`

## Success Criteria

✅ MinkowskiEngine v0.5.4 compiled successfully  
✅ CUDA 11.3 compatibility verified  
✅ Sparse tensor operations working on GPU  
✅ Forward pass through sparse convolution layer successful  
✅ No conda contamination in build environment

---

**Installation completed**: August 26, 2025  
**Tested on**: Ubuntu 22.04, Python 3.9, CUDA 11.3, RTX 30xx GPU  
**Build time**: ~10 minutes on 4-core system

## Related Documentation

For the rest of the environment setup, see the Installation section of the
top-level [README](../README.md).