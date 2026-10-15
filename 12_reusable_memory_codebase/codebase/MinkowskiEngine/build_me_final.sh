#!/bin/bash
# Final build script for MinkowskiEngine with GCC-9

set -euo pipefail

BUILD_ROOT="${ME_BUILD_ROOT:-$HOME/me_sysbuild/MinkowskiEngine}"
VENV_ROOT="${ME_VENV_ROOT:-$HOME/me_sysbuild/venv}"

cd "$BUILD_ROOT"
source "$VENV_ROOT/bin/activate"

# Set up all environment variables
export H="$HOME"
export CUDA_HOME="$H/local/cuda-11.3"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# OpenBLAS
export OPENBLAS_PREFIX="$H/local/openblas"
export BLAS_PREFIX="$OPENBLAS_PREFIX"
export CPATH="$BLAS_PREFIX/include"
export LIBRARY_PATH="$BLAS_PREFIX/lib"
export LD_LIBRARY_PATH="$BLAS_PREFIX/lib:$LD_LIBRARY_PATH"

# GCC-9
export OPT_GCC9_ROOT="$H/opt/gcc-9"
export USE_GCC9="$OPT_GCC9_ROOT/ROOT/usr/bin"
export LD_LIBRARY_PATH="$OPT_GCC9_ROOT/ROOT/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export CC="$USE_GCC9/gcc-9"
export CXX="$USE_GCC9/g++-9"

# Build settings
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4
export FORCE_CUDA=1

echo "Building MinkowskiEngine with:"
echo "Build root: $BUILD_ROOT"
echo "CUDA: $CUDA_HOME"
echo "BLAS: $BLAS_PREFIX"  
echo "CC: $CC"
echo "CXX: $CXX"

# Clean previous build
rm -rf build dist *.egg-info

# Build wheel
python setup.py bdist_wheel --blas=openblas --force_cuda

# Check if wheel was created
WHEEL=$(ls -1 dist/MinkowskiEngine-*.whl 2>/dev/null | head -n1)
if [ -n "$WHEEL" ]; then
    echo "SUCCESS: Wheel created at $WHEEL"
    
    # Install the wheel
    pip install "$WHEEL" --force-reinstall
    
    # Test CUDA functionality
    python - << 'EOF'
import torch
import MinkowskiEngine as ME
print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("MinkowskiEngine:", ME.__version__)

# CUDA smoke test
if torch.cuda.is_available():
    x = ME.SparseTensor(
        features=torch.randn(10, 4, device='cuda'),
        coordinates=ME.utils.batched_coordinates([torch.randint(0,16,(10,4))]).to('cuda'),
    )
    layer = ME.MinkowskiConvolution(4, 8, kernel_size=3, dimension=3).to('cuda')
    y = layer(x)
    print("CUDA test PASSED. Output shape:", y.F.shape)
else:
    print("CUDA not available for testing")
EOF
    
else
    echo "ERROR: No wheel file created"
    exit 1
fi
