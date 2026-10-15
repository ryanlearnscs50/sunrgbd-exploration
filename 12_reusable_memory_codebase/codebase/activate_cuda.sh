#!/bin/bash
# CUDA 11.3 Environment Setup Script
# Usage: source activate_cuda.sh

echo "Activating CUDA 11.3 environment..."

# Set CUDA paths
export CUDA_HOME=$HOME/local/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set architecture for RTX A5000
export TORCH_CUDA_ARCH_LIST="8.6"
export FORCE_CUDA=1

# Limit compilation jobs to avoid memory issues
export MAX_JOBS=2

echo "CUDA 11.3 environment activated:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  nvcc version: $(nvcc --version | grep 'release')"
echo "  PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"