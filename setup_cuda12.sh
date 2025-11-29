#!/bin/bash

# Setup script for DeepSeek-OCR vLLM on CUDA 12.8 (RTX 5090)
# Based on GitHub Issue #240

set -e  # Exit on any error

echo "Starting setup for DeepSeek-OCR vLLM on CUDA 12.8..."

# Step 1: Install Core vLLM, Flash Attention & xformers
echo "Step 1: Installing xformers nightly..."
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128

echo "Step 2: Downloading pre-built wheels..."
wget -O flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
wget -O vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl

echo "Step 3: Installing wheels..."
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps

# Step 2: Install Initial Dependencies
echo "Step 4: Installing initial dependencies..."
pip install pydantic
pip install transformers
pip install cachetools
pip install cloudpickle
pip install psutil
pip install zmq
pip install msgspec
pip install blake3

# Step 3: The Critical torchvision Fix
echo "Step 5: Installing torchvision nightly..."
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

echo "Step 6: Re-installing xformers to fix conflicts..."
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# Step 4: Final Dependencies
echo "Step 7: Installing final dependencies..."
pip install hf_transfer
pip install prometheus_client

# Verification loop
echo "Step 8: Verifying installation..."
while true; do
    if python -c "import vllm; print('vLLM version:', vllm.__version__)" 2>/dev/null; then
        echo "vLLM import successful!"
        break
    else
        echo "vLLM import failed. Installing missing dependencies..."
        # Common missing packages; add more if needed
        pip install pydantic transformers cachetools cloudpickle psutil zmq msgspec blake3 hf_transfer prometheus_client || true
    fi
done

echo "Setup complete! You can now use DeepSeek-OCR with vLLM on CUDA 12.8."