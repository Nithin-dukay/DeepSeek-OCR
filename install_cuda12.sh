#!/bin/bash

# DeepSeek-OCR CUDA 12.8 Installation Script for RTX 5090
# Based on GitHub Issue #240: https://github.com/deepseek-ai/DeepSeek-OCR/issues/240

set -e  # Exit on any error

echo "Starting DeepSeek-OCR CUDA 12.8 installation..."

# Step 1: Install Core vLLM, Flash Attention & xformers
echo "Step 1: Installing xformers nightly..."
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128

echo "Step 1: Downloading pre-built wheels..."
wget -q https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
wget -q https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl

echo "Step 1: Installing wheels..."
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps

# Step 2: Install Initial Dependencies
echo "Step 2: Installing initial dependencies..."
pip install pydantic
pip install transformers
pip install cachetools
pip install cloudpickle
pip install psutil
pip install zmq
pip install msgspec
pip install blake3

# Step 3: The Critical torchvision Fix
echo "Step 3: Installing torchvision and fixing xformers..."
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# CRITICAL: Re-install xformers to fix the environment conflict
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# Step 4: Final Dependencies
echo "Step 4: Installing final dependencies..."
pip install hf_transfer
pip install prometheus_client

# Verification loop
echo "Step 5: Verifying installation..."
MAX_ATTEMPTS=5
attempt=1

while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "Verification attempt $attempt/$MAX_ATTEMPTS..."
    if python -c "import vllm; print('vLLM version:', vllm.__version__)" 2>/dev/null; then
        echo "✅ Installation successful!"
        exit 0
    else
        echo "❌ Import failed. Installing common missing packages..."
        # Install some commonly missing packages
        pip install aiohttp || true
        pip install uvicorn || true
        pip install fastapi || true
        pip install ray || true
        pip install outlines || true
        pip install tiktoken || true
        pip install lm-format-enforcer || true
        ((attempt++))
    fi
done

echo "❌ Installation verification failed after $MAX_ATTEMPTS attempts."
echo "Please check the error messages above and install any missing packages manually."
exit 1