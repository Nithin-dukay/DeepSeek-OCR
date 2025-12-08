#!/bin/bash

# DeepSeek-OCR Installation Script
# This script ensures proper installation order to avoid dependency issues

set -e  # Exit on error

echo "=========================================="
echo "DeepSeek-OCR Installation Script"
echo "=========================================="
echo ""

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: No conda environment detected."
    echo "Please create and activate a conda environment first:"
    echo "  conda create -n deepseek-ocr python=3.12.9 -y"
    echo "  conda activate deepseek-ocr"
    exit 1
fi

echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Check if vLLM wheel file exists
VLLM_WHL="vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl"
if [ ! -f "$VLLM_WHL" ]; then
    echo "Error: vLLM wheel file not found: $VLLM_WHL"
    echo "Please download it from: https://github.com/vllm-project/vllm/releases/tag/v0.8.5"
    exit 1
fi

echo "Found vLLM wheel: $VLLM_WHL"
echo ""

# Step 1: Install PyTorch (CRITICAL - must be installed before vLLM/xformers)
echo "Step 1/4: Installing PyTorch 2.6.0 with CUDA 11.8..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Verify torch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

if [ $? -ne 0 ]; then
    echo "Error: PyTorch installation failed!"
    exit 1
fi

echo ""
echo "✓ PyTorch installed successfully"
echo ""

# Step 2: Install vLLM (this will build xformers)
echo "Step 2/4: Installing vLLM (this may take a while as xformers needs to be built)..."
pip install "$VLLM_WHL"

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: vLLM installation failed!"
    echo "Trying alternative method: installing xformers separately first..."
    echo ""
    pip install xformers==0.0.29.post2
    pip install "$VLLM_WHL"
    
    if [ $? -ne 0 ]; then
        echo "Error: Alternative installation method also failed!"
        exit 1
    fi
fi

echo ""
echo "✓ vLLM installed successfully"
echo ""

# Step 3: Install other requirements
echo "Step 3/4: Installing other requirements..."
pip install -r requirements.txt

echo ""
echo "✓ Requirements installed successfully"
echo ""

# Step 4: Install flash-attention
echo "Step 4/4: Installing flash-attention (this may take a while)..."
pip install flash-attn==2.7.3 --no-build-isolation

echo ""
echo "✓ Flash-attention installed successfully"
echo ""

# Final verification
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
python -c "import torch; import vllm; import transformers; print('All core packages imported successfully!')"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Installation verified successfully!"
    echo ""
    echo "You can now use DeepSeek-OCR. Try running:"
    echo "  cd DeepSeek-OCR-master/DeepSeek-OCR-vllm"
    echo "  python run_dpsk_ocr_image.py"
else
    echo ""
    echo "Warning: Some packages may not have been installed correctly."
    echo "Please check the error messages above."
fi
