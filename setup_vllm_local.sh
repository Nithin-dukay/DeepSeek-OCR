#!/bin/bash
# DeepSeek-OCR Installation Script for vLLM 0.8.5 (Local)
# This script installs DeepSeek-OCR with vLLM 0.8.5 for maximum stability

set -e  # Exit on error

echo "=========================================="
echo "DeepSeek-OCR Installation (vLLM 0.8.5)"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Environment name
ENV_NAME="deepseek-ocr"
PYTHON_VERSION="3.12.9"

echo -e "${GREEN}Step 1: Creating conda environment${NC}"
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo ""

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Warning: Environment '$ENV_NAME' already exists${NC}"
    read -p "Do you want to remove it and create a fresh installation? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Installation cancelled"
        exit 0
    fi
fi

conda create -n $ENV_NAME python=$PYTHON_VERSION -y
echo -e "${GREEN}✓ Environment created${NC}"
echo ""

# Activate environment
echo -e "${GREEN}Step 2: Activating environment${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo -e "${GREEN}✓ Environment activated${NC}"
echo ""

# Install PyTorch with CUDA 11.8
echo -e "${GREEN}Step 3: Installing PyTorch 2.6.0 with CUDA 11.8${NC}"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Download and install vLLM 0.8.5
echo -e "${GREEN}Step 4: Installing vLLM 0.8.5${NC}"
VLLM_WHL="vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl"
VLLM_URL="https://github.com/vllm-project/vllm/releases/download/v0.8.5/$VLLM_WHL"

if [ ! -f "$VLLM_WHL" ]; then
    echo "Downloading vLLM wheel file..."
    wget $VLLM_URL
    echo -e "${GREEN}✓ vLLM wheel downloaded${NC}"
else
    echo "vLLM wheel file already exists, skipping download"
fi

echo "Installing vLLM..."
pip install $VLLM_WHL
echo -e "${GREEN}✓ vLLM installed${NC}"
echo ""

# Install project requirements
echo -e "${GREEN}Step 5: Installing project requirements${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Requirements installed${NC}"
else
    echo -e "${YELLOW}Warning: requirements.txt not found in current directory${NC}"
    echo "Installing core dependencies manually..."
    pip install transformers==4.46.3 tokenizers==0.20.3 PyMuPDF img2pdf einops easydict addict Pillow numpy
    echo -e "${GREEN}✓ Core dependencies installed${NC}"
fi
echo ""

# Install flash-attn (optional)
echo -e "${GREEN}Step 6: Installing flash-attn (optional)${NC}"
echo "This may take several minutes and requires a C++ compiler..."
read -p "Do you want to install flash-attn? (recommended but optional) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if pip install flash-attn==2.7.3 --no-build-isolation; then
        echo -e "${GREEN}✓ flash-attn installed${NC}"
    else
        echo -e "${YELLOW}⚠ flash-attn installation failed (this is optional, continuing...)${NC}"
    fi
else
    echo "Skipping flash-attn installation"
fi
echo ""

# Verify installation
echo -e "${GREEN}Step 7: Verifying installation${NC}"
python -c "
import sys
import torch
import transformers
import vllm

print('Python version:', sys.version.split()[0])
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
print('Transformers version:', transformers.__version__)
print('vLLM version:', vllm.__version__)

# Check for version compatibility
if transformers.__version__ != '4.46.3':
    print('⚠ WARNING: transformers version is not 4.46.3')
    print('  This may cause compatibility issues with vLLM 0.8.5')
else:
    print('✓ All versions are correct')
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Installation completed successfully!"
    echo "==========================================${NC}"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "To test the installation, run:"
    echo "  cd DeepSeek-OCR-master/DeepSeek-OCR-vllm"
    echo "  python run_dpsk_ocr_image.py"
    echo ""
    echo "For more information, see INSTALLATION.md"
else
    echo ""
    echo -e "${RED}Installation verification failed${NC}"
    echo "Please check the error messages above"
    exit 1
fi
