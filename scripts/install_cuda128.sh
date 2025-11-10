#!/bin/bash

# DeepSeek-OCR Installation Script for CUDA 12.8 / RTX 5090
# Based on GitHub Issue #240
# This script automates the installation process for DeepSeek-OCR with vLLM

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
echo "=============================================="
echo "  DeepSeek-OCR Installation for CUDA 12.8"
echo "  RTX 5090 / CUDA 12.8 Compatible"
echo "=============================================="
echo ""

# Check Python version
log_info "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -ne 12 ]; then
    log_warning "Python 3.12.x is recommended. Current version: $PYTHON_VERSION"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_error "Installation cancelled."
        exit 1
    fi
else
    log_success "Python version $PYTHON_VERSION detected"
fi

# Check CUDA
log_info "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    log_success "CUDA version $CUDA_VERSION detected"
else
    log_warning "nvcc not found. Make sure CUDA 12.8 is installed."
fi

if command -v nvidia-smi &> /dev/null; then
    log_info "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    log_warning "nvidia-smi not found. Cannot verify GPU."
fi

echo ""
log_info "Starting installation process..."
echo ""

# Step 1: Install xformers (nightly)
log_info "Step 1/7: Installing xformers nightly build..."
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
log_success "xformers installed successfully"
echo ""

# Step 2: Download pre-built wheels
log_info "Step 2/7: Downloading pre-built wheels..."

FLASH_ATTN_WHEEL="flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
VLLM_WHEEL="vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl"

if [ ! -f "$FLASH_ATTN_WHEEL" ]; then
    log_info "Downloading flash-attn wheel..."
    wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
    log_success "flash-attn wheel downloaded"
else
    log_info "flash-attn wheel already exists, skipping download"
fi

if [ ! -f "$VLLM_WHEEL" ]; then
    log_info "Downloading vLLM wheel..."
    wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
    log_success "vLLM wheel downloaded"
else
    log_info "vLLM wheel already exists, skipping download"
fi
echo ""

# Step 3: Install flash-attn and vLLM
log_info "Step 3/7: Installing flash-attn and vLLM..."
pip install ./$FLASH_ATTN_WHEEL
log_success "flash-attn installed"

pip install ./$VLLM_WHEEL --no-build-isolation --no-deps
log_success "vLLM installed with special flags"
echo ""

# Step 4: Install core dependencies
log_info "Step 4/7: Installing core dependencies..."
CORE_DEPS=(
    "pydantic"
    "transformers"
    "cachetools"
    "cloudpickle"
    "psutil"
    "zmq"
    "msgspec"
    "blake3"
)

for dep in "${CORE_DEPS[@]}"; do
    log_info "Installing $dep..."
    pip install "$dep" -q
done
log_success "Core dependencies installed"
echo ""

# Step 5: Iterative dependency installation
log_info "Step 5/7: Installing additional vLLM dependencies..."
log_info "This may take several iterations to find all missing packages..."

ADDITIONAL_DEPS=(
    "hf_transfer"
    "prometheus_client"
    "ray"
    "fastapi"
    "uvicorn"
    "numpy"
    "pillow"
    "sentencepiece"
    "protobuf"
    "grpcio"
    "aiohttp"
    "openai"
    "tiktoken"
    "lm-format-enforcer"
    "outlines"
    "typing-extensions"
    "filelock"
    "requests"
)

for dep in "${ADDITIONAL_DEPS[@]}"; do
    log_info "Installing $dep..."
    pip install "$dep" -q || log_warning "Failed to install $dep, continuing..."
done
log_success "Additional dependencies installed"
echo ""

# Step 6: Install torchvision and fix
log_info "Step 6/7: Installing torchvision (CRITICAL STEP)..."
log_warning "This will temporarily break the environment..."
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
log_success "torchvision installed"

log_info "Re-installing xformers to fix the environment (CRITICAL FIX)..."
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
log_success "xformers re-installed, environment fixed"
echo ""

# Step 7: Install DeepSeek-OCR requirements
log_info "Step 7/7: Installing DeepSeek-OCR requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    log_success "DeepSeek-OCR requirements installed"
else
    log_warning "requirements.txt not found, skipping"
fi
echo ""

# Final verification
log_info "Running final verification..."
echo ""

log_info "Checking vLLM import..."
if python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null; then
    log_success "vLLM import successful"
else
    log_error "vLLM import failed"
    log_info "Running iterative dependency check..."
    
    # Try to import and show the error
    python -c "import vllm" 2>&1 | grep "ModuleNotFoundError" | while read -r line; do
        MODULE=$(echo "$line" | grep -oP "No module named '\K[^']+")
        if [ ! -z "$MODULE" ]; then
            log_warning "Missing module: $MODULE"
            log_info "Installing $MODULE..."
            pip install "$MODULE" -q
        fi
    done
    
    # Try again
    if python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null; then
        log_success "vLLM import successful after fixing dependencies"
    else
        log_error "vLLM import still failing. Please check manually."
    fi
fi

log_info "Checking torch CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

log_info "Checking flash-attn..."
if python -c "import flash_attn" 2>/dev/null; then
    log_success "flash-attn import successful"
else
    log_warning "flash-attn import failed"
fi

log_info "Checking xformers..."
if python -c "import xformers" 2>/dev/null; then
    log_success "xformers import successful"
else
    log_warning "xformers import failed"
fi

echo ""
echo "=============================================="
log_success "Installation completed!"
echo "=============================================="
echo ""
log_info "Next steps:"
echo "  1. Run verification script: python scripts/verify_installation.py"
echo "  2. Configure your paths in DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py"
echo "  3. Test with: cd DeepSeek-OCR-master/DeepSeek-OCR-vllm && python run_dpsk_ocr_image.py"
echo ""
log_info "If you encounter any issues, check docs/TROUBLESHOOTING.md"
echo ""
