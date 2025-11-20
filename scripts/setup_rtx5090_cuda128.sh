#!/bin/bash

# DeepSeek-OCR Installation Script for RTX 5090 with CUDA 12.8
# Based on GitHub Issue #240
# This script automates the installation process with proper error handling

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_info "Checking prerequisites..."

if ! command_exists python; then
    print_error "Python is not installed. Please install Python 3.12 first."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

if ! command_exists pip; then
    print_error "pip is not installed. Please install pip first."
    exit 1
fi

print_success "Prerequisites check passed"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip
print_success "pip upgraded"

# Step 1: Install xformers (nightly build)
print_info "Step 1/7: Installing xformers nightly build..."
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
print_success "xformers installed"

# Step 2: Download pre-built wheels
print_info "Step 2/7: Downloading pre-built wheels..."

FLASH_ATTN_WHEEL="flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
VLLM_WHEEL="vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl"
WHEEL_URL_BASE="https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1"

if [ ! -f "$FLASH_ATTN_WHEEL" ]; then
    print_info "Downloading flash-attn wheel..."
    wget -q --show-progress "$WHEEL_URL_BASE/$FLASH_ATTN_WHEEL"
    print_success "flash-attn wheel downloaded"
else
    print_warning "flash-attn wheel already exists, skipping download"
fi

if [ ! -f "$VLLM_WHEEL" ]; then
    print_info "Downloading vLLM wheel..."
    wget -q --show-progress "$WHEEL_URL_BASE/$VLLM_WHEEL"
    print_success "vLLM wheel downloaded"
else
    print_warning "vLLM wheel already exists, skipping download"
fi

# Step 3: Install the wheels
print_info "Step 3/7: Installing flash-attn and vLLM wheels..."
pip install ./$FLASH_ATTN_WHEEL
print_success "flash-attn installed"

pip install ./$VLLM_WHEEL --no-build-isolation --no-deps
print_success "vLLM wheel installed"

# Step 4: Install initial dependencies
print_info "Step 4/7: Installing initial dependencies..."
INITIAL_DEPS=(
    "pydantic"
    "transformers"
    "cachetools"
    "cloudpickle"
    "psutil"
    "zmq"
    "msgspec"
    "blake3"
)

for dep in "${INITIAL_DEPS[@]}"; do
    print_info "Installing $dep..."
    pip install "$dep"
done
print_success "Initial dependencies installed"

# Step 5: First verification
print_info "Step 5/7: Running first verification..."
if python -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
    print_success "vLLM import successful (first pass)"
else
    print_warning "vLLM import failed (expected at this stage)"
fi

# Step 6: Install torchvision and fix xformers
print_info "Step 6/7: Installing torchvision..."
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
print_success "torchvision installed"

print_warning "Re-installing xformers to fix PyTorch version conflict..."
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
print_success "xformers re-installed (PyTorch conflict fixed)"

# Step 7: Install final dependencies
print_info "Step 7/7: Installing final dependencies..."
FINAL_DEPS=(
    "hf_transfer"
    "prometheus_client"
)

for dep in "${FINAL_DEPS[@]}"; do
    print_info "Installing $dep..."
    pip install "$dep"
done
print_success "Final dependencies installed"

# Final verification
print_info "Running final verification..."
if python -c "import vllm; print('vLLM version:', vllm.__version__)" 2>/dev/null; then
    print_success "✓ vLLM is properly installed!"
else
    print_error "vLLM import failed. Running diagnostic..."
    python -c "import vllm; print(vllm.__version__)" 2>&1 || true
    print_warning "Some dependencies may still be missing. Run: python scripts/verify_installation.py"
fi

# Install DeepSeek-OCR requirements
print_info "Installing DeepSeek-OCR requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "DeepSeek-OCR requirements installed"
else
    print_warning "requirements.txt not found. Make sure you're in the DeepSeek-OCR directory."
fi

# Cleanup
print_info "Cleaning up downloaded wheels..."
if [ -f "$FLASH_ATTN_WHEEL" ]; then
    rm -f "$FLASH_ATTN_WHEEL"
fi
if [ -f "$VLLM_WHEEL" ]; then
    rm -f "$VLLM_WHEEL"
fi
print_success "Cleanup complete"

# Final message
echo ""
print_success "=========================================="
print_success "Installation complete!"
print_success "=========================================="
echo ""
print_info "Next steps:"
echo "  1. Run verification: python scripts/verify_installation.py"
echo "  2. Configure your paths in: DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py"
echo "  3. Run DeepSeek-OCR:"
echo "     cd DeepSeek-OCR-master/DeepSeek-OCR-vllm"
echo "     python run_dpsk_ocr_image.py"
echo ""
print_info "For troubleshooting, see: docs/INSTALL_RTX5090_CUDA128.md"
