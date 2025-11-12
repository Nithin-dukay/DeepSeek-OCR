#!/bin/bash

################################################################################
# DeepSeek-OCR Installation Script for CUDA 12.8 / RTX 5090
# 
# This script automates the installation process documented in GitHub Issue #240
# for setting up DeepSeek-OCR with vLLM on NVIDIA RTX 5090 with CUDA 12.8
#
# Usage: bash scripts/install_cuda128_rtx5090.sh
#
# Prerequisites:
# - CUDA 12.8 installed
# - Python 3.12 environment activated
# - Internet connection for downloading packages
################################################################################

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

log_step() {
    echo -e "\n${GREEN}===================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}===================================================${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    log_step "Step 0: Checking Prerequisites"
    
    # Check Python version
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"
    
    if [[ ! $PYTHON_VERSION == 3.12* ]]; then
        log_warning "Python 3.12 is recommended. Current version: $PYTHON_VERSION"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Installation cancelled."
            exit 1
        fi
    fi
    
    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        log_info "CUDA version: $CUDA_VERSION"
    else
        log_warning "nvcc not found. Make sure CUDA 12.8 is installed."
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        log_error "pip is not installed. Please install pip first."
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Step 1: Install xformers and download wheels
install_core_packages() {
    log_step "Step 1: Installing Core Packages (xformers, flash-attn, vllm)"
    
    log_info "Installing xformers nightly build..."
    pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
    log_success "xformers installed"
    
    log_info "Downloading pre-built wheels..."
    
    # Create temp directory for wheels
    WHEEL_DIR=$(mktemp -d)
    cd "$WHEEL_DIR"
    
    log_info "Downloading flash-attn wheel..."
    wget -q --show-progress https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
    
    log_info "Downloading vllm wheel..."
    wget -q --show-progress https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
    
    log_success "Wheels downloaded to $WHEEL_DIR"
    
    log_info "Installing flash-attn..."
    pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
    log_success "flash-attn installed"
    
    log_info "Installing vllm (this may take a moment)..."
    pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
    log_success "vllm installed"
    
    # Return to original directory
    cd - > /dev/null
    
    log_success "Core packages installation completed"
}

# Step 2: Install initial dependencies
install_initial_dependencies() {
    log_step "Step 2: Installing Initial Dependencies"
    
    INITIAL_DEPS=(
        "pydantic"
        "transformers"
        "cachetools"
        "cloudpickle"
        "psutil"
        "zmq"
        "msgspec"
        "blake3"
        "numpy"
        "pillow"
        "requests"
        "tqdm"
        "packaging"
        "filelock"
        "huggingface-hub"
    )
    
    for dep in "${INITIAL_DEPS[@]}"; do
        log_info "Installing $dep..."
        pip install "$dep" -q
    done
    
    log_success "Initial dependencies installed"
}

# Step 3: Iterative dependency resolution
resolve_dependencies() {
    log_step "Step 3: Resolving Missing Dependencies"
    
    log_info "Checking for missing dependencies..."
    
    MAX_ITERATIONS=10
    ITERATION=0
    
    while [ $ITERATION -lt $MAX_ITERATIONS ]; do
        ITERATION=$((ITERATION + 1))
        log_info "Iteration $ITERATION/$MAX_ITERATIONS"
        
        # Try to import vllm and capture any missing module errors
        MISSING_MODULE=$(python -c "import vllm; print(vllm.__version__)" 2>&1 | grep "No module named" | sed "s/.*No module named '\(.*\)'.*/\1/" | head -1)
        
        if [ -z "$MISSING_MODULE" ]; then
            log_success "All dependencies resolved!"
            break
        fi
        
        log_warning "Missing module: $MISSING_MODULE"
        log_info "Installing $MISSING_MODULE..."
        
        # Try to install the missing module
        if pip install "$MISSING_MODULE" -q; then
            log_success "$MISSING_MODULE installed"
        else
            log_warning "Could not install $MISSING_MODULE automatically"
        fi
    done
    
    if [ $ITERATION -eq $MAX_ITERATIONS ]; then
        log_warning "Reached maximum iterations. Some dependencies may still be missing."
    fi
}

# Step 4: Install torchvision and fix xformers
install_torchvision_and_fix() {
    log_step "Step 4: Installing torchvision and Fixing xformers (Critical Step)"
    
    log_warning "Installing torchvision will temporarily break the environment"
    log_info "Installing torchvision nightly..."
    pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    log_success "torchvision installed"
    
    log_warning "Re-installing xformers to fix the environment..."
    log_info "This will restore the correct PyTorch nightly version..."
    pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
    log_success "xformers re-installed and environment fixed"
}

# Step 5: Install final dependencies
install_final_dependencies() {
    log_step "Step 5: Installing Final Dependencies"
    
    FINAL_DEPS=(
        "hf_transfer"
        "prometheus_client"
        "sentencepiece"
        "protobuf"
    )
    
    for dep in "${FINAL_DEPS[@]}"; do
        log_info "Installing $dep..."
        pip install "$dep" -q
    done
    
    log_success "Final dependencies installed"
}

# Step 6: Install DeepSeek-OCR requirements
install_deepseek_requirements() {
    log_step "Step 6: Installing DeepSeek-OCR Requirements"
    
    if [ -f "requirements.txt" ]; then
        log_info "Installing requirements from requirements.txt..."
        pip install -r requirements.txt -q
        log_success "DeepSeek-OCR requirements installed"
    else
        log_warning "requirements.txt not found. Skipping this step."
        log_info "Make sure to run this script from the DeepSeek-OCR root directory"
    fi
}

# Verification
verify_installation() {
    log_step "Step 7: Verifying Installation"
    
    log_info "Checking vLLM..."
    if python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>&1; then
        log_success "vLLM is working"
    else
        log_error "vLLM verification failed"
        return 1
    fi
    
    log_info "Checking PyTorch..."
    if python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>&1; then
        log_success "PyTorch is working"
    else
        log_error "PyTorch verification failed"
        return 1
    fi
    
    log_info "Checking torchvision..."
    if python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')" 2>&1; then
        log_success "Torchvision is working"
    else
        log_error "Torchvision verification failed"
        return 1
    fi
    
    log_info "Checking xformers..."
    if python -c "import xformers; print(f'xformers version: {xformers.__version__}')" 2>&1; then
        log_success "xformers is working"
    else
        log_error "xformers verification failed"
        return 1
    fi
    
    log_info "Checking flash_attn..."
    if python -c "import flash_attn; print('flash_attn is installed')" 2>&1; then
        log_success "flash_attn is working"
    else
        log_warning "flash_attn verification failed (this may be okay)"
    fi
    
    log_success "Installation verification completed successfully!"
}

# Main installation flow
main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║     DeepSeek-OCR Installation Script for CUDA 12.8 / RTX 5090 ║"
    echo "║                                                                ║"
    echo "║     Based on GitHub Issue #240                                 ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
    
    log_warning "This script will install multiple packages and may take 10-20 minutes."
    log_warning "Make sure you have activated the correct Python 3.12 environment."
    echo
    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled."
        exit 0
    fi
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run installation steps
    check_prerequisites
    install_core_packages
    install_initial_dependencies
    resolve_dependencies
    install_torchvision_and_fix
    install_final_dependencies
    install_deepseek_requirements
    
    # Verify installation
    if verify_installation; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo
        log_step "Installation Complete!"
        log_success "Total time: $((DURATION / 60)) minutes $((DURATION % 60)) seconds"
        echo
        log_info "Next steps:"
        echo "  1. Configure your paths in DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py"
        echo "  2. Run: cd DeepSeek-OCR-master/DeepSeek-OCR-vllm"
        echo "  3. Test: python run_dpsk_ocr_image.py"
        echo
        log_info "For more information, see docs/INSTALL_CUDA_12.8.md"
    else
        log_error "Installation verification failed. Please check the errors above."
        log_info "You can try running the verification script manually:"
        log_info "  python scripts/verify_installation.py"
        exit 1
    fi
}

# Run main function
main "$@"
