#!/bin/bash

################################################################################
# DeepSeek-OCR Installation Script for CUDA 12.8 (RTX 5090)
#
# This script automates the installation of DeepSeek-OCR with vLLM on systems
# with NVIDIA RTX 5090 and CUDA 12.8.
#
# Based on: https://github.com/deepseek-ai/DeepSeek-OCR/issues/240
# Credits: @ghcdmm and community contributors
#
# Usage: bash install_cuda128_rtx5090.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.12"
XFORMERS_VERSION="0.0.33.dev20251104+cu128"
XFORMERS_INDEX="https://download.pytorch.org/whl/nightly/cu128"
FLASH_ATTN_WHEEL_URL="https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
VLLM_WHEEL_URL="https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl"
TORCHVISION_INDEX="https://download.pytorch.org/whl/nightly/cu128"

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

print_step() {
    echo -e "\n${GREEN}===================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}===================================================${NC}\n"
}

# Function to check if a Python package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Function to verify vllm installation
verify_vllm() {
    print_info "Verifying vllm installation..."
    if python -c "import vllm; print(vllm.__version__)" 2>/dev/null; then
        print_success "vllm is working correctly!"
        python -c "import vllm; print(f'vllm version: {vllm.__version__}')"
        return 0
    else
        return 1
    fi
}

# Function to find and install missing dependencies
install_missing_deps() {
    print_info "Checking for missing dependencies..."
    
    local max_attempts=20
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))
        print_info "Dependency check attempt $attempt/$max_attempts..."
        
        # Try to import vllm and capture the error
        error_output=$(python -c "import vllm; print(vllm.__version__)" 2>&1)
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            print_success "All dependencies satisfied!"
            echo "$error_output"
            return 0
        fi
        
        # Extract missing module name from error
        if echo "$error_output" | grep -q "ModuleNotFoundError"; then
            missing_module=$(echo "$error_output" | grep "No module named" | sed "s/.*No module named '\([^']*\)'.*/\1/" | head -1)
            
            if [ -n "$missing_module" ]; then
                print_warning "Missing module: $missing_module"
                
                # Map module names to package names
                case "$missing_module" in
                    "zmq")
                        package_name="pyzmq"
                        ;;
                    *)
                        package_name="$missing_module"
                        ;;
                esac
                
                print_info "Installing $package_name..."
                pip install "$package_name" || print_warning "Failed to install $package_name, continuing..."
            else
                print_error "Could not parse missing module name"
                echo "$error_output"
                break
            fi
        else
            print_error "Unexpected error:"
            echo "$error_output"
            break
        fi
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "Reached maximum attempts. Some dependencies may still be missing."
        return 1
    fi
    
    return 0
}

# Main installation process
main() {
    print_step "DeepSeek-OCR Installation for CUDA 12.8 (RTX 5090)"
    
    # Check Python version
    print_info "Checking Python version..."
    python_version=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python version: $python_version"
    
    if [[ ! "$python_version" =~ ^3\.12 ]]; then
        print_warning "This script is designed for Python 3.12. You have $python_version"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Step 1: Install xformers
    print_step "Step 1: Installing xformers nightly"
    print_info "Installing xformers==$XFORMERS_VERSION..."
    pip install "xformers==$XFORMERS_VERSION" --extra-index-url "$XFORMERS_INDEX"
    print_success "xformers installed successfully!"
    
    # Step 2: Download pre-built wheels
    print_step "Step 2: Downloading pre-built wheels"
    
    print_info "Downloading flash-attn wheel..."
    if [ -f "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl" ]; then
        print_warning "flash-attn wheel already exists, skipping download"
    else
        wget "$FLASH_ATTN_WHEEL_URL" || {
            print_error "Failed to download flash-attn wheel"
            exit 1
        }
    fi
    
    print_info "Downloading vllm wheel..."
    if [ -f "vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl" ]; then
        print_warning "vllm wheel already exists, skipping download"
    else
        wget "$VLLM_WHEEL_URL" || {
            print_error "Failed to download vllm wheel"
            exit 1
        }
    fi
    
    print_success "Wheels downloaded successfully!"
    
    # Step 3: Install wheels
    print_step "Step 3: Installing flash-attn and vllm wheels"
    
    print_info "Installing flash-attn..."
    pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
    print_success "flash-attn installed!"
    
    print_info "Installing vllm (with --no-build-isolation --no-deps)..."
    pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
    print_success "vllm wheel installed!"
    
    # Step 4: Install initial dependencies
    print_step "Step 4: Installing initial dependencies"
    
    dependencies=(
        "pydantic"
        "transformers"
        "cachetools"
        "cloudpickle"
        "psutil"
        "pyzmq"
        "msgspec"
        "blake3"
    )
    
    for dep in "${dependencies[@]}"; do
        print_info "Installing $dep..."
        pip install "$dep"
    done
    
    print_success "Initial dependencies installed!"
    
    # Step 5: Iterative dependency resolution
    print_step "Step 5: Resolving remaining dependencies"
    install_missing_deps
    
    # Step 6: Install torchvision (this will break things temporarily)
    print_step "Step 6: Installing torchvision nightly"
    print_warning "This will temporarily break the environment by replacing torch nightly"
    print_info "Installing torchvision..."
    pip install torchvision --index-url "$TORCHVISION_INDEX"
    print_success "torchvision installed!"
    
    # Step 7: CRITICAL - Re-install xformers to fix the environment
    print_step "Step 7: CRITICAL - Re-installing xformers to fix torch nightly"
    print_info "Re-installing xformers==$XFORMERS_VERSION..."
    pip install "xformers==$XFORMERS_VERSION" --extra-index-url "$XFORMERS_INDEX"
    print_success "xformers re-installed! Environment should be fixed now."
    
    # Step 8: Install final dependencies
    print_step "Step 8: Installing final dependencies"
    
    final_deps=(
        "hf_transfer"
        "prometheus_client"
    )
    
    for dep in "${final_deps[@]}"; do
        print_info "Installing $dep..."
        pip install "$dep"
    done
    
    print_success "Final dependencies installed!"
    
    # Step 9: Final dependency check
    print_step "Step 9: Final dependency verification"
    install_missing_deps
    
    # Step 10: Verify installation
    print_step "Step 10: Verifying installation"
    
    print_info "Verifying vllm..."
    if verify_vllm; then
        print_success "vllm verification passed!"
    else
        print_error "vllm verification failed!"
        print_info "Try running: python -c 'import vllm; print(vllm.__version__)'"
        print_info "And install any missing dependencies manually."
    fi
    
    print_info "Verifying torch and torchvision..."
    python -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}'); print(f'TorchVision: {torchvision.__version__}')"
    
    print_info "Verifying flash-attn..."
    python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
    
    print_info "Verifying xformers..."
    python -c "import xformers; print(f'xformers: {xformers.__version__}')"
    
    # Final message
    print_step "Installation Complete!"
    print_success "DeepSeek-OCR with vLLM is now installed!"
    echo ""
    print_info "Next steps:"
    echo "  1. Clone DeepSeek-OCR repository if you haven't already:"
    echo "     git clone https://github.com/deepseek-ai/DeepSeek-OCR.git"
    echo ""
    echo "  2. Install DeepSeek-OCR requirements:"
    echo "     cd DeepSeek-OCR"
    echo "     pip install -r requirements.txt"
    echo ""
    echo "  3. Configure and run:"
    echo "     cd DeepSeek-OCR-master/DeepSeek-OCR-vllm"
    echo "     # Edit config.py with your paths"
    echo "     python run_dpsk_ocr_image.py"
    echo ""
    print_info "For troubleshooting, see: docs/TROUBLESHOOTING.md"
    print_info "For detailed manual installation, see: docs/INSTALL_CUDA_12.8.md"
}

# Run main function
main "$@"
