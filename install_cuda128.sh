#!/bin/bash

# DeepSeek-OCR Installation Script for CUDA 12.8 (RTX 5090)
# Based on GitHub Issue #240
# This script automates the complex installation process

set -e  # Exit on error

# Color codes for output
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

# Function to verify vLLM installation
verify_vllm() {
    python -c "import vllm; print(vllm.__version__)" 2>/dev/null
    return $?
}

# Main installation process
main() {
    print_step "DeepSeek-OCR Installation for CUDA 12.8"
    
    print_info "This script will install DeepSeek-OCR with vLLM for CUDA 12.8"
    print_info "Tested on: NVIDIA RTX 5090, Python 3.12"
    print_warning "This installation requires specific dependency ordering!"
    
    # Check Python version
    print_step "Step 0: Checking Python Version"
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python version: $PYTHON_VERSION"
    
    if [[ ! "$PYTHON_VERSION" =~ ^3\.12 ]]; then
        print_warning "Recommended Python version is 3.12, you have $PYTHON_VERSION"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Installation cancelled"
            exit 1
        fi
    fi
    
    # Step 1: Install xformers
    print_step "Step 1: Installing xformers (nightly build)"
    print_info "Installing xformers==0.0.33.dev20251104+cu128..."
    pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
    print_success "xformers installed successfully"
    
    # Step 2: Download pre-built wheels
    print_step "Step 2: Downloading Pre-built Wheels"
    
    if [ ! -f "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl" ]; then
        print_info "Downloading flash-attn wheel..."
        wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
        print_success "flash-attn wheel downloaded"
    else
        print_info "flash-attn wheel already exists, skipping download"
    fi
    
    if [ ! -f "vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl" ]; then
        print_info "Downloading vLLM wheel..."
        wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
        print_success "vLLM wheel downloaded"
    else
        print_info "vLLM wheel already exists, skipping download"
    fi
    
    # Step 3: Install the wheels
    print_step "Step 3: Installing Pre-built Wheels"
    
    print_info "Installing flash-attn..."
    pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
    print_success "flash-attn installed"
    
    print_info "Installing vLLM (with --no-build-isolation --no-deps)..."
    pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
    print_success "vLLM wheel installed"
    
    # Step 4: Install initial dependencies
    print_step "Step 4: Installing Initial Dependencies"
    
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
    
    # Step 5: Iterative dependency resolution
    print_step "Step 5: Resolving Missing Dependencies"
    
    print_info "Checking for missing vLLM dependencies..."
    MAX_ITERATIONS=20
    ITERATION=0
    
    while [ $ITERATION -lt $MAX_ITERATIONS ]; do
        ITERATION=$((ITERATION + 1))
        print_info "Iteration $ITERATION/$MAX_ITERATIONS: Checking vLLM import..."
        
        # Try to import vLLM and capture the error
        ERROR_OUTPUT=$(python -c "import vllm; print(vllm.__version__)" 2>&1)
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            print_success "vLLM imported successfully! Version: $ERROR_OUTPUT"
            break
        else
            # Extract missing module name
            MISSING_MODULE=$(echo "$ERROR_OUTPUT" | grep -oP "No module named '\K[^']+")
            
            if [ -z "$MISSING_MODULE" ]; then
                print_error "Unknown error occurred:"
                echo "$ERROR_OUTPUT"
                print_warning "You may need to install additional dependencies manually"
                break
            fi
            
            print_warning "Missing module: $MISSING_MODULE"
            print_info "Installing $MISSING_MODULE..."
            
            # Try to install the missing module
            pip install "$MISSING_MODULE" || {
                print_warning "Failed to install $MISSING_MODULE, trying alternative names..."
                # Try common alternatives
                case "$MISSING_MODULE" in
                    "zmq")
                        pip install pyzmq
                        ;;
                    *)
                        print_error "Could not resolve $MISSING_MODULE"
                        ;;
                esac
            }
        fi
    done
    
    if [ $ITERATION -eq $MAX_ITERATIONS ]; then
        print_warning "Reached maximum iterations. Some dependencies may still be missing."
    fi
    
    # Step 6: The critical torchvision fix
    print_step "Step 6: Installing torchvision (Critical Fix)"
    
    print_warning "Installing torchvision will temporarily break the environment!"
    print_info "Installing torchvision nightly..."
    pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    print_success "torchvision installed"
    
    print_warning "Re-installing xformers to fix the environment..."
    pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
    print_success "xformers re-installed, environment fixed!"
    
    # Step 7: Install final dependencies
    print_step "Step 7: Installing Final Dependencies"
    
    FINAL_DEPS=(
        "hf_transfer"
        "prometheus_client"
    )
    
    for dep in "${FINAL_DEPS[@]}"; do
        print_info "Installing $dep..."
        pip install "$dep" || print_warning "Failed to install $dep (may not be critical)"
    done
    
    # Step 8: Install DeepSeek-OCR requirements
    print_step "Step 8: Installing DeepSeek-OCR Requirements"
    
    if [ -f "requirements.txt" ]; then
        print_info "Installing from requirements.txt..."
        pip install -r requirements.txt
        print_success "DeepSeek-OCR requirements installed"
    else
        print_warning "requirements.txt not found, skipping"
    fi
    
    # Step 9: Final verification
    print_step "Step 9: Final Verification"
    
    print_info "Verifying installation..."
    
    # Check vLLM
    if verify_vllm; then
        VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
        print_success "vLLM is working! Version: $VLLM_VERSION"
    else
        print_error "vLLM verification failed!"
        exit 1
    fi
    
    # Check PyTorch and CUDA
    print_info "Checking PyTorch and CUDA..."
    TORCH_INFO=$(python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')")
    print_info "$TORCH_INFO"
    
    # Check flash-attn
    if check_package "flash_attn"; then
        print_success "flash-attn is installed"
    else
        print_warning "flash-attn check failed"
    fi
    
    # Check xformers
    if check_package "xformers"; then
        XFORMERS_VERSION=$(python -c "import xformers; print(xformers.__version__)")
        print_success "xformers is installed: $XFORMERS_VERSION"
    else
        print_warning "xformers check failed"
    fi
    
    # Final summary
    print_step "Installation Complete!"
    
    print_success "DeepSeek-OCR installation finished successfully!"
    echo ""
    print_info "Next steps:"
    echo "  1. Run 'python verify_installation.py' for comprehensive verification"
    echo "  2. Configure your settings in 'DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py'"
    echo "  3. Test with: cd DeepSeek-OCR-master/DeepSeek-OCR-vllm && python run_dpsk_ocr_image.py"
    echo ""
    print_info "For troubleshooting, see: INSTALL_CUDA_12.8.md"
    echo ""
}

# Run main function
main "$@"
