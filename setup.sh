#!/bin/bash

# DeepSeek-OCR Setup Script
# This script automates the installation of DeepSeek-OCR and its dependencies
# in the correct order to avoid the GenerationMixin ImportError

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to detect CUDA version
detect_cuda() {
    if command_exists nvidia-smi; then
        CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
        print_info "Detected CUDA version: $CUDA_VERSION"
        
        if [[ "$CUDA_VERSION" == 11.* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            print_info "Using PyTorch with CUDA 11.8"
        elif [[ "$CUDA_VERSION" == 12.* ]]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            print_info "Using PyTorch with CUDA 12.1"
        else
            print_warning "Unsupported CUDA version. Defaulting to CUDA 11.8"
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        fi
    else
        print_warning "CUDA not detected. Installing CPU-only version"
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi
}

# Function to check Python version
check_python() {
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.10 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | grep -oP "\d+\.\d+")
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        print_error "Python 3.10 or higher is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python version $PYTHON_VERSION detected"
}

# Function to create virtual environment
create_venv() {
    print_info "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Skipping creation."
        return
    fi
    
    python3 -m venv venv
    print_success "Virtual environment created"
}

# Function to activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Function to upgrade pip
upgrade_pip() {
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    print_success "pip upgraded"
}

# Function to install transformers
install_transformers() {
    print_info "Installing transformers (Step 1/5)..."
    print_warning "This must be installed FIRST to avoid ImportError!"
    pip install "transformers>=4.51.1" "tokenizers>=0.20.3"
    print_success "Transformers installed"
}

# Function to install PyTorch
install_pytorch() {
    print_info "Installing PyTorch (Step 2/5)..."
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url "$TORCH_INDEX"
    print_success "PyTorch installed"
}

# Function to install vLLM
install_vllm() {
    print_info "Installing vLLM (Step 3/5)..."
    
    read -p "Install vLLM nightly (recommended) or stable 0.8.5? [nightly/stable] (default: nightly): " VLLM_VERSION
    VLLM_VERSION=${VLLM_VERSION:-nightly}
    
    if [ "$VLLM_VERSION" == "nightly" ]; then
        print_info "Installing vLLM nightly build..."
        pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
    else
        print_info "Installing vLLM 0.8.5..."
        print_warning "You need to download the wheel file manually from:"
        print_warning "https://github.com/vllm-project/vllm/releases/tag/v0.8.5"
        read -p "Enter path to vLLM wheel file (or press Enter to skip): " VLLM_WHEEL
        
        if [ -n "$VLLM_WHEEL" ] && [ -f "$VLLM_WHEEL" ]; then
            pip install "$VLLM_WHEEL"
        else
            print_warning "Skipping vLLM installation. Install manually later."
            return
        fi
    fi
    
    print_success "vLLM installed"
}

# Function to install other dependencies
install_dependencies() {
    print_info "Installing other dependencies (Step 4/5)..."
    pip install PyMuPDF img2pdf einops easydict addict Pillow numpy
    print_success "Dependencies installed"
}

# Function to install flash-attn
install_flash_attn() {
    print_info "Installing flash-attn (Step 5/5 - Optional)..."
    print_warning "This may take several minutes and might fail on some systems."
    
    read -p "Install flash-attn? [y/N] (default: y): " INSTALL_FLASH
    INSTALL_FLASH=${INSTALL_FLASH:-y}
    
    if [ "$INSTALL_FLASH" == "y" ] || [ "$INSTALL_FLASH" == "Y" ]; then
        if pip install flash-attn==2.7.3 --no-build-isolation; then
            print_success "flash-attn installed"
        else
            print_warning "flash-attn installation failed. This is optional and can be skipped."
            print_warning "The model will work without it, but may be slower."
        fi
    else
        print_info "Skipping flash-attn installation"
    fi
}

# Function to verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    python3 -c "import transformers; print(f'transformers: {transformers.__version__}')" || print_error "transformers not installed correctly"
    python3 -c "import torch; print(f'torch: {torch.__version__}')" || print_error "torch not installed correctly"
    python3 -c "import vllm; print(f'vllm: {vllm.__version__}')" || print_error "vllm not installed correctly"
    
    print_success "Installation verification complete"
}

# Function to print next steps
print_next_steps() {
    echo ""
    print_success "=========================================="
    print_success "Installation Complete!"
    print_success "=========================================="
    echo ""
    print_info "Next steps:"
    echo "  1. Activate the virtual environment:"
    echo "     ${GREEN}source venv/bin/activate${NC}"
    echo ""
    echo "  2. Navigate to the vLLM directory:"
    echo "     ${GREEN}cd DeepSeek-OCR-master/DeepSeek-OCR-vllm${NC}"
    echo ""
    echo "  3. Update config.py with your settings"
    echo ""
    echo "  4. Run inference:"
    echo "     ${GREEN}python run_dpsk_ocr_image.py${NC}  # For images"
    echo "     ${GREEN}python run_dpsk_ocr_pdf.py${NC}    # For PDFs"
    echo ""
    print_info "For more information, see INSTALLATION.md"
    echo ""
}

# Main installation flow
main() {
    echo ""
    print_info "=========================================="
    print_info "DeepSeek-OCR Setup Script"
    print_info "=========================================="
    echo ""
    
    # Check prerequisites
    check_python
    detect_cuda
    
    # Ask for installation method
    read -p "Create new virtual environment? [Y/n] (default: Y): " CREATE_VENV
    CREATE_VENV=${CREATE_VENV:-Y}
    
    if [ "$CREATE_VENV" == "Y" ] || [ "$CREATE_VENV" == "y" ]; then
        create_venv
        activate_venv
    else
        print_warning "Skipping virtual environment creation"
        print_warning "Make sure you're in the correct environment!"
    fi
    
    # Upgrade pip
    upgrade_pip
    
    # Install dependencies in correct order
    install_transformers
    install_pytorch
    install_vllm
    install_dependencies
    install_flash_attn
    
    # Verify installation
    verify_installation
    
    # Print next steps
    print_next_steps
}

# Run main function
main
