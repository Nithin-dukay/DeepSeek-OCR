#!/bin/bash
# DeepSeek-OCR Installation Script for vLLM Nightly (Upstream)
# This script installs DeepSeek-OCR with the latest vLLM nightly build

set -e  # Exit on error

echo "=========================================="
echo "DeepSeek-OCR Installation (vLLM Nightly)"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if using uv or conda
USE_UV=false
if command -v uv &> /dev/null; then
    echo -e "${GREEN}Found 'uv' package manager${NC}"
    read -p "Do you want to use 'uv' for installation? (recommended by vLLM) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        USE_UV=true
    fi
fi

if [ "$USE_UV" = true ]; then
    echo -e "${GREEN}Step 1: Creating virtual environment with uv${NC}"
    uv venv
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
    echo ""
    
    echo -e "${GREEN}Step 2: Installing vLLM nightly with uv${NC}"
    uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
    echo -e "${GREEN}✓ vLLM nightly installed${NC}"
    echo ""
    
    echo -e "${GREEN}Step 3: Installing additional dependencies${NC}"
    uv pip install Pillow PyMuPDF img2pdf einops easydict addict numpy
    echo -e "${GREEN}✓ Additional dependencies installed${NC}"
    echo ""
    
else
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Error: Neither 'uv' nor 'conda' is available${NC}"
        echo "Please install one of them first:"
        echo "  - uv: pip install uv"
        echo "  - conda: Install Anaconda or Miniconda"
        exit 1
    fi
    
    ENV_NAME="deepseek-ocr-nightly"
    PYTHON_VERSION="3.11"
    
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
    
    echo -e "${GREEN}Step 3: Installing vLLM nightly${NC}"
    pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
    echo -e "${GREEN}✓ vLLM nightly installed${NC}"
    echo ""
    
    echo -e "${GREEN}Step 4: Installing additional dependencies${NC}"
    pip install Pillow PyMuPDF img2pdf einops easydict addict numpy
    echo -e "${GREEN}✓ Additional dependencies installed${NC}"
    echo ""
fi

# Install flash-attn (optional)
echo -e "${GREEN}Installing flash-attn (optional)${NC}"
echo "This may take several minutes and requires a C++ compiler..."
read -p "Do you want to install flash-attn? (recommended but optional) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ "$USE_UV" = true ]; then
        if uv pip install flash-attn --no-build-isolation; then
            echo -e "${GREEN}✓ flash-attn installed${NC}"
        else
            echo -e "${YELLOW}⚠ flash-attn installation failed (this is optional, continuing...)${NC}"
        fi
    else
        if pip install flash-attn --no-build-isolation; then
            echo -e "${GREEN}✓ flash-attn installed${NC}"
        else
            echo -e "${YELLOW}⚠ flash-attn installation failed (this is optional, continuing...)${NC}"
        fi
    fi
else
    echo "Skipping flash-attn installation"
fi
echo ""

# Verify installation
echo -e "${GREEN}Verifying installation${NC}"
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
import packaging.version
if packaging.version.parse(transformers.__version__) < packaging.version.parse('4.51.1'):
    print('⚠ WARNING: transformers version is older than 4.51.1')
    print('  vLLM nightly may require a newer version')
else:
    print('✓ Transformers version is compatible with vLLM nightly')
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Installation completed successfully!"
    echo "==========================================${NC}"
    echo ""
    if [ "$USE_UV" = true ]; then
        echo "To activate the environment, run:"
        echo "  source .venv/bin/activate"
    else
        echo "To activate the environment, run:"
        echo "  conda activate $ENV_NAME"
    fi
    echo ""
    echo "Example usage:"
    echo ""
    cat << 'EOF'
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor],
    max_model_len=8192,
    trust_remote_code=True,
    dtype="bfloat16",
)

image = Image.open("your_image.jpg").convert("RGB")
prompt = "<image>\n<|grounding|>Convert the document to markdown."

model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
    ),
    skip_special_tokens=False,
)

outputs = llm.generate(model_input, sampling_params)
print(outputs[0].outputs[0].text)
EOF
    echo ""
    echo "For more information, see INSTALLATION.md"
else
    echo ""
    echo -e "${RED}Installation verification failed${NC}"
    echo "Please check the error messages above"
    exit 1
fi
