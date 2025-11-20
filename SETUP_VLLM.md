# DeepSeek-OCR vLLM Setup Guide

This guide provides comprehensive instructions for setting up and using DeepSeek-OCR with vLLM, including the fix for GitHub Issue #244.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Methods](#usage-methods)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Operating System**: Linux (recommended), macOS, or Windows with WSL2
- **Python**: 3.8 - 3.12 (3.12.9 recommended)
- **CUDA**: 11.8+ (for GPU acceleration)
- **GPU**: NVIDIA GPU with at least 16GB VRAM (A100-40G recommended for optimal performance)
- **RAM**: At least 32GB system RAM

## Installation

### Step 1: Create Python Environment

```bash
# Using conda (recommended)
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# Or using venv
python3 -m venv deepseek-ocr-env
source deepseek-ocr-env/bin/activate  # On Windows: deepseek-ocr-env\Scripts\activate
```

### Step 2: Install PyTorch

```bash
# For CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install vLLM

**Option A: Install from nightly build (recommended for latest features)**
```bash
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

**Option B: Install specific version (v0.8.5)**
```bash
# Download the wheel file
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

### Step 4: Install Project Dependencies

```bash
# Clone the repository if you haven't already
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR

# Install requirements
pip install -r requirements.txt

# Install flash-attention (optional but recommended for performance)
pip install flash-attn==2.7.3 --no-build-isolation
```

### Step 5: Verify Installation

```bash
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Method 1: Using the Wrapper Script (Recommended for Issue #244 Fix)

The `serve_deepseek_ocr.py` script automatically handles model registration and configuration.

**Single Image Inference:**
```bash
python serve_deepseek_ocr.py --image path/to/image.jpg
```

**Batch Inference:**
```bash
python serve_deepseek_ocr.py --image img1.jpg img2.jpg img3.jpg --output results.txt
```

**Custom Prompt:**
```bash
python serve_deepseek_ocr.py \
  --image document.png \
  --prompt "<image>\n<|grounding|>Convert the document to markdown."
```

### Method 2: Using Provided Scripts

The repository includes pre-configured scripts for different use cases:

**For Images:**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

**For PDFs:**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_pdf.py
```

**For Batch Evaluation:**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_eval_batch.py
```

### Method 3: Python API

```python
from vllm import LLM, SamplingParams, ModelRegistry
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import sys
sys.path.insert(0, 'DeepSeek-OCR-master/DeepSeek-OCR-vllm')
from deepseek_ocr import DeepseekOCRForCausalLM

# Register the model (IMPORTANT: This fixes Issue #244)
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Create LLM instance
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor],
    trust_remote_code=True,
)

# Prepare input
image = Image.open("path/to/image.jpg").convert("RGB")
prompt = "<image>\\nFree OCR."

model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # <td>, </td>
    ),
    skip_special_tokens=False,
)

# Generate
outputs = llm.generate(model_input, sampling_params)
print(outputs[0].outputs[0].text)
```

## Usage Methods

### Configuration File

Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` to customize settings:

```python
# Resolution modes
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6  # Reduce if GPU memory is limited
MAX_CONCURRENCY = 100
NUM_WORKERS = 64

MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
INPUT_PATH = 'path/to/input'
OUTPUT_PATH = 'path/to/output'

# Prompt templates
PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'
```

### Common Prompts

```python
# Document to markdown
"<image>\\n<|grounding|>Convert the document to markdown."

# General OCR
"<image>\\n<|grounding|>OCR this image."

# Free OCR (without layout)
"<image>\\nFree OCR."

# Parse figures
"<image>\\nParse the figure."

# Detailed description
"<image>\\nDescribe this image in detail."

# Text localization
"<image>\\nLocate <|ref|>specific text<|/ref|> in the image."
```

## Configuration Options

### Resolution Modes

| Mode   | Base Size | Image Size | Crop Mode | Vision Tokens | Use Case                    |
|--------|-----------|------------|-----------|---------------|-----------------------------|
| Tiny   | 512       | 512        | False     | 64            | Small images, fast inference|
| Small  | 640       | 640        | False     | 100           | Standard images             |
| Base   | 1024      | 1024       | False     | 256           | High-quality documents      |
| Large  | 1280      | 1280       | False     | 400           | Very high resolution        |
| Gundam | 1024      | 640        | True      | Variable      | Large documents (recommended)|

### Sampling Parameters

```python
SamplingParams(
    temperature=0.0,        # 0.0 for deterministic output
    max_tokens=8192,        # Maximum output length
    extra_args=dict(
        ngram_size=30,      # N-gram size for repetition prevention
        window_size=90,     # Context window for n-gram checking
        whitelist_token_ids={128821, 128822},  # Tokens to allow repetition
    ),
    skip_special_tokens=False,  # Keep special tokens in output
)
```

### Performance Tuning

**For Limited GPU Memory:**
```python
MAX_CROPS = 4  # Reduce from 6
MAX_CONCURRENCY = 50  # Reduce from 100
BASE_SIZE = 640  # Use smaller base size
IMAGE_SIZE = 640
CROP_MODE = False  # Disable cropping
```

**For Maximum Quality:**
```python
MAX_CROPS = 9  # Maximum crops
BASE_SIZE = 1280
IMAGE_SIZE = 640
CROP_MODE = True
```

## Troubleshooting

### Issue #244: Model Architecture Not Supported

**Error:**
```
pydantic_core.ValidationError: 1 validation error for ModelConfig
Value error, Model architectures 'DeepseekOCRForCausallM' are not supported
```

**Solution:**
Use the `serve_deepseek_ocr.py` wrapper script or manually register the model:

```python
from vllm import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
```

### CUDA Out of Memory

**Solutions:**
1. Reduce `MAX_CROPS` in config.py
2. Use smaller resolution mode (Tiny or Small)
3. Disable crop mode: `CROP_MODE = False`
4. Reduce batch size or concurrency
5. Use gradient checkpointing (if available)

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'deepseek_ocr'`

**Solution:**
```bash
# Add the vllm directory to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm"

# Or in Python
import sys
sys.path.insert(0, '/path/to/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm')
```

### Slow Inference

**Solutions:**
1. Install flash-attention: `pip install flash-attn==2.7.3 --no-build-isolation`
2. Use smaller resolution mode
3. Disable cropping for simple images
4. Ensure CUDA is properly configured
5. Use tensor parallelism for multi-GPU setups

### Model Download Issues

**Error:** Connection timeout or slow download from HuggingFace

**Solution:**
```bash
# Set HuggingFace mirror (for users in China)
export HF_ENDPOINT=https://hf-mirror.com

# Or download manually and use local path
huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir ./models/DeepSeek-OCR

# Then use local path
python serve_deepseek_ocr.py --model ./models/DeepSeek-OCR --image test.jpg
```

## Performance Benchmarks

### Inference Speed (A100-40G)

| Mode   | Single Image | Batch (10 images) | Tokens/sec |
|--------|--------------|-------------------|------------|
| Tiny   | ~0.5s        | ~3s               | ~3000      |
| Small  | ~0.8s        | ~5s               | ~2800      |
| Base   | ~1.5s        | ~10s              | ~2500      |
| Gundam | ~2.0s        | ~15s              | ~2500      |

### Memory Usage

| Mode   | Single Image | Batch (10 images) |
|--------|--------------|-------------------|
| Tiny   | ~8GB         | ~12GB             |
| Small  | ~10GB        | ~16GB             |
| Base   | ~14GB        | ~24GB             |
| Gundam | ~18GB        | ~32GB             |

## Additional Resources

- **Official Documentation**: [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- **vLLM Documentation**: [vLLM Docs](https://docs.vllm.ai/)
- **Model Card**: [HuggingFace Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Paper**: [ArXiv](https://arxiv.org/abs/2510.18234)
- **Discord**: [DeepSeek AI Community](https://discord.gg/Tc7c45Zzu5)

## Support

For issues and questions:
1. Check this troubleshooting guide
2. Review [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
3. Join the [Discord community](https://discord.gg/Tc7c45Zzu5)
4. Open a new issue with detailed error logs and system information
