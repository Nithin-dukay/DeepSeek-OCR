# Quick Start Guide - CUDA 12.8 / RTX 5090

This is a condensed version of the full installation guide. For detailed explanations and troubleshooting, see [INSTALL_CUDA_12.8.md](INSTALL_CUDA_12.8.md).

## Prerequisites

- NVIDIA RTX 5090 (or compatible GPU)
- CUDA 12.8
- Python 3.12
- Linux system

## Quick Installation

### Option 1: Automated Script (Recommended)

```bash
# Clone the repository
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR

# Create and activate conda environment
conda create -n deepseek-ocr-cuda128 python=3.12 -y
conda activate deepseek-ocr-cuda128

# Run automated installation script
bash scripts/install_cuda128_rtx5090.sh
```

The script will:
- Install all required packages in the correct order
- Handle dependency conflicts automatically
- Verify the installation
- Provide detailed progress information

### Option 2: Manual Installation

```bash
# 1. Create environment
conda create -n deepseek-ocr-cuda128 python=3.12 -y
conda activate deepseek-ocr-cuda128

# 2. Install xformers
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Download and install wheels
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps

# 4. Install dependencies
pip install pydantic transformers cachetools cloudpickle psutil zmq msgspec blake3 numpy pillow requests tqdm packaging filelock huggingface-hub

# 5. Install torchvision
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 6. CRITICAL: Re-install xformers
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# 7. Install final dependencies
pip install hf_transfer prometheus_client sentencepiece protobuf

# 8. Install DeepSeek-OCR requirements
pip install -r requirements.txt
```

## Verification

```bash
# Run verification script
python scripts/verify_installation.py

# Or manually verify
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

## Quick Test

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Edit config.py to set your image path
# INPUT_PATH = 'path/to/your/image.jpg'
# OUTPUT_PATH = 'path/to/output/directory'

# Run inference
python run_dpsk_ocr_image.py
```

## Common Issues

### Issue: ModuleNotFoundError

**Solution:** Install the missing module and verify again:
```bash
pip install <missing_module>
python -c "import vllm; print(vllm.__version__)"
```

### Issue: vllm C++ error

**Solution:** Re-install xformers:
```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: Out of memory

**Solution:** Reduce batch size in `config.py`:
```python
MAX_CONCURRENCY = 50  # Reduce from 100
MAX_CROPS = 4  # Reduce from 6
```

## Resources

- [Full Installation Guide](INSTALL_CUDA_12.8.md)
- [GitHub Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240)
- [Pre-built Wheels](https://github.com/ghcdmm/DeepSeek-OCR/releases/tag/1)

## Credits

Installation process documented by the community in Issue #240. Pre-built wheels by @ghcdmm.
