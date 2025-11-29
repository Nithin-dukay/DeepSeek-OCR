# Quick Start Guide - CUDA 12.8 Installation

This is a condensed version of the full installation guide. For detailed instructions, see [INSTALL_CUDA_12.8.md](INSTALL_CUDA_12.8.md).

## Prerequisites
- CUDA 12.8
- Python 3.12
- NVIDIA RTX 5090 (or compatible GPU)

## Installation (3 Steps)

### 1. Automated Installation
```bash
chmod +x install_cuda128.sh
./install_cuda128.sh
```

### 2. Verify Installation
```bash
python verify_installation.py
```

### 3. Configure and Test
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
# Edit config.py with your settings
python run_dpsk_ocr_image.py
```

## Manual Installation (If Automated Fails)

```bash
# 1. Install xformers
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# 2. Download wheels
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl

# 3. Install wheels
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps

# 4. Install dependencies
pip install pydantic transformers cachetools cloudpickle psutil zmq msgspec blake3

# 5. Install torchvision + fix
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# 6. Install remaining deps
pip install hf_transfer prometheus_client
pip install -r requirements.txt
```

## Troubleshooting

### vLLM won't import
```bash
# Check what's missing
python -c "import vllm; print(vllm.__version__)"
# Install the missing package
pip install <missing_package>
```

### C++ error after torchvision
```bash
# Re-install xformers
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

## Support

- Full guide: [INSTALL_CUDA_12.8.md](INSTALL_CUDA_12.8.md)
- GitHub Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- Original solution: Issue #240
