# Installation Guide for DeepSeek-OCR

This guide provides detailed instructions for installing DeepSeek-OCR as a Python package.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- CUDA 11.8+ (for GPU support)
- PyTorch 2.6.0 (recommended)

## Installation Methods

### Method 1: Install from Source (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
   cd DeepSeek-OCR
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   conda create -n deepseek-ocr python=3.12.9 -y
   conda activate deepseek-ocr
   ```
   
   Or using venv:
   ```bash
   python3 -m venv deepseek-ocr-env
   source deepseek-ocr-env/bin/activate  # On Windows: deepseek-ocr-env\Scripts\activate
   ```

3. **Install PyTorch (CUDA 11.8):**
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```
   
   This will install DeepSeek-OCR in editable mode along with all required dependencies.

### Method 2: Install with vLLM Support

For high-performance batch inference with vLLM:

1. **Download vLLM wheel:**
   ```bash
   wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
   ```

2. **Install with vLLM extras:**
   ```bash
   pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
   pip install -e ".[vllm]"
   pip install flash-attn==2.7.3 --no-build-isolation
   ```

### Method 3: Install for Development

If you want to contribute or develop:

```bash
pip install -e ".[dev]"
```

This installs additional development tools like pytest, black, flake8, and mypy.

### Method 4: Install All Optional Dependencies

```bash
pip install -e ".[all]"
```

## Building Distribution Packages

### Build Source Distribution

```bash
python setup.py sdist
```

This creates a `.tar.gz` file in the `dist/` directory.

### Build Wheel Distribution

```bash
python setup.py bdist_wheel
```

This creates a `.whl` file in the `dist/` directory.

### Using build (Modern Method)

```bash
pip install build
python -m build
```

This creates both source and wheel distributions.

## Verifying Installation

After installation, verify that the package is correctly installed:

```python
import sys
sys.path.insert(0, 'DeepSeek-OCR-master')

# Test imports
try:
    from transformers import AutoModel, AutoTokenizer
    print("✓ Transformers imported successfully")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import einops
    import addict
    import PIL
    print("✓ All required dependencies imported successfully")
except ImportError as e:
    print(f"✗ Dependency import failed: {e}")
```

## Package Structure

After installation, the package structure is:

```
deepseek-ocr/
├── DeepSeek-OCR-hf/          # Hugging Face Transformers implementation
│   ├── __init__.py
│   └── run_dpsk_ocr.py
├── DeepSeek-OCR-vllm/        # vLLM implementation
│   ├── __init__.py
│   ├── config.py
│   ├── deepseek_ocr.py
│   ├── deepencoder/          # Vision encoder modules
│   ├── process/              # Processing utilities
│   └── run_dpsk_ocr_*.py     # Inference scripts
└── __init__.py
```

## Troubleshooting

### Issue: ModuleNotFoundError for setuptools

**Solution:**
```bash
python3 -m ensurepip --default-pip
python3 -m pip install --upgrade pip setuptools wheel
```

### Issue: CUDA version mismatch

**Solution:**
Check your CUDA version and install the appropriate PyTorch version:
```bash
nvcc --version  # Check CUDA version
# Install matching PyTorch from https://pytorch.org/get-started/locally/
```

### Issue: vLLM and transformers version conflict

**Note:** If you want vLLM and transformers codes to run in the same environment, you don't need to worry about installation errors like: `vllm 0.8.5+cu118 requires transformers>=4.51.1`

The package is configured to handle this compatibility issue.

### Issue: Flash Attention installation fails

**Solution:**
Ensure you have the correct CUDA toolkit and compiler:
```bash
# Install build essentials
sudo apt-get install build-essential

# Install flash-attn without isolation
pip install flash-attn==2.7.3 --no-build-isolation
```

## Uninstallation

To uninstall the package:

```bash
pip uninstall deepseek-ocr
```

## Additional Resources

- **GitHub Repository:** https://github.com/deepseek-ai/DeepSeek-OCR
- **Paper:** https://arxiv.org/abs/2510.18234
- **Model Download:** https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **vLLM Documentation:** https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html

## Support

For issues and questions:
- Open an issue on GitHub: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- Join Discord: https://discord.gg/Tc7c45Zzu5
- Follow on Twitter: @deepseek_ai
