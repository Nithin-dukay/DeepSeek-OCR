# DeepSeek-OCR Installation Guide for RTX 5090 with CUDA 12.8

This guide provides detailed instructions for setting up DeepSeek-OCR with vLLM on NVIDIA RTX 5090 GPUs running CUDA 12.8.

## Overview

The RTX 5090 with CUDA 12.8 requires a specific installation sequence due to dependency conflicts between PyTorch nightly builds, xformers, torchvision, and vLLM. This guide is based on a verified working setup from [GitHub Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240).

## Prerequisites

- NVIDIA RTX 5090 GPU
- CUDA 12.8 installed
- Python 3.12 (recommended)
- Conda or venv for environment management

## Environment Setup

### 1. Create a Clean Python Environment

```bash
# Using conda (recommended)
conda create -n deepseek-ocr-rtx5090 python=3.12 -y
conda activate deepseek-ocr-rtx5090

# OR using venv
python3.12 -m venv deepseek-ocr-rtx5090
source deepseek-ocr-rtx5090/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

## Installation Steps

### Step 1: Install Core vLLM Components

The key to a successful installation is using pre-built wheels and installing packages in a specific order.

#### 1.1 Install xformers (Nightly Build)

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

**Why this matters:** This installs the correct PyTorch nightly build (cu128) that's compatible with CUDA 12.8.

#### 1.2 Download Pre-built Wheels

```bash
# Download flash-attn wheel
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl

# Download vLLM wheel
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
```

**Note:** These wheels are specifically built for Python 3.12 and CUDA 12.8. If you're using a different Python version, you'll need to find or build compatible wheels.

#### 1.3 Install the Wheels

```bash
# Install flash-attn
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl

# Install vLLM with special flags
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
```

**Important flags:**
- `--no-build-isolation`: Prevents pip from creating an isolated build environment
- `--no-deps`: Skips automatic dependency installation (we'll install them manually)

### Step 2: Install Initial Dependencies

After installing the vLLM wheel, many dependencies will be missing. Install the core ones:

```bash
pip install pydantic
pip install transformers
pip install cachetools
pip install cloudpickle
pip install psutil
pip install zmq
pip install msgspec
pip install blake3
```

### Step 3: Verify Installation (First Pass)

Run this command to check what's still missing:

```bash
python -c "import vllm; print(vllm.__version__)"
```

If you get a `ModuleNotFoundError`, note the missing module and install it:

```bash
pip install <missing_module>
```

Repeat this process until the verification command runs without errors.

### Step 4: The Critical torchvision Fix

This is the trickiest part of the installation.

#### 4.1 Install torchvision

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**⚠️ WARNING:** Installing torchvision will uninstall your PyTorch nightly and replace it with a different version, which will break vLLM with C++ errors.

#### 4.2 Re-install xformers (CRITICAL)

To fix the broken environment, re-run the xformers installation:

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

This will reinstall the correct PyTorch nightly build and fix the vLLM C++ errors.

### Step 5: Install Final Dependencies

After fixing the torchvision/xformers conflict, install remaining dependencies:

```bash
pip install hf_transfer
pip install prometheus_client
```

### Step 6: Final Verification

Run the verification command again:

```bash
python -c "import vllm; print(vllm.__version__)"
```

If it prints the version number without errors, your installation is complete!

### Step 7: Install DeepSeek-OCR Requirements

```bash
# Clone the repository if you haven't already
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR

# Install additional requirements
pip install -r requirements.txt
```

## Automated Installation

For convenience, we provide an automated installation script:

```bash
cd DeepSeek-OCR
bash scripts/setup_rtx5090_cuda128.sh
```

This script automates all the steps above with proper error handling.

## Verification Script

To check your installation and identify any missing dependencies:

```bash
python scripts/verify_installation.py
```

This script will:
- Check if vLLM is properly installed
- Identify missing dependencies
- Verify CUDA availability
- Test basic vLLM functionality

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError after installation

**Solution:** Run the verification command and install missing modules one by one:

```bash
python -c "import vllm; print(vllm.__version__)"
# If it fails with: ModuleNotFoundError: No module named 'xxx'
pip install xxx
```

#### 2. vLLM C++ errors

**Symptoms:** Errors mentioning C++ compilation or CUDA kernel issues

**Solution:** Re-install xformers to get the correct PyTorch nightly:

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 3. CUDA version mismatch

**Symptoms:** Errors about CUDA version incompatibility

**Solution:** Ensure all packages are using cu128 builds:

```bash
pip list | grep cu128
```

If any packages show cu118 or other versions, reinstall them with the cu128 variant.

#### 4. Out of memory errors

**Solution:** Reduce concurrency in `config.py`:

```python
MAX_CONCURRENCY = 50  # Lower this value
MAX_CROPS = 4  # Reduce from 6 to 4
```

## Running DeepSeek-OCR

Once installation is complete, you can run DeepSeek-OCR:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Edit config.py to set your input/output paths
# Then run:
python run_dpsk_ocr_image.py  # For images
# or
python run_dpsk_ocr_pdf.py    # For PDFs
```

## Performance Notes

On an RTX 5090 with this setup, you can expect:
- High throughput for batch processing
- Efficient memory usage with proper MAX_CROPS settings
- Stable performance with the correct dependency versions

## Credits

This installation guide is based on the solution provided by the community in [GitHub Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240). Special thanks to @ghcdmm for providing the pre-built wheels.

## Additional Resources

- [Official DeepSeek-OCR Repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Support

If you encounter issues not covered in this guide:
1. Check the [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
2. Join the [DeepSeek Discord](https://discord.gg/Tc7c45Zzu5)
3. Review the [vLLM troubleshooting guide](https://docs.vllm.ai/en/latest/getting_started/debugging.html)
