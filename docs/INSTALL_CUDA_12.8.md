# DeepSeek-OCR Installation Guide for CUDA 12.8 / RTX 5090

This guide provides detailed instructions for installing DeepSeek-OCR with vLLM on systems with NVIDIA RTX 5090 (or similar high-end GPUs) running CUDA 12.8.

> **Note:** This installation process was successfully tested and documented by the community in [GitHub Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240). Special thanks to @ghcdmm for providing the pre-built wheels.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
  - [Step 1: Install Core vLLM, Flash Attention & xformers](#step-1-install-core-vllm-flash-attention--xformers)
  - [Step 2: Install Initial Dependencies](#step-2-install-initial-dependencies)
  - [Step 3: The Critical torchvision Fix](#step-3-the-critical-torchvision-fix)
  - [Step 4: Final Dependencies](#step-4-final-dependencies)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Automated Installation](#automated-installation)

## Prerequisites

- **GPU:** NVIDIA RTX 5090 or compatible GPU with CUDA 12.8 support
- **CUDA:** CUDA 12.8 installed and configured
- **Python:** Python 3.12 (recommended)
- **System:** Linux (tested on cloud servers with Amazon Linux 2023)
- **Conda:** Anaconda or Miniconda installed

## Installation Steps

### Environment Setup

First, create and activate a new conda environment:

```bash
conda create -n deepseek-ocr-cuda128 python=3.12 -y
conda activate deepseek-ocr-cuda128
```

### Step 1: Install Core vLLM, Flash Attention & xformers

The key to a successful installation is using pre-built wheels and installing packages in a specific order. The nightly dependencies are fragile and require careful handling.

#### 1.1 Install xformers nightly

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

This will install the compatible PyTorch nightly build along with xformers.

#### 1.2 Download pre-built wheels

Download the pre-built wheels for flash-attn and vllm:

```bash
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
```

> **Important:** Using `pip install vllm` directly will NOT work. You must use these pre-built wheels.

#### 1.3 Install the wheels

```bash
# Install flash-attn
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl

# Install vllm with special flags
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
```

> **Note:** The `--no-build-isolation --no-deps` flags are critical for vllm installation.

### Step 2: Install Initial Dependencies

After installing the wheels, vllm will still be missing many required packages. Install the core dependencies:

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

#### Iterative Dependency Resolution

After installing the initial dependencies, you'll likely encounter `ModuleNotFoundError` when trying to import vllm. The best approach is to iteratively discover and install missing packages:

```bash
# Run this command to check for missing dependencies
python -c "import vllm; print(vllm.__version__)"
```

If you get a `ModuleNotFoundError`, install the missing package and repeat the verification command until it succeeds.

Common missing packages include:
- `numpy`
- `pillow`
- `requests`
- `tqdm`
- `packaging`
- `filelock`
- `huggingface-hub`

### Step 3: The Critical torchvision Fix

This is the trickiest part of the installation. At some point, your script (or vLLM) will crash with:

```
ModuleNotFoundError: No module named 'torchvision'
```

#### 3.1 Install torchvision nightly

You must install the nightly build that matches your PyTorch version:

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

> **⚠️ CRITICAL WARNING:** Installing torchvision will break your environment by uninstalling your torch nightly and replacing it with an incompatible version. This will cause a vllm C++ error.

#### 3.2 Re-install xformers (REQUIRED)

To fix the broken environment, you MUST re-run the xformers install command from Step 1.1:

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

This will pull in the correct torch nightly again and fix the vllm C++ error.

#### Why does this happen?

The torchvision installation process:
1. Detects your current PyTorch version
2. Uninstalls it
3. Installs a version it thinks is compatible (but isn't)
4. This breaks the C++ bindings that vllm depends on

Re-installing xformers forces pip to reinstall the correct PyTorch nightly version that's compatible with both xformers and vllm.

### Step 4: Final Dependencies

After fixing the torchvision/xformers conflict, run the verification command again:

```bash
python -c "import vllm; print(vllm.__version__)"
```

You may need to install a few final packages:

```bash
pip install hf_transfer
pip install prometheus_client
pip install sentencepiece
pip install protobuf
```

Keep running the verification command and installing missing packages until it succeeds without any `ImportError`.

### Install DeepSeek-OCR Requirements

Finally, install the DeepSeek-OCR specific requirements:

```bash
# Clone the repository if you haven't already
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR

# Install requirements
pip install -r requirements.txt
```

## Verification

### Verify vLLM Installation

Run the verification script:

```bash
python scripts/verify_installation.py
```

Or manually verify:

```bash
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
python -c "import xformers; print(f'xformers version: {xformers.__version__}')"
```

Expected output (versions may vary slightly):
```
vLLM version: 0.8.5+cu128
PyTorch version: 2.7.0.dev20251104+cu128
Torchvision version: 0.22.0.dev20251104+cu128
xformers version: 0.0.33.dev20251104+cu128
```

### Test DeepSeek-OCR

Test the installation with a sample image:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Update config.py with your image path
# Then run:
python run_dpsk_ocr_image.py
```

## Troubleshooting

### Issue: `ModuleNotFoundError` after installing packages

**Solution:** This is expected during the iterative installation process. Simply install the missing package and continue:

```bash
pip install <missing_package_name>
python -c "import vllm; print(vllm.__version__)"
```

### Issue: vllm C++ error or segmentation fault

**Symptoms:**
```
ImportError: /path/to/vllm/_C.so: undefined symbol: ...
```

**Solution:** This usually means your PyTorch version is incompatible. Re-install xformers:

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: CUDA version mismatch

**Symptoms:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:** Ensure you're using the correct CUDA version (12.8) and that all packages are built for cu128:
- Check: `nvcc --version`
- Verify all wheel URLs contain `cu128`

### Issue: Out of memory errors

**Solution:** Reduce the batch size or concurrency in `config.py`:

```python
MAX_CONCURRENCY = 50  # Reduce from 100
MAX_CROPS = 4  # Reduce from 6
```

### Issue: Pre-built wheels not found (404 error)

**Solution:** The wheels are hosted on a community release. If the links are broken:
1. Check the original issue #240 for updated links
2. Try building from source (more complex, not recommended)
3. Use the official vLLM nightly build (may require different steps)

### Issue: Python version mismatch

**Symptoms:**
```
ERROR: vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl is not a supported wheel on this platform.
```

**Solution:** The pre-built wheels are for Python 3.12. Ensure you're using Python 3.12:

```bash
python --version  # Should show Python 3.12.x
```

If not, recreate your conda environment with Python 3.12.

## Automated Installation

For a fully automated installation, use the provided script:

```bash
bash scripts/install_cuda128_rtx5090.sh
```

This script will:
1. Check prerequisites
2. Install all packages in the correct order
3. Handle the torchvision/xformers conflict automatically
4. Verify the installation
5. Provide a detailed log

## Package Versions Summary

For reference, here are the key package versions used in this setup:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12.x | Required for pre-built wheels |
| CUDA | 12.8 | Must match system CUDA |
| PyTorch | 2.7.0.dev20251104+cu128 | Nightly build |
| xformers | 0.0.33.dev20251104+cu128 | Nightly build |
| torchvision | 0.22.0.dev20251104+cu128 | Nightly build |
| vllm | 0.8.5+cu128 | Pre-built wheel |
| flash-attn | 2.8.3 | Pre-built wheel |

## Additional Resources

- [Original Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240)
- [vLLM Documentation](https://docs.vllm.ai/)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [Pre-built Wheels Release](https://github.com/ghcdmm/DeepSeek-OCR/releases/tag/1)

## Credits

- Installation process documented by the community in Issue #240
- Pre-built wheels provided by @ghcdmm
- Thanks to Issue #238 contributors for initial troubleshooting

## License

This installation guide is part of the DeepSeek-OCR project. See the main LICENSE file for details.
