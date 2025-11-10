# DeepSeek-OCR Installation Guide for CUDA 12.8 / RTX 5090

This guide provides detailed instructions for installing DeepSeek-OCR with vLLM on systems running CUDA 12.8, specifically tested on NVIDIA RTX 5090.

> **Note:** This installation process addresses GitHub Issue #240 and has been verified to work on cloud servers with RTX 5090 GPUs.

## Prerequisites

- **GPU**: NVIDIA RTX 5090 (or compatible GPU with CUDA 12.8 support)
- **CUDA**: Version 12.8
- **Python**: 3.12.x (recommended: 3.12.9)
- **OS**: Linux (tested on cloud servers)

## Quick Start

For automated installation, use our installation script:

```bash
bash scripts/install_cuda128.sh
```

For manual installation, follow the detailed steps below.

## Manual Installation Steps

### Step 1: Create Conda Environment

```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

### Step 2: Install Core vLLM Dependencies

The installation order is **critical**. Do not skip or reorder these steps.

#### 2.1 Install xformers (Nightly Build)

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

This will automatically install the compatible PyTorch nightly build.

#### 2.2 Download Pre-built Wheels

Download the pre-built wheels for flash-attn and vLLM:

```bash
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
```

> **Important:** Using `pip install vllm` directly will **not** work. You must use these pre-built wheels.

#### 2.3 Install Flash Attention and vLLM

```bash
# Install flash-attn
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl

# Install vLLM with special flags (CRITICAL!)
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
```

The `--no-build-isolation` and `--no-deps` flags are **essential** to prevent dependency conflicts.

### Step 3: Install Initial Dependencies

After installing vLLM, many required packages will be missing. Install the core dependencies:

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

### Step 4: Iterative Dependency Installation

At this point, vLLM will still have missing dependencies. Use this verification loop to find and install them:

```bash
python -c "import vllm; print(vllm.__version__)"
```

If you get a `ModuleNotFoundError`, install the missing module and repeat. Common missing packages include:

```bash
pip install hf_transfer
pip install prometheus_client
pip install ray
pip install fastapi
pip install uvicorn
pip install numpy
pip install pillow
```

Keep running the verification command and installing missing packages until it succeeds.

### Step 5: The Critical torchvision Fix

⚠️ **This is the most important step!**

#### 5.1 Install torchvision

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Problem:** Installing torchvision will **break your environment** by uninstalling the torch nightly and replacing it with an incompatible version. This causes vLLM C++ errors.

#### 5.2 Re-install xformers (CRITICAL!)

To fix the broken environment, re-run the xformers installation:

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

This will restore the correct torch nightly build and fix the vLLM C++ errors.

### Step 6: Install DeepSeek-OCR Requirements

```bash
pip install -r requirements.txt
```

### Step 7: Final Verification

Run the verification command one more time to ensure everything is working:

```bash
python -c "import vllm; print(vllm.__version__)"
```

You should see the vLLM version printed without any errors.

For a comprehensive verification, use our verification script:

```bash
python scripts/verify_installation.py
```

## Common Issues and Solutions

### Issue 1: `ModuleNotFoundError: No module named 'torchvision'`

**Solution:** Follow Step 5 exactly. Make sure to re-install xformers after installing torchvision.

### Issue 2: vLLM C++ errors after installing torchvision

**Solution:** Re-install xformers as described in Step 5.2. This will restore the correct torch version.

### Issue 3: `pip install vllm` fails or causes errors

**Solution:** Do not use `pip install vllm`. You must use the pre-built wheel from Step 2.2.

### Issue 4: Import errors for various packages

**Solution:** Follow the iterative installation process in Step 4. Run the verification command, install the missing package, and repeat.

### Issue 5: CUDA version mismatch

**Solution:** Ensure your system has CUDA 12.8 installed. Check with:

```bash
nvcc --version
nvidia-smi
```

## Verification Checklist

After installation, verify the following:

- [ ] `python -c "import vllm; print(vllm.__version__)"` runs without errors
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] `python -c "import torch; print(torch.version.cuda)"` shows `12.8`
- [ ] `python -c "import flash_attn"` runs without errors
- [ ] `python -c "import xformers"` runs without errors
- [ ] `python scripts/verify_installation.py` passes all checks

## Testing Your Installation

To test DeepSeek-OCR with vLLM:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
# Update config.py with your input/output paths
python run_dpsk_ocr_image.py
```

## Credits

This installation guide is based on the solution provided by the community in GitHub Issue #240. Special thanks to:
- Issue #238 contributors
- @ghcdmm for providing the pre-built wheels

## Additional Resources

- [Official vLLM Documentation](https://docs.vllm.ai/)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

## Support

If you encounter issues not covered in this guide, please:
1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Search existing GitHub issues
3. Open a new issue with detailed error messages and your environment details
