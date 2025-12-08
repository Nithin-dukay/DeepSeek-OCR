# DeepSeek-OCR Installation Guide for CUDA 12.8 (RTX 5090)

This guide provides detailed instructions for installing DeepSeek-OCR with vLLM on systems with NVIDIA RTX 5090 and CUDA 12.8.

> **Note:** This guide is based on the solution from [Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240) and [Issue #238](https://github.com/deepseek-ai/DeepSeek-OCR/issues/238). Special thanks to @ghcdmm and the community contributors.

## Environment Requirements

- **GPU:** NVIDIA RTX 5090 (or compatible GPU with CUDA 12.8 support)
- **CUDA:** 12.8
- **Python:** 3.12
- **OS:** Linux (tested on cloud servers)

## Installation Overview

The installation process requires a specific order due to fragile nightly dependencies. The key steps are:

1. Install core vLLM, Flash Attention & xformers
2. Install initial dependencies
3. Apply the critical torchvision fix
4. Install final dependencies

## Step-by-Step Installation

### Step 1: Create and Activate Conda Environment

```bash
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr
```

### Step 2: Install Core vLLM, Flash Attention & xformers

**Important:** Using `pip install vllm` directly will NOT work. You must use pre-built wheels.

#### 2.1 Install Compatible xformers Nightly

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2.2 Download Pre-built Wheels

Download the pre-built wheels for flash-attn and vllm:

```bash
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
```

#### 2.3 Install the Wheels

**Important:** The flags `--no-build-isolation` and `--no-deps` are critical for vllm installation!

```bash
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
```

### Step 3: Install Initial Dependencies

After installing the wheels, vllm will be missing many required packages. Install the core dependencies:

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

### Step 4: Iterative Dependency Resolution

At this point, you'll need to iteratively find and install missing dependencies. Use the verification command to identify what's missing:

```bash
python -c "import vllm; print(vllm.__version__)"
```

If you get a `ModuleNotFoundError`, install the missing package and run the verification command again. Repeat until the command runs successfully.

### Step 5: The Critical torchvision Fix

**This is the most important step!** At some point, your script will crash with:

```
ModuleNotFoundError: No module named 'torchvision'
```

#### 5.1 Install torchvision Nightly

Install the nightly build that matches your PyTorch version:

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 5.2 CRITICAL: Re-install xformers

⚠️ **WARNING:** Installing torchvision will break your environment by uninstalling torch nightly and replacing it. This causes vllm C++ errors.

**Solution:** Re-run the xformers install command from Step 2.1. This will pull in the correct torch nightly again and fix the vllm C++ error:

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Step 6: Install Final Dependencies

After fixing the torchvision and xformers conflict, install the remaining dependencies:

```bash
pip install hf_transfer
pip install prometheus_client
```

Continue running the verification command and installing any remaining missing packages:

```bash
python -c "import vllm; print(vllm.__version__)"
```

### Step 7: Install DeepSeek-OCR Requirements

Finally, install the DeepSeek-OCR specific requirements:

```bash
cd DeepSeek-OCR
pip install -r requirements.txt
```

## Verification

After completing all steps, verify your installation:

```bash
# Verify vllm
python -c "import vllm; print(vllm.__version__)"

# Verify torch and torchvision
python -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}'); print(f'TorchVision: {torchvision.__version__}')"

# Verify flash-attn
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"

# Verify xformers
python -c "import xformers; print(f'xformers: {xformers.__version__}')"
```

Expected output should show:
- vllm: 0.8.5+cu128
- PyTorch: 2.7.0.dev (or similar nightly)
- TorchVision: 0.22.0.dev (or similar nightly)
- Flash Attention: 2.8.3
- xformers: 0.0.33.dev

## Running DeepSeek-OCR with vLLM

Once installation is complete, you can run DeepSeek-OCR:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Update config.py with your paths
# Then run:
python run_dpsk_ocr_image.py  # For images
python run_dpsk_ocr_pdf.py    # For PDFs
```

## Common Issues

For troubleshooting common issues, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

## Alternative: Automated Installation

For an automated installation process, use the provided script:

```bash
bash scripts/install_cuda128_rtx5090.sh
```

## Credits

This installation guide is based on the community solution from:
- [Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240)
- [Issue #238](https://github.com/deepseek-ai/DeepSeek-OCR/issues/238)
- Special thanks to @ghcdmm for providing the pre-built wheels

## Notes

- The nightly dependencies are fragile and may break with updates
- Always use the exact versions specified in this guide
- If you encounter issues, refer to the troubleshooting guide
- This setup has been tested on cloud servers with RTX 5090 and CUDA 12.8
