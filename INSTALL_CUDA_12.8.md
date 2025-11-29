# DeepSeek-OCR Installation Guide for CUDA 12.8 (RTX 5090)

This guide provides detailed instructions for installing DeepSeek-OCR with vLLM on systems with CUDA 12.8, specifically tested on NVIDIA RTX 5090.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Installation (Automated)](#quick-installation-automated)
- [Manual Installation](#manual-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

## Prerequisites

- **CUDA Version**: 12.8
- **GPU**: NVIDIA RTX 5090 (or compatible GPU with CUDA 12.8 support)
- **Python**: 3.12 (recommended)
- **Operating System**: Linux (tested on cloud servers)

## Quick Installation (Automated)

We provide an automated installation script that handles all the complex dependency ordering:

```bash
# Make the script executable
chmod +x install_cuda128.sh

# Run the installation script
./install_cuda128.sh
```

The script will:
1. Install core vLLM dependencies (xformers, flash-attn, vLLM)
2. Install required Python packages
3. Handle the critical torchvision fix
4. Verify the installation

## Manual Installation

If you prefer to install manually or need to troubleshoot, follow these detailed steps:

### Step 1: Create and Activate Conda Environment

```bash
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr
```

### Step 2: Install Core vLLM Dependencies

**Important**: The order of installation is critical!

#### 2.1 Install xformers (nightly build)

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2.2 Download Pre-built Wheels

These pre-built wheels are essential for compatibility:

```bash
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
```

#### 2.3 Install the Wheels

**Note**: The flags `--no-build-isolation` and `--no-deps` are crucial for vLLM!

```bash
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
```

### Step 3: Install Initial Dependencies

After installing the wheels, vLLM will be missing many packages. Install them:

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

### Step 4: Verify vLLM Installation

Run this command to check what's missing:

```bash
python -c "import vllm; print(vllm.__version__)"
```

If you get `ModuleNotFoundError`, install the missing package and repeat this step.

### Step 5: The Critical torchvision Fix

This is the trickiest part of the installation:

#### 5.1 Install torchvision (nightly build)

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Warning**: Installing torchvision will break your environment by replacing torch nightly!

#### 5.2 Re-install xformers (CRITICAL!)

This fixes the vLLM C++ error caused by torchvision:

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Step 6: Install Final Dependencies

Continue installing missing packages as they appear:

```bash
pip install hf_transfer
pip install prometheus_client
```

Keep running the verification command until all dependencies are satisfied:

```bash
python -c "import vllm; print(vllm.__version__)"
```

### Step 7: Install DeepSeek-OCR Requirements

```bash
pip install -r requirements.txt
```

## Verification

We provide a comprehensive verification script to check your installation:

```bash
python verify_installation.py
```

This script will:
- Check all required packages are installed
- Verify vLLM can be imported
- Test CUDA availability
- Validate GPU detection
- Check DeepSeek-OCR specific dependencies

### Manual Verification

You can also manually verify key components:

```bash
# Check vLLM version
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check flash-attn
python -c "import flash_attn; print('flash-attn imported successfully')"

# Check xformers
python -c "import xformers; print(f'xformers version: {xformers.__version__}')"
```

## Troubleshooting

### Issue: ModuleNotFoundError after installing vLLM

**Solution**: This is expected! Keep running the verification command and install missing packages one by one:

```bash
python -c "import vllm; print(vllm.__version__)"
# Install the missing package
pip install <missing_package>
# Repeat until successful
```

### Issue: vLLM C++ error after installing torchvision

**Solution**: Re-install xformers to restore the correct torch nightly:

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: "No module named 'torchvision'"

**Solution**: Install torchvision nightly, then re-install xformers:

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: CUDA version mismatch

**Solution**: Ensure you're using the correct CUDA 12.8 index URL for all PyTorch-related packages:

```bash
--index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: Installation fails with "pip install vllm"

**Solution**: Do NOT use `pip install vllm` directly. You must use the pre-built wheel:

```bash
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
```

### Issue: Out of memory during inference

**Solution**: Adjust the GPU memory utilization in your config:

```python
llm = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=0.75,  # Reduce from 0.9 if needed
    # ... other parameters
)
```

### Issue: Slow inference speed

**Solution**: 
1. Increase `MAX_CONCURRENCY` in `config.py` if you have sufficient GPU memory
2. Adjust `NUM_WORKERS` for image preprocessing
3. Consider using smaller image sizes if quality permits

## Common Dependency Versions

After successful installation, you should have approximately these versions:

- Python: 3.12
- vLLM: 0.8.5+cu128
- flash-attn: 2.8.3
- xformers: 0.0.33.dev20251104+cu128
- torch: 2.7.0.dev (nightly)
- torchvision: 0.22.0.dev (nightly)
- CUDA: 12.8

## Performance Tips

1. **GPU Memory**: RTX 5090 has 32GB VRAM. Adjust `gpu_memory_utilization` based on your workload
2. **Batch Size**: Increase `MAX_CONCURRENCY` for better throughput with multiple images
3. **Image Preprocessing**: Increase `NUM_WORKERS` to parallelize image processing
4. **Crop Mode**: Enable `CROP_MODE=True` for better quality on large documents

## Credits

This installation guide is based on the solution provided in [GitHub Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240).

Special thanks to:
- [@ghcdmm](https://github.com/ghcdmm) for providing the pre-built wheels
- The community members who contributed to solving this installation challenge
- [Issue #238](https://github.com/deepseek-ai/DeepSeek-OCR/issues/238) for initial insights

## Additional Resources

- [Official DeepSeek-OCR Repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM Documentation](https://docs.vllm.ai/)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)

## Support

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
2. Run the verification script: `python verify_installation.py`
3. Create a new issue with your error logs and system information

---

**Last Updated**: November 2025  
**Tested On**: NVIDIA RTX 5090, CUDA 12.8, Python 3.12
