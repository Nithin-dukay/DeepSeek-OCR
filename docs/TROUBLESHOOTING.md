# DeepSeek-OCR Troubleshooting Guide

This guide covers common issues and solutions for DeepSeek-OCR installation and usage, with a focus on CUDA 12.8 / RTX 5090 configurations.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Import Errors](#import-errors)
- [CUDA and GPU Issues](#cuda-and-gpu-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [General Tips](#general-tips)

---

## Installation Issues

### Issue: `ModuleNotFoundError: No module named 'torchvision'`

**Symptoms:**
- Error occurs when running vLLM or DeepSeek-OCR scripts
- Missing torchvision module

**Solution:**
1. Install torchvision from nightly build:
   ```bash
   pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

2. **CRITICAL:** Re-install xformers immediately after:
   ```bash
   pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
   ```

**Why this happens:** Installing torchvision replaces the torch nightly with an incompatible version, breaking vLLM.

---

### Issue: vLLM C++ errors after installing torchvision

**Symptoms:**
- Errors like `undefined symbol` or C++ compilation errors
- vLLM fails to import after installing torchvision
- Errors mentioning CUDA or torch C++ extensions

**Solution:**
Re-install xformers to restore the correct torch version:
```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

**Prevention:** Always re-install xformers after installing torchvision. This is documented in Step 5.2 of the installation guide.

---

### Issue: `pip install vllm` fails or causes errors

**Symptoms:**
- Installation fails with compilation errors
- vLLM doesn't work after installation
- Missing CUDA libraries

**Solution:**
Do NOT use `pip install vllm`. Instead, use the pre-built wheel:

```bash
# Download the wheel
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl

# Install with special flags
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
```

**Why:** The pre-built wheel is specifically compiled for CUDA 12.8 and Python 3.12.

---

### Issue: Multiple `ModuleNotFoundError` for various packages

**Symptoms:**
- Repeated import errors for different modules
- vLLM complains about missing dependencies

**Solution:**
Use the iterative installation approach:

1. Run the verification command:
   ```bash
   python -c "import vllm; print(vllm.__version__)"
   ```

2. Note the missing module name from the error

3. Install the missing module:
   ```bash
   pip install <module_name>
   ```

4. Repeat steps 1-3 until successful

**Common missing modules:**
```bash
pip install hf_transfer prometheus_client ray fastapi uvicorn numpy pillow
```

---

### Issue: Python version incompatibility

**Symptoms:**
- Wheel installation fails with "not a supported wheel"
- Python version errors

**Solution:**
The pre-built wheels require Python 3.12. Create a new environment:

```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

Then restart the installation process.

---

## Import Errors

### Issue: `ImportError: cannot import name 'NGramPerReqLogitsProcessor'`

**Symptoms:**
- Error when trying to use upstream vLLM code
- Missing logits processor

**Solution:**
This is for the upstream vLLM version. For the custom installation, use:

```python
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
```

Instead of:
```python
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
```

---

### Issue: `ModuleNotFoundError: No module named 'flash_attn'`

**Symptoms:**
- flash_attn import fails
- Model initialization errors

**Solution:**
Install flash-attn from the pre-built wheel:

```bash
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
```

**Note:** Do not use `pip install flash-attn` as it may not be compatible with CUDA 12.8.

---

## CUDA and GPU Issues

### Issue: `CUDA not available` or `torch.cuda.is_available()` returns `False`

**Symptoms:**
- PyTorch doesn't detect GPU
- CUDA unavailable errors

**Diagnosis:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

**Solutions:**

1. **CUDA not installed:** Install CUDA 12.8 from NVIDIA website

2. **Wrong PyTorch version:** Reinstall PyTorch with CUDA support:
   ```bash
   pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
   ```

3. **Driver issues:** Update NVIDIA drivers:
   ```bash
   # Check driver version
   nvidia-smi
   # Update if needed (method varies by OS)
   ```

---

### Issue: CUDA version mismatch

**Symptoms:**
- Errors mentioning CUDA version incompatibility
- `torch.version.cuda` doesn't match system CUDA

**Solution:**
Ensure you're using CUDA 12.8 compatible packages:

```bash
# Verify system CUDA
nvcc --version  # Should show 12.8

# Verify PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"  # Should show 12.8 or compatible
```

If mismatch, reinstall with correct CUDA version:
```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

---

### Issue: Out of memory (OOM) errors

**Symptoms:**
- `CUDA out of memory` errors
- Process killed during inference

**Solutions:**

1. **Reduce batch size:** In `config.py`, lower `MAX_CONCURRENCY`:
   ```python
   MAX_CONCURRENCY = 50  # Reduce from 100
   ```

2. **Reduce max crops:** In `config.py`:
   ```python
   MAX_CROPS = 4  # Reduce from 6
   ```

3. **Adjust GPU memory utilization:** In the engine args:
   ```python
   gpu_memory_utilization=0.6  # Reduce from 0.75
   ```

4. **Use smaller image sizes:** In `config.py`:
   ```python
   BASE_SIZE = 640  # Reduce from 1024
   IMAGE_SIZE = 512  # Reduce from 640
   ```

---

## Runtime Errors

### Issue: `RuntimeError: CUDA error: invalid device ordinal`

**Symptoms:**
- Error when trying to use GPU
- Invalid device errors

**Solution:**
Check available GPUs and set correct device:

```bash
# Check available GPUs
nvidia-smi

# Set correct GPU in code
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

In Python:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

---

### Issue: Model download fails or is very slow

**Symptoms:**
- Timeout errors when downloading model
- Very slow download speeds

**Solutions:**

1. **Use HF_HUB_ENABLE_HF_TRANSFER:**
   ```bash
   pip install hf_transfer
   export HF_HUB_ENABLE_HF_TRANSFER=1
   ```

2. **Use mirror (China users):**
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **Download manually:**
   ```bash
   git lfs install
   git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
   ```

---

### Issue: Inference is very slow

**Symptoms:**
- Much slower than expected
- Low GPU utilization

**Solutions:**

1. **Check GPU usage:**
   ```bash
   nvidia-smi -l 1  # Monitor GPU usage
   ```

2. **Increase concurrency:** In `config.py`:
   ```python
   MAX_CONCURRENCY = 150  # Increase if you have memory
   ```

3. **Enable tensor parallelism:** For multi-GPU:
   ```python
   tensor_parallel_size=2  # Use 2 GPUs
   ```

4. **Disable eager mode:**
   ```python
   enforce_eager=False  # Should be False for better performance
   ```

---

## Performance Issues

### Issue: High memory usage during preprocessing

**Symptoms:**
- System memory (RAM) fills up
- Process killed by OOM killer

**Solution:**
Reduce number of workers in `config.py`:
```python
NUM_WORKERS = 32  # Reduce from 64
```

---

### Issue: Slow image preprocessing

**Symptoms:**
- Long wait before inference starts
- CPU bottleneck

**Solution:**
Increase workers if you have CPU cores available:
```python
NUM_WORKERS = 128  # Increase if you have cores
```

---

## General Tips

### Verify Installation

Always run the verification script after installation:
```bash
python scripts/verify_installation.py
```

### Clean Installation

If you have persistent issues, try a clean installation:

```bash
# Deactivate and remove old environment
conda deactivate
conda env remove -n deepseek-ocr

# Create fresh environment
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# Run installation script
bash scripts/install_cuda128.sh
```

### Check Logs

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Environment Variables

Useful environment variables:
```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1  # For debugging

# vLLM settings
export VLLM_USE_V1=0

# HuggingFace settings
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/path/to/cache

# Triton (for CUDA 11.8)
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas
```

### Get Help

If you're still stuck:

1. **Check existing issues:** Search [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
2. **Provide details:** When opening an issue, include:
   - Python version (`python --version`)
   - CUDA version (`nvcc --version`)
   - GPU model (`nvidia-smi`)
   - Full error traceback
   - Output of `python scripts/verify_installation.py`
3. **Join Discord:** [DeepSeek AI Discord](https://discord.gg/Tc7c45Zzu5)

---

## Quick Reference: Installation Order

For CUDA 12.8 / RTX 5090, the correct order is:

1. Install xformers (gets torch nightly)
2. Download pre-built wheels
3. Install flash-attn wheel
4. Install vLLM wheel with `--no-build-isolation --no-deps`
5. Install core dependencies
6. Install additional dependencies iteratively
7. Install torchvision
8. **Re-install xformers** (critical!)
9. Install project requirements

**Never skip step 8!**

---

## Additional Resources

- [Installation Guide for CUDA 12.8](INSTALL_CUDA_12.8.md)
- [Official Documentation](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
