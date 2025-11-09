# DeepSeek-OCR Troubleshooting Guide

This guide helps you resolve common issues when running DeepSeek-OCR.

## Table of Contents

1. [vLLM Engine Core Crash (Issue #110)](#vllm-engine-core-crash-issue-110)
2. [CUDA and GPU Issues](#cuda-and-gpu-issues)
3. [Memory Issues](#memory-issues)
4. [Installation Issues](#installation-issues)
5. [Model Loading Issues](#model-loading-issues)

---

## vLLM Engine Core Crash (Issue #110)

### Symptom

```
ERROR 10-23 18:21:12 [core_client.py:597] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

### Cause

This error occurs when there's a version mismatch between vLLM nightly builds (v1 architecture) and the DeepSeek-OCR code (configured for v0 architecture).

### Solutions

#### Solution 1: Use Stable vLLM (RECOMMENDED)

```bash
# Create environment
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# Install PyTorch with CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Download and install vLLM 0.8.5
# Download from: https://github.com/vllm-project/vllm/releases/tag/v0.8.5
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install other requirements
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

#### Solution 2: Use Fixed Scripts

Use the provided fixed scripts with automatic version detection:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Use the fixed version
python run_dpsk_ocr_image_fixed.py
```

#### Solution 3: Manual Fix

Edit the run scripts and comment out the v0 enforcement:

```python
# In run_dpsk_ocr_image.py, run_dpsk_ocr_pdf.py, run_dpsk_ocr_eval_batch.py
# Comment out this line:
# os.environ['VLLM_USE_V1'] = '0'
```

### Verification

```bash
# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# Test compatibility
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python vllm_version_compat.py
```

---

## CUDA and GPU Issues

### Issue: CUDA not available

**Symptom:**
```python
torch.cuda.is_available() returns False
```

**Solutions:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

2. **Verify CUDA installation:**
   ```bash
   nvcc --version
   ```

3. **Reinstall PyTorch with correct CUDA version:**
   ```bash
   # For CUDA 11.8
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121
   ```

### Issue: TRITON_PTXAS_PATH error

**Symptom:**
```
TRITON_PTXAS_PATH not found
```

**Solution:**

The scripts automatically set this for CUDA 11.8. If you have a different CUDA version:

```python
import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-XX.X/bin/ptxas"  # Replace XX.X with your version
```

---

## Memory Issues

### Issue: Out of Memory (OOM)

**Symptom:**
```
CUDA out of memory
RuntimeError: CUDA error: out of memory
```

**Solutions:**

1. **Reduce GPU memory utilization in config.py:**
   ```python
   # In LLM initialization
   gpu_memory_utilization=0.75  # Reduce from 0.9 to 0.75 or lower
   ```

2. **Reduce max concurrency:**
   ```python
   # In config.py
   MAX_CONCURRENCY = 50  # Reduce from 100
   ```

3. **Reduce max crops:**
   ```python
   # In config.py
   MAX_CROPS = 4  # Reduce from 6
   ```

4. **Use smaller image sizes:**
   ```python
   # In config.py
   BASE_SIZE = 640  # Instead of 1024
   IMAGE_SIZE = 512  # Instead of 640
   ```

5. **Enable swap space:**
   ```python
   # In LLM initialization
   swap_space=4  # Enable 4GB swap
   ```

### Issue: CPU Memory Issues

**Symptom:**
```
MemoryError or system freezing
```

**Solutions:**

1. **Reduce number of workers:**
   ```python
   # In config.py
   NUM_WORKERS = 32  # Reduce from 64
   ```

2. **Process images in smaller batches:**
   ```python
   # Process PDFs page by page instead of all at once
   ```

---

## Installation Issues

### Issue: Flash Attention installation fails

**Symptom:**
```
ERROR: Failed building wheel for flash-attn
```

**Solutions:**

1. **Install build dependencies:**
   ```bash
   pip install ninja packaging wheel
   ```

2. **Install with no build isolation:**
   ```bash
   pip install flash-attn==2.7.3 --no-build-isolation
   ```

3. **Use pre-built wheels (if available):**
   ```bash
   pip install flash-attn==2.7.3 --find-links https://github.com/Dao-AILab/flash-attention/releases
   ```

### Issue: Transformers version conflict

**Symptom:**
```
vllm 0.8.5+cu118 requires transformers>=4.51.1
```

**Solution:**

This warning can be ignored if you're using the vLLM and transformers codes in the same environment. The project is tested with transformers==4.46.3.

---

## Model Loading Issues

### Issue: Model not found

**Symptom:**
```
OSError: deepseek-ai/DeepSeek-OCR is not a local folder
```

**Solutions:**

1. **Check MODEL_PATH in config.py:**
   ```python
   MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # For HuggingFace download
   # OR
   MODEL_PATH = '/path/to/local/model'  # For local model
   ```

2. **Ensure internet connection for HuggingFace download**

3. **Download model manually:**
   ```bash
   git lfs install
   git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
   ```

### Issue: Tokenizer errors

**Symptom:**
```
KeyError: '<image>'
```

**Solution:**

Ensure the tokenizer is properly initialized in config.py:

```python
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
```

---

## Performance Issues

### Issue: Slow inference

**Solutions:**

1. **Enable compilation (experimental):**
   ```python
   # In deepseek_ocr.py
   self.sam_model = torch.compile(self.sam_model, mode="reduce-overhead")
   self.vision_model = torch.compile(self.vision_model, mode="reduce-overhead")
   ```

2. **Increase concurrency:**
   ```python
   # In config.py
   MAX_CONCURRENCY = 150  # If you have enough GPU memory
   ```

3. **Use enforce_eager=False:**
   ```python
   # In LLM initialization
   enforce_eager=False  # Enable CUDA graphs
   ```

4. **Increase number of workers:**
   ```python
   # In config.py
   NUM_WORKERS = 128  # If you have enough CPU cores
   ```

---

## Getting Help

If you encounter issues not covered here:

1. **Check existing GitHub issues:** https://github.com/deepseek-ai/DeepSeek-OCR/issues
2. **Create a new issue with:**
   - Error message and full stack trace
   - Your environment details (OS, CUDA version, GPU model)
   - vLLM version: `python -c "import vllm; print(vllm.__version__)"`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Steps to reproduce

3. **Join the Discord community:** https://discord.gg/Tc7c45Zzu5

---

## Quick Diagnostic Script

Run this script to check your environment:

```python
import sys
import torch

print("=" * 60)
print("DeepSeek-OCR Environment Diagnostic")
print("=" * 60)

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

try:
    import vllm
    print(f"vLLM version: {vllm.__version__}")
except ImportError:
    print("vLLM: NOT INSTALLED")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except ImportError:
    print("Transformers: NOT INSTALLED")

try:
    import flash_attn
    print(f"Flash Attention: INSTALLED")
except ImportError:
    print("Flash Attention: NOT INSTALLED")

print("=" * 60)
```

Save this as `check_environment.py` and run it to diagnose issues.
