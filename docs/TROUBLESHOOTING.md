# DeepSeek-OCR Troubleshooting Guide

This guide covers common issues and solutions when installing and running DeepSeek-OCR with vLLM, particularly for CUDA 12.8 and RTX 5090 setups.

## Table of Contents

- [Installation Issues](#installation-issues)
  - [ModuleNotFoundError](#modulenotfounderror)
  - [vllm C++ Errors](#vllm-c-errors)
  - [Dependency Conflicts](#dependency-conflicts)
  - [Wheel Installation Failures](#wheel-installation-failures)
- [Runtime Issues](#runtime-issues)
  - [CUDA Out of Memory](#cuda-out-of-memory)
  - [Model Loading Errors](#model-loading-errors)
  - [Import Errors After Installation](#import-errors-after-installation)
- [Performance Issues](#performance-issues)
- [Getting Help](#getting-help)

## Installation Issues

### ModuleNotFoundError

**Problem:** After installing vllm, you get `ModuleNotFoundError: No module named 'xxx'` when trying to import vllm.

**Cause:** The vllm wheel installed with `--no-deps` flag doesn't include all dependencies.

**Solution:**

1. **Identify the missing module:**
   ```bash
   python -c "import vllm; print(vllm.__version__)"
   ```
   The error message will show which module is missing.

2. **Install the missing module:**
   ```bash
   pip install <missing_module_name>
   ```

3. **Common missing modules and their package names:**
   - `zmq` → Install `pyzmq`
   - `PIL` → Install `Pillow`
   - `cv2` → Install `opencv-python`
   - `yaml` → Install `PyYAML`

4. **Repeat the verification:**
   Keep running the verification command and installing missing modules until it succeeds.

**Automated Solution:**

Use the provided installation script which automatically detects and installs missing dependencies:
```bash
bash scripts/install_cuda128_rtx5090.sh
```

### vllm C++ Errors

**Problem:** You get C++ compilation or runtime errors when importing vllm, such as:
```
ImportError: /path/to/vllm/_C.so: undefined symbol: ...
```

**Cause:** This typically happens when torch nightly is replaced by a stable version, causing ABI incompatibility.

**Solution:**

1. **Re-install xformers nightly:**
   ```bash
   pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
   ```
   This will pull in the correct torch nightly version.

2. **Verify torch version:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   You should see a nightly version like `2.7.0.dev20251104+cu128`

3. **If the issue persists, reinstall vllm:**
   ```bash
   pip uninstall vllm -y
   pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
   ```

### Dependency Conflicts

**Problem:** Installing one package breaks another, or you get version conflict errors.

**Common Scenario: torchvision breaks torch nightly**

**Cause:** Installing torchvision from PyPI can downgrade torch nightly to a stable version.

**Solution:**

1. **Always install torchvision from nightly:**
   ```bash
   pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

2. **After installing torchvision, immediately re-install xformers:**
   ```bash
   pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
   ```

3. **Verify versions:**
   ```bash
   python -c "import torch, torchvision; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}')"
   ```

**General Dependency Conflict Resolution:**

1. Create a fresh conda environment:
   ```bash
   conda create -n deepseek-ocr-fresh python=3.12 -y
   conda activate deepseek-ocr-fresh
   ```

2. Follow the installation guide strictly in order: [INSTALL_CUDA_12.8.md](./INSTALL_CUDA_12.8.md)

### Wheel Installation Failures

**Problem:** Failed to download or install pre-built wheels.

**Solution:**

1. **Check internet connection and retry:**
   ```bash
   wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
   wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
   ```

2. **If download fails, try alternative sources:**
   - Check the original issue for updated links: [Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240)
   - Check [Issue #238](https://github.com/deepseek-ai/DeepSeek-OCR/issues/238) for alternative wheels

3. **Verify wheel integrity:**
   ```bash
   ls -lh *.whl
   ```
   Ensure the files are not corrupted (should be several hundred MB each).

4. **If installation fails with permission errors:**
   ```bash
   pip install --user ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
   pip install --user ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
   ```

## Runtime Issues

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory` when running inference.

**Solution:**

1. **Reduce batch size in config.py:**
   ```python
   MAX_CONCURRENCY = 50  # Reduce from 100
   ```

2. **Reduce max crops:**
   ```python
   MAX_CROPS = 4  # Reduce from 6
   ```

3. **Use smaller resolution mode:**
   ```python
   BASE_SIZE = 640  # Instead of 1024
   IMAGE_SIZE = 640
   CROP_MODE = False
   ```

4. **Clear CUDA cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

5. **Monitor GPU memory:**
   ```bash
   nvidia-smi -l 1
   ```

### Model Loading Errors

**Problem:** Errors when loading the DeepSeek-OCR model.

**Solution:**

1. **Verify model path in config.py:**
   ```python
   MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # For HuggingFace
   # OR
   MODEL_PATH = '/path/to/local/model'  # For local model
   ```

2. **Check HuggingFace authentication (if using private models):**
   ```bash
   huggingface-cli login
   ```

3. **Verify model files are complete:**
   ```bash
   ls -lh ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/
   ```

4. **Re-download model if corrupted:**
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/
   # Then run your script again to re-download
   ```

### Import Errors After Installation

**Problem:** `ImportError` or `AttributeError` when running DeepSeek-OCR scripts.

**Solution:**

1. **Verify all packages are installed:**
   ```bash
   pip list | grep -E "vllm|torch|transformers|flash-attn|xformers"
   ```

2. **Check for conflicting installations:**
   ```bash
   pip list | grep -E "torch|vllm"
   ```
   Ensure you don't have multiple versions installed.

3. **Reinstall requirements.txt:**
   ```bash
   cd DeepSeek-OCR
   pip install -r requirements.txt --force-reinstall
   ```

4. **Verify Python environment:**
   ```bash
   which python
   python --version
   ```
   Ensure you're using the correct conda environment.

## Performance Issues

### Slow Inference Speed

**Problem:** Inference is slower than expected.

**Solution:**

1. **Increase concurrency (if you have enough GPU memory):**
   ```python
   MAX_CONCURRENCY = 150  # Increase from 100
   ```

2. **Increase number of workers:**
   ```python
   NUM_WORKERS = 128  # Increase from 64
   ```

3. **Use appropriate resolution mode:**
   - For simple documents: Use Tiny or Small mode
   - For complex documents: Use Base or Gundam mode

4. **Enable tensor parallelism (for multi-GPU):**
   ```python
   llm = LLM(
       model="deepseek-ai/DeepSeek-OCR",
       tensor_parallel_size=2,  # For 2 GPUs
       ...
   )
   ```

5. **Monitor GPU utilization:**
   ```bash
   nvidia-smi dmon -s u
   ```
   If GPU utilization is low, increase concurrency.

### High Memory Usage

**Problem:** System runs out of RAM (not GPU memory).

**Solution:**

1. **Reduce number of workers:**
   ```python
   NUM_WORKERS = 32  # Reduce from 64
   ```

2. **Process files in smaller batches:**
   Instead of processing all files at once, process them in chunks.

3. **Monitor memory usage:**
   ```bash
   htop
   ```

## Getting Help

If you're still experiencing issues:

1. **Check existing issues:**
   - [Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240) - CUDA 12.8 installation
   - [Issue #238](https://github.com/deepseek-ai/DeepSeek-OCR/issues/238) - Related installation issues
   - [All Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)

2. **Gather diagnostic information:**
   ```bash
   # System info
   nvidia-smi
   python --version
   pip list | grep -E "vllm|torch|transformers|flash-attn|xformers"
   
   # CUDA info
   nvcc --version
   echo $CUDA_HOME
   ```

3. **Create a new issue:**
   - Include the diagnostic information above
   - Describe the exact error message
   - Include steps to reproduce
   - Mention your hardware (GPU model, CUDA version)

4. **Community resources:**
   - [DeepSeek Discord](https://discord.gg/Tc7c45Zzu5)
   - [DeepSeek Twitter](https://twitter.com/deepseek_ai)

## Quick Reference: Common Commands

```bash
# Verify installation
python -c "import vllm; print(vllm.__version__)"

# Check versions
python -c "import torch, torchvision, xformers, flash_attn; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'xformers: {xformers.__version__}'); print(f'flash-attn: {flash_attn.__version__}')"

# Monitor GPU
nvidia-smi -l 1

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

# Reinstall xformers (fixes most issues)
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# Fresh start
conda deactivate
conda env remove -n deepseek-ocr
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr
bash scripts/install_cuda128_rtx5090.sh
```

## Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12 | Required for pre-built wheels |
| CUDA | 12.8 | For RTX 5090 |
| vllm | 0.8.5+cu128 | Pre-built wheel |
| flash-attn | 2.8.3 | Pre-built wheel |
| xformers | 0.0.33.dev20251104+cu128 | Nightly build |
| torch | 2.7.0.dev (nightly) | Installed with xformers |
| torchvision | 0.22.0.dev (nightly) | Must be nightly |
| transformers | >=4.46.3 | From requirements.txt |

---

**Last Updated:** December 2025  
**Based on:** [Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240)
