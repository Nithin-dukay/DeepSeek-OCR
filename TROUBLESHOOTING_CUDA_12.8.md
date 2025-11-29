# Troubleshooting Guide - CUDA 12.8 Installation

This guide addresses common issues when installing DeepSeek-OCR with vLLM on CUDA 12.8 systems.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Import Errors](#import-errors)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [Environment Issues](#environment-issues)

## Installation Issues

### Issue: "pip install vllm" fails or installs wrong version

**Symptoms:**
- Installation hangs or fails
- Wrong CUDA version installed
- Compilation errors

**Solution:**
Do NOT use `pip install vllm` directly. Use the pre-built wheel:

```bash
wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
```

### Issue: ModuleNotFoundError after installing vLLM

**Symptoms:**
```
ModuleNotFoundError: No module named 'pydantic'
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
This is expected! The vLLM wheel is installed without dependencies. Install them iteratively:

```bash
# Check what's missing
python -c "import vllm; print(vllm.__version__)"

# Install the missing package
pip install <missing_package>

# Repeat until successful
```

**Common missing packages:**
```bash
pip install pydantic transformers cachetools cloudpickle psutil zmq msgspec blake3 hf_transfer prometheus_client
```

### Issue: vLLM C++ error after installing torchvision

**Symptoms:**
```
ImportError: /path/to/vllm.so: undefined symbol: ...
RuntimeError: CUDA error: ...
```

**Solution:**
Installing torchvision breaks the torch nightly. Re-install xformers to fix:

```bash
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: "No module named 'torchvision'"

**Symptoms:**
```
ModuleNotFoundError: No module named 'torchvision'
```

**Solution:**
Install torchvision nightly, then immediately re-install xformers:

```bash
pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: CUDA version mismatch

**Symptoms:**
```
RuntimeError: CUDA version mismatch
The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (12.8)
```

**Solution:**
Ensure you're using the CUDA 12.8 index URL:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: Python version incompatibility

**Symptoms:**
```
ERROR: vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl is not a supported wheel on this platform.
```

**Solution:**
The pre-built wheel requires Python 3.12. Create a new environment:

```bash
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr
```

## Import Errors

### Issue: "No module named 'zmq'"

**Solution:**
Install pyzmq (not zmq):

```bash
pip install pyzmq
```

### Issue: "No module named 'fitz'"

**Solution:**
Install PyMuPDF:

```bash
pip install PyMuPDF
```

### Issue: "No module named 'PIL'"

**Solution:**
Install Pillow:

```bash
pip install Pillow
```

### Issue: Cannot import DeepseekOCRForCausalLM

**Symptoms:**
```
ModuleNotFoundError: No module named 'deepseek_ocr'
```

**Solution:**
Ensure you're running from the correct directory:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

## Runtime Errors

### Issue: Out of memory (OOM) errors

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. Reduce GPU memory utilization:
```python
llm = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=0.75,  # Reduce from 0.9
    # ...
)
```

2. Reduce max concurrency in `config.py`:
```python
MAX_CONCURRENCY = 50  # Reduce from 100
```

3. Reduce max crops for large images:
```python
MAX_CROPS = 4  # Reduce from 6
```

4. Use smaller image sizes:
```python
BASE_SIZE = 640  # Reduce from 1024
IMAGE_SIZE = 512  # Reduce from 640
```

### Issue: Slow inference speed

**Symptoms:**
- Very slow token generation
- Low GPU utilization

**Solutions:**

1. Increase concurrency (if you have memory):
```python
MAX_CONCURRENCY = 150  # Increase from 100
```

2. Increase preprocessing workers:
```python
NUM_WORKERS = 128  # Increase from 64
```

3. Enable eager mode if compilation fails:
```python
llm = LLM(
    model=MODEL_PATH,
    enforce_eager=True,  # Disable graph compilation
    # ...
)
```

4. Check GPU utilization:
```bash
nvidia-smi -l 1
```

### Issue: "CUDA error: device-side assert triggered"

**Symptoms:**
```
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions:**

1. Check input data validity (image format, size)
2. Reduce batch size or concurrency
3. Enable CUDA error checking for more details:
```bash
export CUDA_LAUNCH_BLOCKING=1
python run_dpsk_ocr_image.py
```

### Issue: Repeated output or no EOS token

**Symptoms:**
- Model generates repeated text
- Output doesn't stop

**Solutions:**

1. Check ngram logits processor settings:
```python
logits_processors = [NoRepeatNGramLogitsProcessor(
    ngram_size=30,  # Adjust this
    window_size=90,  # And this
    whitelist_token_ids={128821, 128822}
)]
```

2. Adjust sampling parameters:
```python
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
)
```

## Performance Issues

### Issue: Low throughput with PDF processing

**Solutions:**

1. Increase concurrency:
```python
MAX_CONCURRENCY = 200  # For high-memory GPUs
```

2. Increase worker threads:
```python
NUM_WORKERS = 128
```

3. Use ThreadPoolExecutor for preprocessing (already implemented in the code)

### Issue: High memory usage during preprocessing

**Solutions:**

1. Reduce number of workers:
```python
NUM_WORKERS = 32  # Reduce from 64
```

2. Process in smaller batches
3. Clear cache between batches:
```python
torch.cuda.empty_cache()
```

## Environment Issues

### Issue: Conda environment conflicts

**Solution:**
Create a fresh environment:

```bash
conda deactivate
conda env remove -n deepseek-ocr
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr
./install_cuda128.sh
```

### Issue: Multiple CUDA versions installed

**Solution:**
Ensure CUDA 12.8 is in your PATH:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Issue: Permission denied when running scripts

**Solution:**
Make scripts executable:

```bash
chmod +x install_cuda128.sh
chmod +x verify_installation.py
```

## Verification Commands

Use these commands to diagnose issues:

```bash
# Check Python version
python --version

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check vLLM
python -c "import vllm; print(vllm.__version__)"

# Check all packages
python verify_installation.py

# Check GPU
nvidia-smi

# Check CUDA version
nvcc --version
```

## Getting Help

If your issue isn't covered here:

1. **Run verification script:**
   ```bash
   python verify_installation.py
   ```

2. **Check logs:**
   - Look for error messages in the output
   - Enable verbose logging if available

3. **Gather system information:**
   ```bash
   python --version
   nvcc --version
   nvidia-smi
   pip list | grep -E "torch|vllm|xformers|flash"
   ```

4. **Search existing issues:**
   - [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
   - Look for similar error messages

5. **Create a new issue:**
   - Include error messages
   - Include system information
   - Include steps to reproduce

## Additional Resources

- [Installation Guide](INSTALL_CUDA_12.8.md)
- [Quick Start Guide](QUICKSTART_CUDA_12.8.md)
- [Original Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240)
- [vLLM Documentation](https://docs.vllm.ai/)

---

**Last Updated**: November 2025
