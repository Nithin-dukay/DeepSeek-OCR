# Fix for GitHub Issue #110: vLLM Engine Core Crash

## Problem Description

**Error Message:**
```
ERROR 10-23 18:21:12 [core_client.py:597] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

This error occurs when using vLLM nightly build with DeepSeek-OCR due to a version compatibility issue between vLLM v1 (nightly) and the v0 architecture that DeepSeek-OCR is configured to use.

## Root Cause

1. **vLLM Nightly Build**: Uses the new v1 architecture by default
2. **DeepSeek-OCR Configuration**: Explicitly forces v0 mode with `os.environ['VLLM_USE_V1'] = '0'`
3. **Conflict**: The v1 engine cannot properly run in forced v0 mode, causing the engine core process to crash

## Solutions

### Solution 1: Use Stable vLLM Version (RECOMMENDED)

Instead of using the nightly build, use the stable vLLM v0.8.5 as documented in the main README:

```bash
# Create virtual environment
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Download and install vLLM 0.8.5 wheel
# Download from: https://github.com/vllm-project/vllm/releases/tag/v0.8.5
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install other requirements
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

### Solution 2: Remove v0 Enforcement for Nightly Build

If you must use the nightly build, modify the run scripts to allow v1 architecture:

**For `run_dpsk_ocr_image.py`:**
```python
# Comment out or remove this line:
# os.environ['VLLM_USE_V1'] = '0'

# Or change to:
# os.environ['VLLM_USE_V1'] = '1'
```

**Apply the same change to:**
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py`
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py`

**Note:** This may require additional code changes as the v1 API differs from v0.

### Solution 3: Use Specific vLLM Nightly Version

Use a specific nightly version that's known to work:

```bash
uv venv
source .venv/bin/activate

# Install a specific nightly version (adjust date as needed)
uv pip install vllm==0.6.5.dev --extra-index-url https://wheels.vllm.ai/nightly

# Or use the latest stable pre-release
uv pip install vllm==0.6.4.post1
```

### Solution 4: Update Code for vLLM v1 Compatibility

For full v1 support, the codebase needs updates. Here's a compatibility wrapper approach:

```python
import os
from packaging import version
import vllm

# Detect vLLM version and set appropriate mode
vllm_version = version.parse(vllm.__version__)

if vllm_version >= version.parse("0.6.0"):
    # Use v1 for newer versions
    os.environ['VLLM_USE_V1'] = '1'
    print(f"Using vLLM v1 architecture (version: {vllm.__version__})")
else:
    # Use v0 for older versions
    os.environ['VLLM_USE_V1'] = '0'
    print(f"Using vLLM v0 architecture (version: {vllm.__version__})")
```

## Recommended Action

**Use Solution 1** - Install the stable vLLM v0.8.5 version as documented in the main README. This is the tested and supported configuration.

## Verification Steps

After applying the fix:

1. **Check vLLM version:**
   ```bash
   python -c "import vllm; print(vllm.__version__)"
   ```

2. **Test with a simple image:**
   ```bash
   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
   # Update config.py with your image path
   python run_dpsk_ocr_image.py
   ```

3. **Monitor for errors:**
   - No "Engine core proc died" errors
   - Successful model loading
   - Proper inference output

## Additional Troubleshooting

### If you still encounter issues:

1. **Check CUDA compatibility:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Verify GPU memory:**
   ```bash
   nvidia-smi
   ```

3. **Check for conflicting environment variables:**
   ```bash
   env | grep VLLM
   ```

4. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## Related Issues

- vLLM v1 migration guide: https://docs.vllm.ai/en/latest/
- DeepSeek-OCR official documentation: README.md

## Contributing

If you find additional solutions or improvements, please contribute to the repository or update this document.
