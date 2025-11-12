# Resolution for GitHub Issue #240

## Issue Summary

**Title:** [Solved] How I Got DeepSeek-OCR vLLM Working on an RTX 5090

**Problem:** Users with NVIDIA RTX 5090 GPUs running CUDA 12.8 were unable to install DeepSeek-OCR with vLLM using the standard installation instructions (which are for CUDA 11.8).

**Root Cause:** 
- Complex dependency conflicts between PyTorch nightly, torchvision, xformers, and vLLM
- Installing torchvision breaks the PyTorch nightly installation, causing C++ errors in vLLM
- Pre-built wheels are required; `pip install vllm` directly does not work
- Installation order is critical

## Solution Implemented

This resolution provides comprehensive documentation and automation tools to help users successfully install DeepSeek-OCR with vLLM on RTX 5090 / CUDA 12.8 systems.

### Files Created

1. **docs/INSTALL_CUDA_12.8.md**
   - Comprehensive installation guide with detailed explanations
   - Step-by-step instructions following the exact process from Issue #240
   - Troubleshooting section for common issues
   - Package version reference table
   - Links to resources and credits

2. **scripts/install_cuda128_rtx5090.sh**
   - Fully automated installation script
   - Handles all dependency installations in the correct order
   - Automatically resolves missing dependencies iteratively
   - Includes progress indicators and colored output
   - Performs verification after installation
   - Error handling and logging

3. **scripts/verify_installation.py**
   - Python script to verify all required packages are installed
   - Checks package versions and compatibility
   - Tests CUDA availability and operations
   - Provides detailed output with color-coded status
   - Suggests next steps based on results

4. **docs/QUICK_START_CUDA_12.8.md**
   - Condensed quick reference guide
   - Both automated and manual installation options
   - Common issues and quick solutions
   - Quick test instructions

5. **README.md** (Updated)
   - Added prominent section for CUDA 12.8 installation
   - Links to detailed installation guide
   - Note specifically for RTX 5090 users
   - Reference to Issue #240

## Key Installation Steps

The solution documents and automates the following critical steps:

1. **Install xformers nightly** (pulls in PyTorch nightly)
   ```bash
   pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
   ```

2. **Download and install pre-built wheels**
   ```bash
   wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
   wget https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl
   pip install ./flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
   pip install ./vllm-0.8.5+cu128-cp312-cp312-linux_x86_64.whl --no-build-isolation --no-deps
   ```

3. **Install initial dependencies**
   - pydantic, transformers, cachetools, cloudpickle, psutil, zmq, msgspec, blake3, etc.

4. **Iterative dependency resolution**
   - Run `python -c "import vllm; print(vllm.__version__)"` repeatedly
   - Install any missing modules reported
   - Continue until successful

5. **Install torchvision** (breaks environment)
   ```bash
   pip install torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
   ```

6. **CRITICAL: Re-install xformers** (fixes environment)
   ```bash
   pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
   ```

7. **Install final dependencies**
   - hf_transfer, prometheus_client, sentencepiece, protobuf

8. **Install DeepSeek-OCR requirements**
   ```bash
   pip install -r requirements.txt
   ```

## Why This Works

### The torchvision/xformers Conflict

The most critical insight from Issue #240 is understanding the torchvision/xformers conflict:

1. **Initial state:** xformers installation brings in PyTorch nightly with correct C++ bindings for vLLM
2. **Problem:** Installing torchvision uninstalls PyTorch nightly and replaces it with an incompatible version
3. **Solution:** Re-installing xformers forces pip to reinstall the correct PyTorch nightly version

This is why the installation order is critical and why xformers must be installed twice.

### Pre-built Wheels

Using pre-built wheels is essential because:
- Building vLLM from source with CUDA 12.8 is complex and error-prone
- The pre-built wheels are compiled with the correct CUDA version and dependencies
- The `--no-build-isolation --no-deps` flags prevent pip from trying to rebuild or modify dependencies

## Usage

### For End Users

**Automated Installation (Recommended):**
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
conda create -n deepseek-ocr-cuda128 python=3.12 -y
conda activate deepseek-ocr-cuda128
bash scripts/install_cuda128_rtx5090.sh
```

**Manual Installation:**
Follow the detailed steps in `docs/INSTALL_CUDA_12.8.md`

**Verification:**
```bash
python scripts/verify_installation.py
```

### For Developers

The installation script and verification script can be used as templates for:
- Creating installation scripts for other CUDA versions
- Debugging installation issues
- Understanding the dependency chain

## Testing

All scripts have been tested for:
- ✅ Syntax validity (Python and Bash)
- ✅ Proper error handling
- ✅ Clear output and progress indicators
- ✅ Comprehensive verification checks
- ✅ Documentation completeness

## Benefits

1. **Reduced Installation Time:** Automated script handles all steps, reducing manual effort
2. **Fewer Errors:** Correct installation order prevents common pitfalls
3. **Better Debugging:** Verification script provides detailed diagnostics
4. **Community Knowledge:** Documents community-discovered solutions
5. **Maintainability:** Clear documentation makes it easier to update for future versions

## Credits

- **Original Issue:** GitHub Issue #240
- **Community Contributors:** Thanks to @ghcdmm for pre-built wheels and Issue #238 contributors
- **Documentation:** Based on detailed steps shared by the community

## Future Improvements

Potential enhancements for future versions:

1. Support for additional CUDA versions (12.6, 12.7, etc.)
2. Docker container with pre-configured environment
3. Automated testing in CI/CD pipeline
4. Support for other GPU models (RTX 4090, A100, etc.)
5. Integration with official vLLM releases when CUDA 12.8 support is stable

## Related Issues

- Issue #238: Initial troubleshooting for vLLM installation
- Issue #240: Successful installation process for RTX 5090 / CUDA 12.8

## License

This solution is part of the DeepSeek-OCR project and follows the same license terms.
