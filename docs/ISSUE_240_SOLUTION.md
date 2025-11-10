# Solution for GitHub Issue #240: DeepSeek-OCR vLLM on RTX 5090 / CUDA 12.8

## Issue Summary

GitHub Issue #240 documented a successful installation of DeepSeek-OCR with vLLM on an NVIDIA RTX 5090 with CUDA 12.8. The installation process was complex due to fragile nightly dependencies and required a very specific installation order.

## Solution Overview

This solution provides comprehensive documentation and automation tools to help users successfully install DeepSeek-OCR with vLLM on CUDA 12.8 / RTX 5090 systems.

## What Was Implemented

### 1. Documentation

#### **docs/INSTALL_CUDA_12.8.md**
Comprehensive installation guide covering:
- Prerequisites and system requirements
- Step-by-step manual installation instructions
- Critical torchvision fix (the most important step)
- Verification procedures
- Common issues and solutions
- Testing instructions

#### **docs/TROUBLESHOOTING.md**
Detailed troubleshooting guide with:
- Installation issues and solutions
- Import error fixes
- CUDA and GPU problem resolution
- Runtime error solutions
- Performance optimization tips
- Quick reference for installation order

#### **docs/ISSUE_240_SOLUTION.md** (this file)
Summary document explaining the solution implementation

### 2. Automation Scripts

#### **scripts/install_cuda128.sh**
Automated installation script that:
- Checks Python and CUDA versions
- Downloads pre-built wheels automatically
- Installs dependencies in the correct order
- Handles the critical torchvision/xformers fix
- Performs iterative dependency installation
- Runs verification checks
- Provides colored output and progress indicators

**Usage:**
```bash
bash scripts/install_cuda128.sh
```

#### **scripts/verify_installation.py**
Comprehensive verification script that:
- Checks Python version compatibility
- Verifies CUDA installation and GPU detection
- Tests PyTorch CUDA support
- Validates vLLM installation
- Checks flash-attn and xformers
- Verifies all critical dependencies
- Provides detailed diagnostic information
- Color-coded output for easy reading

**Usage:**
```bash
python scripts/verify_installation.py
```

### 3. Requirements File

#### **requirements_cuda128.txt**
Documented list of all dependencies for CUDA 12.8 setup with:
- Clear warnings about installation order
- Comments explaining each dependency category
- References to the installation guide

### 4. README Updates

Updated the main README.md to:
- Add prominent CUDA 12.8 installation section
- Link to specialized installation guide
- Reference automated installation script
- Add troubleshooting section
- Maintain backward compatibility with CUDA 11.8 instructions

## Key Technical Details

### The Critical Installation Order

The most important aspect of this solution is the **installation order**:

1. **Install xformers first** - This pulls in the correct torch nightly
2. **Download pre-built wheels** - For flash-attn and vLLM
3. **Install flash-attn** - From the pre-built wheel
4. **Install vLLM** - With `--no-build-isolation --no-deps` flags
5. **Install core dependencies** - Basic packages vLLM needs
6. **Iterative dependency installation** - Find and install missing packages
7. **Install torchvision** - This breaks the environment!
8. **Re-install xformers** - **CRITICAL!** This fixes the broken environment

### Why Step 8 is Critical

Installing torchvision (step 7) causes it to:
- Uninstall the torch nightly build
- Install an incompatible torch version
- Break vLLM's C++ extensions

Re-installing xformers (step 8) fixes this by:
- Restoring the correct torch nightly build
- Fixing the C++ extension compatibility
- Making vLLM work again

**This is the most common point of failure and must not be skipped!**

### Pre-built Wheels

The solution uses pre-built wheels from:
```
https://github.com/ghcdmm/DeepSeek-OCR/releases/download/1/
```

These wheels are specifically compiled for:
- Python 3.12
- CUDA 12.8
- Linux x86_64

Using `pip install vllm` directly will **not work** because it tries to compile from source, which often fails with CUDA 12.8.

## Files Created

```
DeepSeek-OCR/
├── docs/
│   ├── INSTALL_CUDA_12.8.md      # Detailed installation guide
│   ├── TROUBLESHOOTING.md         # Comprehensive troubleshooting
│   └── ISSUE_240_SOLUTION.md      # This summary document
├── scripts/
│   ├── install_cuda128.sh         # Automated installation script
│   └── verify_installation.py     # Installation verification script
├── requirements_cuda128.txt       # CUDA 12.8 dependencies list
└── README.md                      # Updated with CUDA 12.8 section
```

## Usage Workflow

### For New Users

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
   cd DeepSeek-OCR
   ```

2. **Create conda environment:**
   ```bash
   conda create -n deepseek-ocr python=3.12.9 -y
   conda activate deepseek-ocr
   ```

3. **Run automated installation:**
   ```bash
   bash scripts/install_cuda128.sh
   ```

4. **Verify installation:**
   ```bash
   python scripts/verify_installation.py
   ```

5. **Test with DeepSeek-OCR:**
   ```bash
   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
   # Update config.py with your paths
   python run_dpsk_ocr_image.py
   ```

### For Users with Issues

1. **Check the troubleshooting guide:**
   ```bash
   cat docs/TROUBLESHOOTING.md
   ```

2. **Run verification to diagnose:**
   ```bash
   python scripts/verify_installation.py
   ```

3. **Try clean installation:**
   ```bash
   conda deactivate
   conda env remove -n deepseek-ocr
   conda create -n deepseek-ocr python=3.12.9 -y
   conda activate deepseek-ocr
   bash scripts/install_cuda128.sh
   ```

## Testing and Verification

The solution has been designed based on the successful installation reported in Issue #240. The verification script checks:

- ✓ Python 3.12.x
- ✓ CUDA 12.8 availability
- ✓ PyTorch with CUDA support
- ✓ vLLM import and version
- ✓ flash-attn availability
- ✓ xformers availability
- ✓ All critical dependencies

## Benefits of This Solution

1. **Automated Installation**: One command to install everything
2. **Comprehensive Documentation**: Detailed guides for manual installation
3. **Verification Tools**: Easy way to check if installation succeeded
4. **Troubleshooting Support**: Solutions for common problems
5. **Community Contribution**: Based on real user experience (Issue #240)
6. **Maintainable**: Clear structure for future updates

## Credits

This solution is based on:
- **GitHub Issue #240**: Community-reported successful installation
- **Issue #238**: Referenced in the original issue
- **@ghcdmm**: Provider of pre-built wheels

## Future Improvements

Potential enhancements:
- Support for other CUDA versions (12.1, 12.4, etc.)
- Docker container with pre-configured environment
- CI/CD testing for installation scripts
- Additional GPU model testing (RTX 4090, A100, etc.)
- Windows installation support

## Support

If you encounter issues:

1. **Read the documentation:**
   - [Installation Guide](INSTALL_CUDA_12.8.md)
   - [Troubleshooting Guide](TROUBLESHOOTING.md)

2. **Run verification:**
   ```bash
   python scripts/verify_installation.py
   ```

3. **Check existing issues:**
   - [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
   - Especially Issue #240

4. **Open a new issue** with:
   - Output of `python scripts/verify_installation.py`
   - Full error traceback
   - System information (GPU, CUDA version, Python version)

## Conclusion

This solution provides a complete, automated, and well-documented approach to installing DeepSeek-OCR with vLLM on CUDA 12.8 / RTX 5090 systems. It addresses the complex dependency chain and critical installation order documented in Issue #240, making it accessible to all users.

---

**Last Updated:** November 10, 2025  
**Issue Reference:** [#240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240)  
**Status:** ✅ Implemented and Documented
