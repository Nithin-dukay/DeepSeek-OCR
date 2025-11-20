# Solution for GitHub Issue #240: RTX 5090 / CUDA 12.8 Setup

## Issue Summary
Users with NVIDIA RTX 5090 GPUs running CUDA 12.8 were experiencing difficulties setting up DeepSeek-OCR with vLLM due to complex dependency conflicts and installation ordering requirements.

## Solution Implemented

This solution provides comprehensive documentation and automation tools to help users successfully install DeepSeek-OCR on RTX 5090 with CUDA 12.8.

### Files Created

1. **`docs/INSTALL_RTX5090_CUDA128.md`**
   - Comprehensive installation guide with step-by-step instructions
   - Detailed explanations of critical steps
   - Troubleshooting section for common issues
   - Performance optimization tips

2. **`scripts/setup_rtx5090_cuda128.sh`**
   - Automated installation script
   - Handles all dependency installation in correct order
   - Includes error handling and colored output
   - Automatically cleans up downloaded wheels

3. **`scripts/verify_installation.py`**
   - Python script to verify installation completeness
   - Checks all required dependencies
   - Tests CUDA availability
   - Identifies missing packages with installation commands
   - Provides detailed diagnostic information

4. **`README.md` (Updated)**
   - Added new section for RTX 5090 / CUDA 12.8 installation
   - Updated table of contents
   - Links to detailed installation guide
   - Quick setup instructions

### Key Features

#### Installation Guide
- Clear prerequisites and environment setup
- Step-by-step installation with explanations
- Critical sections highlighted (torchvision/xformers conflict)
- Troubleshooting for common issues
- Performance notes and optimization tips

#### Setup Script
- Automated installation process
- Color-coded output for better readability
- Error handling and validation
- Automatic cleanup of temporary files
- Progress indicators

#### Verification Script
- Comprehensive dependency checking
- CUDA availability testing
- vLLM component verification
- Missing package detection with install commands
- Detailed diagnostic output

### Installation Process Overview

The solution addresses the following critical steps:

1. **Install xformers nightly** (CUDA 12.8 compatible)
2. **Download pre-built wheels** (flash-attn and vLLM)
3. **Install wheels with special flags** (`--no-build-isolation --no-deps`)
4. **Install initial dependencies** (pydantic, transformers, etc.)
5. **Install torchvision** (triggers PyTorch version conflict)
6. **Re-install xformers** (fixes the conflict - CRITICAL STEP)
7. **Install final dependencies** (hf_transfer, prometheus_client)
8. **Verify installation** (check all components)

### Usage

#### Quick Setup (Recommended)
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
conda create -n deepseek-ocr-rtx5090 python=3.12 -y
conda activate deepseek-ocr-rtx5090
bash scripts/setup_rtx5090_cuda128.sh
python scripts/verify_installation.py
```

#### Manual Setup
Follow the detailed guide in `docs/INSTALL_RTX5090_CUDA128.md`

### Testing Performed

- ✅ Bash script syntax validation
- ✅ Python script syntax validation
- ✅ File permissions set correctly (executable)
- ✅ README.md updated with proper links
- ✅ Documentation completeness verified

### Benefits

1. **Reduced Setup Time**: Automated script reduces setup from hours to minutes
2. **Error Prevention**: Proper ordering prevents common installation failures
3. **Easy Troubleshooting**: Verification script quickly identifies issues
4. **Community Support**: Based on verified community solution
5. **Comprehensive Documentation**: Detailed guide for understanding the process

### Credits

This solution is based on the working setup shared by the community in GitHub Issue #240, with special thanks to @ghcdmm for providing the pre-built wheels.

### Future Improvements

Potential enhancements for future versions:
- Support for other CUDA versions (12.6, 12.7)
- Support for other Python versions (3.10, 3.11)
- Docker container with pre-configured environment
- CI/CD integration for testing
- Additional GPU models support

### Related Issues

- Issue #240: Original issue with working solution
- Issue #238: Referenced in the original issue

### Maintenance Notes

- Pre-built wheels are hosted at: https://github.com/ghcdmm/DeepSeek-OCR/releases/tag/1
- xformers version: 0.0.33.dev20251104+cu128
- vLLM version: 0.8.5+cu128
- flash-attn version: 2.8.3
- Python version: 3.12 (recommended)
- CUDA version: 12.8

If these versions become outdated, the installation guide and scripts will need to be updated accordingly.
