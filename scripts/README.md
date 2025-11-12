# DeepSeek-OCR Installation Scripts

This directory contains automated scripts for installing and verifying DeepSeek-OCR with vLLM.

## Scripts

### install_cuda128_rtx5090.sh

**Purpose:** Automated installation script for DeepSeek-OCR with vLLM on CUDA 12.8 / RTX 5090 systems.

**Usage:**
```bash
bash scripts/install_cuda128_rtx5090.sh
```

**What it does:**
1. Checks prerequisites (Python version, CUDA, pip)
2. Installs xformers nightly build
3. Downloads and installs pre-built wheels for flash-attn and vLLM
4. Installs initial dependencies
5. Iteratively resolves missing dependencies
6. Installs torchvision and fixes the xformers conflict
7. Installs final dependencies
8. Installs DeepSeek-OCR requirements
9. Verifies the installation

**Features:**
- Color-coded output for easy reading
- Progress indicators
- Error handling
- Automatic dependency resolution
- Installation verification
- Time tracking

**Requirements:**
- Python 3.12 (recommended)
- CUDA 12.8
- Internet connection
- Conda environment activated

**Estimated time:** 10-20 minutes depending on internet speed

---

### verify_installation.py

**Purpose:** Comprehensive verification script to check if all required packages are correctly installed.

**Usage:**
```bash
python scripts/verify_installation.py
```

**What it checks:**
1. Python version (3.10+ required, 3.12 recommended)
2. Core ML packages (PyTorch, torchvision, xformers, flash_attn)
3. vLLM installation and components
4. Transformers library
5. Other dependencies (pydantic, PIL, numpy, etc.)
6. CUDA availability and compatibility
7. DeepSeek-OCR project files

**Output:**
- Color-coded status for each check (✓ success, ✗ error, ⚠ warning)
- Package versions
- CUDA information (version, device count, device names)
- Summary with pass/fail counts
- Troubleshooting suggestions

**Exit codes:**
- `0`: All critical checks passed
- `1`: Some critical checks failed

**Example output:**
```
======================================================================
DeepSeek-OCR Installation Verification
======================================================================

Python Version
--------------
ℹ Python version: 3.12.0
✓ Python 3.12 detected (recommended)

Core ML Packages
----------------
✓ PyTorch: 2.7.0.dev20251104+cu128
ℹ   CUDA available: Yes (version 12.8)
ℹ   GPU devices: 1
ℹ   Device 0: NVIDIA GeForce RTX 5090
✓ Torchvision: 0.22.0.dev20251104+cu128
✓ xformers: 0.0.33.dev20251104+cu128
✓ flash_attn: 2.8.3

vLLM
----
✓ vLLM: 0.8.5+cu128
✓   Core components importable
✓   Engine components importable
✓   Model registry importable

...

Summary
-------
Total checks: 8
Passed: 8
Failed: 0

✓ All checks passed! Installation is complete.
```

---

## Troubleshooting

### Script won't run

**Problem:** Permission denied
```bash
bash: ./install_cuda128_rtx5090.sh: Permission denied
```

**Solution:** Make the script executable
```bash
chmod +x scripts/install_cuda128_rtx5090.sh
```

### Python not found

**Problem:** `python: command not found`

**Solution:** Use `python3` instead or create an alias
```bash
# Use python3
python3 scripts/verify_installation.py

# Or create alias
alias python=python3
```

### Installation script fails

**Problem:** Script exits with error

**Solution:** 
1. Check the error message
2. Ensure you have activated the correct conda environment
3. Verify CUDA 12.8 is installed: `nvcc --version`
4. Check internet connection
5. Try manual installation following docs/INSTALL_CUDA_12.8.md

### Verification fails

**Problem:** Verification script reports failures

**Solution:**
1. Read the specific error messages
2. Install missing packages: `pip install <package_name>`
3. For vLLM C++ errors, re-install xformers:
   ```bash
   pip install xformers==0.0.33.dev20251104+cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu128
   ```
4. See docs/INSTALL_CUDA_12.8.md for detailed troubleshooting

## Additional Resources

- [Full Installation Guide](../docs/INSTALL_CUDA_12.8.md)
- [Quick Start Guide](../docs/QUICK_START_CUDA_12.8.md)
- [Issue Resolution Summary](../ISSUE_240_RESOLUTION.md)
- [GitHub Issue #240](https://github.com/deepseek-ai/DeepSeek-OCR/issues/240)

## Contributing

If you encounter issues or have improvements for these scripts:

1. Check existing issues on GitHub
2. Create a new issue with detailed information
3. Submit a pull request with improvements

## License

These scripts are part of the DeepSeek-OCR project and follow the same license terms.
