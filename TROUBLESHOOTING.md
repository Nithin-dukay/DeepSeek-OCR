# DeepSeek-OCR Troubleshooting Guide

## Issue #296: ModuleNotFoundError: No module named 'torch' when installing xformers

### Problem Description

When installing vLLM using the command:
```bash
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

You may encounter this error:
```
Collecting xformers==0.0.29.post2 (from vllm==0.8.5+cu118)
  Using cached xformers-0.0.29.post2.tar.gz (8.5 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [20 lines of output]
      ...
      ModuleNotFoundError: No module named 'torch'
      [end of output]
```

### Root Cause

The issue occurs because:
1. `vllm-0.8.5` has a dependency on `xformers==0.0.29.post2`
2. `xformers` is distributed as a source package (`.tar.gz`) that needs to be built
3. During the build process, `xformers` setup.py imports `torch` to determine build configuration
4. If `torch` is not already installed, the build fails immediately

This is a **dependency ordering problem** - `xformers` requires `torch` to be present during its build process, not just at runtime.

### Solution

**The fix is simple: Install PyTorch BEFORE installing vLLM.**

#### Method 1: Manual Installation (Recommended)

Follow these steps in order:

```bash
# Step 1: Install PyTorch first (CRITICAL!)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Step 2: Verify torch is installed
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

# Step 3: Now install vLLM (xformers will build successfully)
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Step 4: Install other requirements
pip install -r requirements.txt

# Step 5: Install flash-attention
pip install flash-attn==2.7.3 --no-build-isolation
```

#### Method 2: Using Installation Scripts

We provide automated installation scripts:

**Option A: Bash Script (Linux/Mac)**
```bash
chmod +x install.sh
./install.sh
```

**Option B: Python Script (Cross-platform)**
```bash
python install.py
```

Both scripts ensure the correct installation order and provide detailed progress information.

#### Method 3: Install xformers Separately

If the above methods still fail, try installing xformers separately:

```bash
# Install PyTorch first
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install xformers separately
pip install xformers==0.0.29.post2

# Then install vLLM
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install other requirements
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

### Verification

After installation, verify everything is working:

```bash
python -c "import torch; import vllm; import transformers; print('All packages imported successfully!')"
```

### Why This Happens

This is a common issue with Python packages that have build-time dependencies:

1. **Build-time vs Runtime dependencies**: `xformers` needs `torch` during the build process (build-time), not just when running (runtime)
2. **pip's dependency resolution**: pip tries to resolve all dependencies together, but doesn't distinguish between build-time and runtime dependencies
3. **Source distributions**: When a package is distributed as source (`.tar.gz`), pip must build it, which requires all build dependencies to be present

### Prevention

To avoid this issue in the future:

1. **Always read installation instructions carefully** - the order matters!
2. **Install build dependencies first** - packages like PyTorch, NumPy, etc. should be installed before packages that depend on them during build
3. **Use virtual environments** - this ensures a clean installation environment
4. **Check for pre-built wheels** - pre-built wheels (`.whl` files) don't have this issue because they're already compiled

## Other Common Issues

### Issue: CUDA version mismatch

**Symptom**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Solution**: Ensure your CUDA version matches the PyTorch installation:
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of memory errors

**Symptom**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Solution**:
1. Reduce batch size in your inference code
2. Use smaller image resolutions
3. Enable gradient checkpointing if training
4. Clear CUDA cache: `torch.cuda.empty_cache()`

### Issue: flash-attention compilation fails

**Symptom**: Errors during `pip install flash-attn==2.7.3`

**Solution**:
1. Ensure you have a compatible CUDA version (11.6+)
2. Ensure you have a compatible GPU (compute capability 7.5+)
3. Install with `--no-build-isolation` flag (already in instructions)
4. If it still fails, you can skip flash-attention (performance will be slower)

### Issue: Transformers version conflict

**Symptom**: `vllm 0.8.5+cu118 requires transformers>=4.51.1`

**Solution**: This is expected if you're using both vLLM and the transformers inference code. You can safely ignore this warning - the code will work with transformers 4.46.3 as specified in requirements.txt.

## Getting Help

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues) page
2. Provide the following information when reporting:
   - Python version: `python --version`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - CUDA version: `nvcc --version`
   - GPU model: `nvidia-smi`
   - Full error traceback
   - Installation commands you ran

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [xformers GitHub](https://github.com/facebookresearch/xformers)
