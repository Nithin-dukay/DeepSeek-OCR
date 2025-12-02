# DeepSeek-OCR Troubleshooting Guide

This guide addresses common issues when running DeepSeek-OCR, especially on Windows 11 and systems with limited GPU support.

## Table of Contents
- [Flash-Attention Issues](#flash-attention-issues)
- [Windows 11 Specific Issues](#windows-11-specific-issues)
- [GPU Compatibility Issues](#gpu-compatibility-issues)
- [Memory Issues](#memory-issues)
- [Installation Issues](#installation-issues)
- [Performance Optimization](#performance-optimization)

---

## Flash-Attention Issues

### Issue: "flash_attn is not installed"

**Symptoms:**
```
ImportError: flash_attn is not installed
```

**Solution:**
You don't need flash-attention! The model works fine without it.

**For Windows/CPU users:**
```python
# Remove the _attn_implementation parameter
model = AutoModel.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    use_safetensors=True
)
```

**Or use the provided scripts:**
```bash
python run_dpsk_ocr_windows.py  # Auto-detects hardware
python run_dpsk_ocr_cpu.py      # CPU-only
```

### Issue: Flash-attention installation fails on Windows

**Symptoms:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```
or
```
Building wheel for flash-attn failed
```

**Solution:**
Flash-attention has poor Windows support. **You don't need it!** Use the default attention mechanism:

1. Use `run_dpsk_ocr_windows.py` script (recommended)
2. Or modify your code to remove `_attn_implementation='flash_attention_2'`

The model will automatically use PyTorch's built-in attention, which works perfectly on Windows.

---

## Windows 11 Specific Issues

### Issue: Model fails to load on Windows 11

**Symptoms:**
- Import errors
- CUDA errors
- Model loading hangs

**Solutions:**

1. **Use the Windows-compatible script:**
   ```bash
   cd DeepSeek-OCR-master/DeepSeek-OCR-hf
   python run_dpsk_ocr_windows.py
   ```

2. **Check your Python version:**
   - Python 3.8-3.11 recommended
   - Python 3.13 may have compatibility issues with some dependencies
   - Downgrade if needed: `conda install python=3.11`

3. **Verify PyTorch installation:**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

4. **Install PyTorch for Windows:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchvision torchaudio
   ```

### Issue: "CUDA out of memory" on Windows

**Solution:**
Windows reserves more VRAM for the OS. Try:

1. Close all other applications
2. Use smaller image sizes:
   ```python
   base_size = 512
   image_size = 512
   crop_mode = False
   ```
3. Use CPU mode if GPU has limited memory

---

## GPU Compatibility Issues

### Issue: RTX 1060 / Older GPU not working

**Symptoms:**
- Flash-attention errors
- CUDA compatibility warnings
- Poor performance

**Solution:**

RTX 1060 has compute capability 6.1, which has limited support for newer features.

**Recommended approach:**
```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load without flash-attention, use float16 for older GPUs
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.float16  # Use float16 instead of bfloat16
)

model = model.eval().cuda()
```

**Or simply use:**
```bash
python run_dpsk_ocr_windows.py  # Automatically handles GPU compatibility
```

### Issue: "bfloat16 not supported on this GPU"

**Solution:**
Use float16 instead:
```python
model = model.to(torch.float16)  # Instead of torch.bfloat16
```

The `run_dpsk_ocr_windows.py` script handles this automatically.

---

## Memory Issues

### Issue: "RuntimeError: CUDA out of memory"

**Solutions:**

1. **Reduce image size:**
   ```python
   # Use Tiny configuration
   base_size = 512
   image_size = 512
   crop_mode = False
   ```

2. **Reduce batch size** (if processing multiple images)

3. **Clear CUDA cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Use CPU mode:**
   ```bash
   python run_dpsk_ocr_cpu.py
   ```

### Issue: "Out of memory" on CPU

**Symptoms:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:114] . DefaultCPUAllocator: can't allocate memory
```

**Solutions:**

1. **Close other applications** to free RAM

2. **Use smaller image sizes:**
   ```python
   base_size = 512
   image_size = 512
   ```

3. **Increase system swap/page file** (Windows):
   - Settings → System → About → Advanced system settings
   - Performance Settings → Advanced → Virtual memory
   - Set custom size (e.g., 32GB)

4. **Process images one at a time** instead of batches

---

## Installation Issues

### Issue: "No module named 'transformers'"

**Solution:**
```bash
pip install transformers==4.46.3
```

### Issue: "No module named 'einops'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Dependencies conflict

**Solution:**
Create a fresh environment:
```bash
# Using conda
conda create -n deepseek-ocr python=3.11 -y
conda activate deepseek-ocr

# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Skip flash-attn on Windows
# pip install flash-attn  # NOT needed on Windows!
```

### Issue: Python 3.13 compatibility

**Symptoms:**
Various import errors or package installation failures

**Solution:**
Python 3.13 is very new. Use Python 3.11 instead:
```bash
conda create -n deepseek-ocr python=3.11 -y
conda activate deepseek-ocr
```

---

## Performance Optimization

### CPU is too slow

**Expected behavior:**
CPU inference is 10-100x slower than GPU. This is normal.

**Optimization tips:**

1. **Use smallest image size for testing:**
   ```python
   base_size = 512
   image_size = 512
   crop_mode = False
   ```

2. **Disable unnecessary features:**
   ```python
   test_compress = False  # Skip compression testing
   ```

3. **Consider cloud GPU services:**
   - Google Colab (free GPU)
   - AWS, Azure, or GCP instances
   - Vast.ai (affordable GPU rental)

### GPU inference is slower than expected

**Possible causes:**

1. **Not using flash-attention** (intentional on Windows/older GPUs)
   - This is expected and necessary for compatibility
   - Performance is still reasonable

2. **Using float16 instead of bfloat16**
   - Necessary for older GPUs
   - Minimal performance impact

3. **Large image sizes**
   - Try smaller configurations first
   - Gundam mode (crop_mode=True) is slower but more accurate

---

## Common Error Messages

### "trust_remote_code=True required"

**Solution:**
Always include `trust_remote_code=True` when loading the model:
```python
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
```

### "Image file not found"

**Solution:**
Use absolute paths or verify the file exists:
```python
import os
image_file = os.path.abspath('test.png')
print(f"Image exists: {os.path.exists(image_file)}")
```

### "Output directory not writable"

**Solution:**
Create the directory first:
```python
import os
os.makedirs('output', exist_ok=True)
```

---

## Still Having Issues?

If you're still experiencing problems:

1. **Check the GitHub Issues:** [DeepSeek-OCR Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)

2. **Provide detailed information:**
   - Operating System (Windows 11, Linux, etc.)
   - Python version: `python --version`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - GPU model (if applicable)
   - Full error message and stack trace

3. **Try the minimal test:**
   ```python
   import torch
   from transformers import AutoModel, AutoTokenizer
   
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")
   
   model_name = 'deepseek-ai/DeepSeek-OCR'
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   print("✓ Tokenizer loaded successfully")
   ```

4. **Join the community:**
   - Discord: [DeepSeek AI Discord](https://discord.gg/Tc7c45Zzu5)
   - Twitter: [@deepseek_ai](https://twitter.com/deepseek_ai)
