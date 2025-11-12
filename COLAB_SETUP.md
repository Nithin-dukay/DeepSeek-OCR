# DeepSeek-OCR Setup Guide for Google Colab

This guide helps you set up DeepSeek-OCR in Google Colab and avoid common issues.

## Quick Setup (Copy-Paste Ready)

```python
# 1. Clone the repository
!git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
%cd DeepSeek-OCR

# 2. Install the CORRECT transformers version (CRITICAL!)
!pip install transformers==4.46.3 --force-reinstall

# 3. Install other dependencies
!pip install -r requirements.txt

# 4. RESTART THE RUNTIME
# Go to: Runtime -> Restart runtime
# This is REQUIRED for the transformers version change to take effect

# 5. After restart, verify your environment
!python check_environment.py

# 6. Run DeepSeek-OCR
%cd DeepSeek-OCR-master/DeepSeek-OCR-hf
!python run_dpsk_ocr.py
```

## Common Issues and Solutions

### Issue 1: ImportError: cannot import name 'LlamaFlashAttention2'

**Error Message:**
```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

**Cause:** You have transformers 4.47+ installed, which has breaking changes.

**Solution:**
```python
!pip install transformers==4.46.3 --force-reinstall
# Then: Runtime -> Restart runtime (REQUIRED!)
```

### Issue 2: Version Check Fails

**Error Message:**
```
⚠️  WARNING: Incompatible transformers version detected!
```

**Solution:**
1. Check your current version:
   ```python
   import transformers
   print(transformers.__version__)
   ```

2. If it's not 4.46.3, reinstall:
   ```python
   !pip install transformers==4.46.3 --force-reinstall
   ```

3. **IMPORTANT:** Restart the runtime (Runtime -> Restart runtime)

4. Verify after restart:
   ```python
   import transformers
   print(transformers.__version__)  # Should show: 4.46.3
   ```

### Issue 3: CUDA Out of Memory

**Solution:** Use smaller model configurations:

```python
# Instead of:
# base_size = 1024, image_size = 640  (Gundam mode)

# Try:
# Tiny mode
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, 
                  base_size=512, image_size=512, crop_mode=False)

# Or Small mode
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, 
                  base_size=640, image_size=640, crop_mode=False)
```

## Environment Verification

Before running DeepSeek-OCR, always verify your environment:

```python
!python check_environment.py
```

This will check:
- ✅ transformers version (must be 4.46.3)
- ✅ All required dependencies
- ✅ Optional packages
- ✅ Provide fix instructions if needed

## Why transformers 4.46.3?

DeepSeek-OCR uses custom model code that depends on `LlamaFlashAttention2` from the transformers library. This class was removed/renamed in transformers 4.47+, causing compatibility issues.

**Key Points:**
- ⚠️ transformers 4.47+ will NOT work
- ✅ transformers 4.46.3 is the tested and working version
- 🔄 Always restart runtime after changing transformers version

## Complete Example

```python
# Complete working example for Colab
!git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
%cd DeepSeek-OCR

# Install correct version
!pip install transformers==4.46.3 --force-reinstall
!pip install -r requirements.txt

# STOP HERE - Restart runtime now!
# Runtime -> Restart runtime

# After restart, run this cell:
%cd DeepSeek-OCR
!python check_environment.py

# If all checks pass, run DeepSeek-OCR:
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# Use the model
prompt = "<image>\\nFree OCR."
image_file = 'your_image.jpg'
output_path = 'output'

res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=1024, 
    image_size=640, 
    crop_mode=True, 
    save_results=True
)
```

## Getting Help

If you continue to have issues:

1. Check the [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
2. Specifically see [Issue #7](https://github.com/deepseek-ai/DeepSeek-OCR/issues/7) for this error
3. Run `!python check_environment.py` and share the output

## Related Issues

- [DeepSeek-OCR Issue #7](https://github.com/deepseek-ai/DeepSeek-OCR/issues/7)
- [DeepSeek-VL2 Issue #87](https://github.com/deepseek-ai/DeepSeek-VL2/issues/87)
