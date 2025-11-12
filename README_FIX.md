# DeepSeek-OCR Issue #244 Fix

## Overview

This directory contains a comprehensive fix for GitHub Issue #244, which causes a `ValidationError` when trying to serve the DeepSeek-OCR model with vLLM.

## The Problem

```
pydantic_core.ValidationError: 1 validation error for ModelConfig
Value error, Model architectures 'DeepseekOCRForCausallM' are not supported
```

**Root Cause**: The model's config.json has a typo: `DeepseekOCRForCausallM` (double 'l') instead of `DeepseekOCRForCausalLM` (single 'l').

## Quick Solutions

### 🚀 Option 1: Use the Serving Script (Recommended)

```bash
python serve_deepseek_ocr.py --model deepseek-ai/DeepSeek-OCR --port 8000
```

This automatically handles model registration and uses OCR-optimized settings.

### 🔧 Option 2: Fix the Config File

```bash
python fix_model_config.py
```

This permanently fixes the typo in your downloaded model's config.json.

### 📦 Option 3: Import Registration Module

```python
import register_deepseek_ocr  # Add this line before using vLLM
from vllm import LLM

llm = LLM(model="deepseek-ai/DeepSeek-OCR")
```

## Files Included

### Python Scripts

| File | Size | Purpose |
|------|------|---------|
| `register_deepseek_ocr.py` | 1.6K | Model registration module |
| `serve_deepseek_ocr.py` | 5.2K | vLLM serving script |
| `fix_model_config.py` | 7.3K | Config fix utility |
| `test_fix.py` | 7.2K | Test suite |

### Documentation

| File | Size | Purpose |
|------|------|---------|
| `QUICK_START.md` | 769B | Quick reference |
| `ISSUE_244_FIX.md` | 7.4K | Comprehensive guide |
| `SOLUTION_SUMMARY.md` | 5.2K | Executive summary |
| `CHANGELOG_ISSUE_244.md` | 4.8K | Complete changelog |
| `README_FIX.md` | This file | Fix overview |

## Testing

Run the test suite to verify everything works:

```bash
python test_fix.py
```

Expected output:
```
✓ All tests passed!
```

## Usage Examples

### Start vLLM Server

```bash
python serve_deepseek_ocr.py \
  --model deepseek-ai/DeepSeek-OCR \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

### Fix Model Config

```bash
# Auto-detect model location
python fix_model_config.py

# Or specify path
python fix_model_config.py /path/to/DeepSeek-OCR
```

### Use in Python Script

```python
# Import registration module
import register_deepseek_ocr

# Use vLLM normally
from vllm import LLM, SamplingParams

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
)

outputs = llm.generate(["<image>\nFree OCR."], sampling_params)
```

## Documentation

- **Quick Start**: See `QUICK_START.md` for fast solutions
- **Detailed Guide**: See `ISSUE_244_FIX.md` for comprehensive documentation
- **Summary**: See `SOLUTION_SUMMARY.md` for overview and examples
- **Changelog**: See `CHANGELOG_ISSUE_244.md` for all changes

## Troubleshooting

### "Could not register DeepSeek-OCR model"

Ensure vLLM is installed:
```bash
pip install vllm
```

### "Could not find config.json"

Download the model first:
```bash
huggingface-cli download deepseek-ai/DeepSeek-OCR
```

### "CUDA out of memory"

Reduce GPU memory usage:
```bash
python serve_deepseek_ocr.py --gpu-memory-utilization 0.7 --max-model-len 4096
```

## Features

✅ Multiple solution approaches  
✅ Backward compatible with existing scripts  
✅ Comprehensive test suite  
✅ Well-documented with examples  
✅ OCR-optimized default settings  
✅ Automatic model registration  
✅ Config backup before modification  
✅ Clear error messages  

## Support

For more help:
1. Check `QUICK_START.md` for quick solutions
2. Read `ISSUE_244_FIX.md` for detailed documentation
3. Run `test_fix.py` to diagnose problems
4. See main `README.md` troubleshooting section

## License

This fix is part of the DeepSeek-OCR project and follows the same license.
