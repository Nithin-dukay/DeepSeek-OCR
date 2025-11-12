# Quick Start Guide - Fix for Issue #244

## Problem
Getting `ValidationError: Model architectures 'DeepseekOCRForCausallM' are not supported` when using vLLM?

## Quick Fix (Choose One)

### Option 1: Use Our Serving Script ⭐ Recommended
```bash
python serve_deepseek_ocr.py --model deepseek-ai/DeepSeek-OCR --port 8000
```

### Option 2: Fix Your Model Config
```bash
python fix_model_config.py
```
Then use vLLM normally:
```bash
vllm serve deepseek-ai/DeepSeek-OCR --trust-remote-code
```

### Option 3: Use in Python Code
```python
import register_deepseek_ocr  # Add this line
from vllm import LLM

llm = LLM(model="deepseek-ai/DeepSeek-OCR")
```

## Test the Fix
```bash
python test_fix.py
```

## Need Help?
See `ISSUE_244_FIX.md` for detailed documentation.
