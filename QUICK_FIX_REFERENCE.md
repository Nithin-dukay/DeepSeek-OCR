# Quick Fix Reference - Issue #244

## The Problem
```
pydantic_core.ValidationError: Model architectures 'DeepseekOCRForCausallM' are not supported
```

## Quick Solutions

### 1️⃣ Use Existing Scripts (Easiest)
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
# Edit config.py to set INPUT_PATH and OUTPUT_PATH
python run_dpsk_ocr_image.py
```

### 2️⃣ Use Custom Server Script
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python serve_deepseek_ocr.py --model deepseek-ai/DeepSeek-OCR --port 8000
```

### 3️⃣ Use Example Script
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python example_usage.py --image your_image.png
```

### 4️⃣ Python Code (Copy-Paste Ready)
```python
import sys
sys.path.insert(0, 'DeepSeek-OCR-master/DeepSeek-OCR-vllm')

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image

# IMPORTANT: Register the model first!
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Now create LLM
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    trust_remote_code=True
)

# Use it
image = Image.open("image.png").convert("RGB")
outputs = llm.generate([{
    "prompt": "<image>\\nFree OCR.",
    "multi_modal_data": {"image": image}
}], SamplingParams(temperature=0.0, max_tokens=8192))

print(outputs[0].outputs[0].text)
```

## Why It Failed

❌ **Wrong**: Using `vllm serve` directly without model registration
```bash
vllm serve deepseek-ai/DeepSeek-OCR  # This fails!
```

✅ **Right**: Register model first, then use vLLM
```python
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
# Now vLLM knows about the model
```

## Key Points

1. **Model Registration Required**: Custom models must be registered with `ModelRegistry`
2. **Correct Architecture Name**: `DeepseekOCRForCausalLM` (single 'l' in Causal)
3. **Use Provided Scripts**: They handle registration automatically
4. **Trust Remote Code**: Always use `trust_remote_code=True`

## Files Created

- ✅ `__init__.py` - Auto-registers model on import
- ✅ `register_model.py` - Standalone registration script
- ✅ `serve_deepseek_ocr.py` - Custom server with registration
- ✅ `example_usage.py` - Complete usage example
- ✅ `test_registration.py` - Test model registration
- ✅ `VLLM_SERVE_GUIDE.md` - Detailed guide
- ✅ `ISSUE_244_FIX_SUMMARY.md` - Complete fix documentation

## Need Help?

See detailed documentation:
- `VLLM_SERVE_GUIDE.md` - Comprehensive guide
- `ISSUE_244_FIX_SUMMARY.md` - Technical details
- `README.md` - Updated with troubleshooting section
