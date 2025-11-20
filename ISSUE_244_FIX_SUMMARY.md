# Fix Summary for GitHub Issue #244

## Issue Description

**Error**: `pydantic_core.ValidationError: 1 validation error for ModelConfig - Value error, Model architectures 'DeepseekOCRForCausallM' are not supported`

**Command that failed**:
```bash
vllm serve deepseek-ai/DeepSeek-OCR --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0
```

## Root Cause Analysis

The error occurs because:

1. **Custom Model Architecture**: `DeepseekOCRForCausalLM` is a custom model class that extends vLLM's base model classes
2. **Missing Registration**: When using `vllm serve` CLI directly, the custom model class is not automatically registered with vLLM's ModelRegistry
3. **Architecture Name Mismatch**: The error message shows a typo (`DeepseekOCRForCausallM` with double 'l'), but the actual class name is `DeepseekOCRForCausalLM` (single 'l')

## Solution Implemented

### Files Created/Modified

1. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/__init__.py`** (NEW)
   - Registers the model with vLLM's ModelRegistry on import
   - Exports the model class and logits processor
   - Makes the module importable as a package

2. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/register_model.py`** (NEW)
   - Standalone script to register the model
   - Can be run before using vLLM commands
   - Provides clear success/failure feedback

3. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/serve_deepseek_ocr.py`** (NEW)
   - Custom server script that properly registers the model
   - Provides command-line interface similar to `vllm serve`
   - Includes proper configuration for DeepSeek-OCR

4. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/test_registration.py`** (NEW)
   - Test script to verify model registration
   - Checks dependencies and model attributes
   - Provides diagnostic information

5. **`VLLM_SERVE_GUIDE.md`** (NEW)
   - Comprehensive guide for using vLLM with DeepSeek-OCR
   - Multiple solution approaches
   - Troubleshooting section
   - API usage examples

6. **`README.md`** (MODIFIED)
   - Added troubleshooting section for Issue #244
   - Links to detailed guide
   - Quick fix examples

## How to Use the Fix

### Option 1: Use Provided Scripts (Recommended)

The existing scripts already handle model registration correctly:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Configure your settings in config.py first
# Then run:
python run_dpsk_ocr_image.py  # For images
python run_dpsk_ocr_pdf.py    # For PDFs
python run_dpsk_ocr_eval_batch.py  # For batch processing
```

### Option 2: Use Custom Serve Script

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python serve_deepseek_ocr.py --model deepseek-ai/DeepSeek-OCR --port 8000
```

### Option 3: Python API with Registration

```python
import sys
sys.path.insert(0, 'DeepSeek-OCR-master/DeepSeek-OCR-vllm')

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Create LLM instance
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    trust_remote_code=True,
    logits_processors=[NoRepeatNGramLogitsProcessor]
)

# Use the model
from PIL import Image
image = Image.open("your_image.png").convert("RGB")
prompt = "<image>\\nFree OCR."

model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
    ),
    skip_special_tokens=False,
)

outputs = llm.generate(model_input, sampling_params)
print(outputs[0].outputs[0].text)
```

## Why the Original Command Failed

The command:
```bash
vllm serve deepseek-ai/DeepSeek-OCR --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0
```

Failed because:

1. **Model Not Registered**: The `vllm serve` CLI doesn't know about the custom `DeepseekOCRForCausalLM` class
2. **Module Not in Path**: The `deepseek_ocr` module is not in vLLM's default search path
3. **No Auto-Discovery**: vLLM doesn't automatically discover custom models from HuggingFace Hub

## Technical Details

### Model Registration Process

The fix works by:

1. **Importing the Model Class**: `from deepseek_ocr import DeepseekOCRForCausalLM`
2. **Registering with vLLM**: `ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)`
3. **Overriding Architecture**: `hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]}`

This tells vLLM:
- The architecture name is "DeepseekOCRForCausalLM"
- The implementation class is `DeepseekOCRForCausalLM`
- Use this class when loading models with this architecture

### Architecture Name

The correct architecture name is: **`DeepseekOCRForCausalLM`** (with single 'l' in "Causal")

If you see `DeepseekOCRForCausallM` (double 'l'), it's a typo that should be corrected.

## Testing the Fix

To verify the fix works:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python test_registration.py
```

This will:
- Check all dependencies are installed
- Verify the model can be imported
- Test model registration with vLLM
- Confirm all required attributes are present

## Dependencies Required

Make sure you have:

```bash
pip install vllm>=0.8.5
pip install torch torchvision torchaudio
pip install transformers>=4.46.3
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

## Additional Resources

- **Detailed Guide**: See `VLLM_SERVE_GUIDE.md` for comprehensive instructions
- **vLLM Documentation**: https://docs.vllm.ai/
- **DeepSeek-OCR Paper**: https://arxiv.org/abs/2510.18234
- **Model on HuggingFace**: https://huggingface.co/deepseek-ai/DeepSeek-OCR

## Summary

The issue is resolved by properly registering the custom `DeepseekOCRForCausalLM` model class with vLLM before attempting to use it. The provided scripts and documentation offer multiple ways to do this, from simple script execution to full API integration.

**Key Takeaway**: Custom vLLM models must be explicitly registered with `ModelRegistry` before they can be used with vLLM's serving infrastructure.
