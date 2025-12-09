# Quick Start Guide - Issue #288 Fix

## TL;DR

**Problem**: Modified prompts cause the model to output repeated numbers.

**Solution**: Use the updated `NoRepeatNGramLogitsProcessor` with new parameters.

## Quick Fix (Copy-Paste)

### For vLLM Users:

Replace your old logits processor:

```python
# OLD (causes issues with modified prompts)
logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90)]

# NEW (works with all prompts)
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=10,      # NEW: prevents premature blocking
        enable_adaptive_ngram=True    # NEW: adaptive n-gram sizing
    )
]
```

### For Prompt Issues:

Use the prompt utilities to validate and fix your prompts:

```python
from process.prompt_utils import suggest_prompt_fix

# Your problematic prompt
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

# Get a suggested fix
fixed_prompt = suggest_prompt_fix(prompt)
print(fixed_prompt)
# Output: <image>\n<|grounding|>Convert the document to markdown.
```

## Recommended Prompts

Use these tested prompts for best results:

```python
from process.prompt_utils import get_recommended_prompts

prompts = get_recommended_prompts()

# Document to Markdown
prompt = prompts["document_to_markdown"]
# "<image>\n<|grounding|>Convert the document to markdown."

# Simple OCR
prompt = prompts["ocr_simple"]
# "<image>\n<|grounding|>Extract text."

# Free OCR (no layout)
prompt = prompts["free_ocr"]
# "<image>\nFree OCR."
```

## Complete Example

```python
import os
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from PIL import Image

# Initialize model
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
    max_model_len=8192,
)

# Use improved logits processor
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=10,
        enable_adaptive_ngram=True
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

# Your prompt (now works even if modified!)
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# Load image
image = Image.open("your_image.jpg").convert("RGB")

# Process
processor = DeepseekOCRProcessor()
processed = processor.tokenize_with_images(
    images=[image],
    bos=True,
    eos=True,
    cropping=True
)

# Generate
outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": processed}
    }],
    sampling_params=sampling_params
)

# Get result
result = outputs[0].outputs[0].text
print(result)
```

## Testing Your Setup

Run the test script to verify everything works:

```bash
python3 test_prompt_utils.py
```

Expected output: `✓ ALL TESTS PASSED`

## Common Issues

### Issue: Still getting repeated numbers

**Solution**: Increase `min_generated_tokens`:

```python
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=20,  # Increased from 10
        enable_adaptive_ngram=True
    )
]
```

### Issue: Output is too repetitive

**Solution**: Decrease `min_generated_tokens` or increase `ngram_size`:

```python
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=40,  # Increased from 30
        window_size=90,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=5,  # Decreased from 10
        enable_adaptive_ngram=True
    )
]
```

## Files Changed

1. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py`**
   - Enhanced with prompt tracking and adaptive behavior

2. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/prompt_utils.py`** (NEW)
   - Prompt validation and optimization utilities

3. **`ISSUE_288_FIX.md`** (NEW)
   - Comprehensive documentation

4. **`test_prompt_utils.py`** (NEW)
   - Test suite for the fix

## Need Help?

1. Read the full documentation: `ISSUE_288_FIX.md`
2. Run the test suite: `python3 test_prompt_utils.py`
3. Check your prompt: Use `prompt_utils.explain_prompt_issue(your_prompt)`

## Summary

✅ **The fix allows modified prompts to work correctly**  
✅ **No breaking changes to existing code**  
✅ **Backward compatible with old parameters**  
✅ **Includes utilities to help optimize prompts**  
✅ **Fully tested and documented**
