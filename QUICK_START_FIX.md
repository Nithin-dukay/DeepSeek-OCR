# Quick Start: Fix for Issue #288

## Problem
Modified prompts cause the model to output repeated numbers instead of performing OCR.

## Solution
Two files have been updated/added to fix this issue:

### 1. Updated: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py`
- Enhanced N-gram processor that doesn't interfere with prompt content
- Adds adaptive blocking and minimum generation threshold

### 2. New: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/prompt_utils.py`
- Prompt validation and optimization utilities
- Recommended prompt templates
- Helpful warnings and suggestions

## Quick Usage

### Option 1: Use Recommended Prompts (Easiest)
```python
from process.prompt_utils import PromptValidator

# Get a recommended prompt
prompt = PromptValidator.get_recommended_prompt("document_markdown")
# Result: "<image>\n<|grounding|>Convert the document to markdown."
```

### Option 2: Validate Your Custom Prompt
```python
from process.prompt_utils import validate_and_optimize_prompt

# Your custom prompt
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

# Validate and optimize
optimized = validate_and_optimize_prompt(prompt, verbose=True)
# Will show warnings and suggestions
```

### Option 3: Use Improved Processor Settings
```python
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Create processor with improved settings
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=10,    # NEW: Wait before blocking
        enable_adaptive=True         # NEW: Smarter blocking
    )
]
```

## What Changed?

### Before (Problematic)
```python
# Old processor was too aggressive
logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90)]

# Custom prompts would fail
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
# Result: Model outputs repeated numbers ❌
```

### After (Fixed)
```python
# New processor with smart defaults
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        min_generated_tokens=10,    # Waits for generation to start
        enable_adaptive=True         # Only blocks repeated patterns
    )
]

# Custom prompts now work
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
# Result: Model performs OCR correctly ✅
```

## Testing

Run the test script to verify the fix:
```bash
python test_prompt_validation.py
```

Expected output:
```
✅ ALL TESTS PASSED!
```

## Recommended Prompts

Instead of creating custom prompts, use these tested templates:

| Task | Prompt |
|------|--------|
| Document to Markdown | `<image>\n<|grounding|>Convert the document to markdown.` |
| OCR Image | `<image>\n<|grounding|>OCR this image.` |
| Free OCR | `<image>\nFree OCR.` |
| Parse Figure | `<image>\nParse the figure.` |
| Describe Image | `<image>\nDescribe this image in detail.` |

## Need More Details?

See `ISSUE_288_FIX.md` for:
- Complete technical explanation
- Advanced configuration options
- Troubleshooting guide
- Migration guide for existing code

## Summary

✅ **The fix is backward compatible** - existing code works without changes
✅ **Modified prompts now work** - no more repeated number outputs
✅ **Better validation** - get warnings and suggestions for problematic prompts
✅ **Smarter blocking** - prevents false-positive repetition detection

The model now handles custom prompts correctly while still preventing unwanted repetitions!
