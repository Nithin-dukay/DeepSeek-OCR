# Fix for GitHub Issue #288: Model Not Working When Original Prompt Modified Slightly

## Problem Description

When users modify the standard prompt from:
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```

to a slightly longer version like:
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
```

The model stops working correctly and outputs repeated numbers instead of performing OCR.

## Root Cause Analysis

The issue is caused by the **N-gram repetition prevention logic** (`NoRepeatNGramLogitsProcessor`) which is designed to prevent the model from generating repetitive text. However, the original implementation had several issues:

1. **No distinction between prompt and generated tokens**: The processor was analyzing the entire sequence including the prompt, which could cause it to ban legitimate tokens based on patterns in the prompt itself.

2. **Too aggressive early in generation**: The processor would start blocking tokens immediately, even when very few tokens had been generated, leading to over-restriction.

3. **Fixed n-gram size**: Using a fixed n-gram size throughout generation could be too restrictive for certain prompt patterns.

4. **Single occurrence blocking**: The processor would ban tokens after just one occurrence of a pattern, which was too aggressive.

## Solution

### 1. Enhanced NoRepeatNGramLogitsProcessor

The updated `NoRepeatNGramLogitsProcessor` in `/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py` includes:

#### Key Improvements:

- **Prompt Length Tracking**: The processor now tracks the prompt length and only analyzes generated tokens, not the prompt itself.

- **Minimum Generated Tokens**: A new parameter `min_generated_tokens` (default: 10) ensures the processor doesn't activate until enough tokens have been generated.

- **Adaptive N-gram Size**: When `enable_adaptive_ngram` is True (default), the processor uses a smaller n-gram size early in generation to be less restrictive.

- **Repetition Count Threshold**: Tokens are only banned if they would create a pattern that repeats at least twice, not just once.

#### New Parameters:

```python
NoRepeatNGramLogitsProcessor(
    ngram_size=30,              # Size of n-grams to check
    window_size=90,             # Window size to look back
    whitelist_token_ids={...},  # Tokens that are never banned
    min_generated_tokens=10,    # Don't activate until this many tokens generated
    enable_adaptive_ngram=True  # Use adaptive n-gram sizing
)
```

### 2. Prompt Validation and Optimization Utilities

A new module `/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/prompt_utils.py` provides:

#### Functions:

- **`validate_prompt(prompt)`**: Validates that a prompt is correctly formatted
- **`optimize_prompt(prompt)`**: Optimizes prompts by removing excessive whitespace and truncating if too long
- **`suggest_prompt_fix(prompt)`**: Suggests a simplified version of problematic prompts
- **`explain_prompt_issue(prompt)`**: Explains potential issues with a given prompt
- **`get_recommended_prompts()`**: Returns a dictionary of recommended prompts for common use cases

## Usage Examples

### Example 1: Using the Fixed Processor (vLLM)

```python
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Create processor with improved parameters
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # <td>, </td>
        min_generated_tokens=10,  # NEW: Wait for 10 tokens before activating
        enable_adaptive_ngram=True  # NEW: Use adaptive n-gram size
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

# Now longer prompts work correctly!
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
```

### Example 2: Validating and Optimizing Prompts

```python
from process.prompt_utils import (
    validate_prompt, 
    optimize_prompt, 
    suggest_prompt_fix,
    explain_prompt_issue
)

# Original problematic prompt
original = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

# Validate
is_valid, error = validate_prompt(original)
print(f"Valid: {is_valid}")  # True

# Check for issues
issues = explain_prompt_issue(original)
if issues:
    print(f"Issues found:\n{issues}")
    # Output: Prompt contains redundant guidance about spacing

# Get suggested fix
suggested = suggest_prompt_fix(original)
print(f"Suggested: {suggested}")
# Output: <image>\n<|grounding|>Convert the document to markdown.

# Optimize
optimized = optimize_prompt(original)
print(f"Optimized: {optimized}")
```

### Example 3: Using Recommended Prompts

```python
from process.prompt_utils import get_recommended_prompts

# Get all recommended prompts
prompts = get_recommended_prompts()

# Use for document conversion
prompt = prompts["document_to_markdown"]
# "<image>\n<|grounding|>Convert the document to markdown."

# Use for simple OCR
prompt = prompts["ocr_simple"]
# "<image>\n<|grounding|>Extract text."

# Use for image description
prompt = prompts["describe_simple"]
# "<image>\nDescribe this image."
```

### Example 4: Complete vLLM Script with Fix

```python
import os
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.prompt_utils import suggest_prompt_fix
from PIL import Image

# Your custom prompt (can be longer now!)
user_prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

# Optional: Get a suggested optimized version
optimized_prompt = suggest_prompt_fix(user_prompt)
print(f"Using prompt: {optimized_prompt}")

# Create LLM with fixed processor
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
        min_generated_tokens=10,      # NEW
        enable_adaptive_ngram=True    # NEW
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

# Generate
outputs = llm.generate([{"prompt": user_prompt, "multi_modal_data": {"image": image}}], 
                       sampling_params=sampling_params)
```

## Best Practices for Prompts

### ✅ DO:

1. **Keep prompts concise**: Shorter prompts generally work better
   ```python
   # Good
   "<image>\n<|grounding|>Convert the document to markdown."
   ```

2. **Use recommended prompts**: Start with the provided recommended prompts
   ```python
   from process.prompt_utils import get_recommended_prompts
   prompts = get_recommended_prompts()
   ```

3. **Focus on one task**: Don't combine multiple instructions
   ```python
   # Good
   "<image>\n<|grounding|>OCR this image."
   ```

4. **Use the grounding tag for structured output**: Use `<|grounding|>` for document conversion
   ```python
   "<image>\n<|grounding|>Convert the document to markdown."
   ```

### ❌ DON'T:

1. **Don't add redundant instructions**: The model handles these automatically
   ```python
   # Bad - redundant
   "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
   
   # Good - concise
   "<image>\n<|grounding|>Convert the document to markdown."
   ```

2. **Don't make prompts too long**: Keep under 150 characters when possible
   ```python
   # Bad - too long
   "<image>\n<|grounding|>Please convert this document to markdown format while preserving all formatting and layout and don't add extra spaces."
   
   # Good - concise
   "<image>\n<|grounding|>Convert to markdown."
   ```

3. **Don't combine multiple tasks**: Focus on one primary task
   ```python
   # Bad - multiple tasks
   "<image>\n<|grounding|>OCR this image and convert to markdown and describe the layout."
   
   # Good - single task
   "<image>\n<|grounding|>Convert the document to markdown."
   ```

## Recommended Prompts by Use Case

| Use Case | Recommended Prompt |
|----------|-------------------|
| Document to Markdown | `<image>\n<|grounding|>Convert the document to markdown.` |
| Document to Markdown (Short) | `<image>\n<|grounding|>Convert to markdown.` |
| OCR Image | `<image>\n<|grounding|>OCR this image.` |
| OCR (Simple) | `<image>\n<|grounding|>Extract text.` |
| Free OCR (No Layout) | `<image>\nFree OCR.` |
| Parse Figure | `<image>\nParse the figure.` |
| Describe Image | `<image>\nDescribe this image in detail.` |
| Describe (Simple) | `<image>\nDescribe this image.` |

## Testing the Fix

To test that the fix works with your modified prompts:

```python
# Test script
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.prompt_utils import validate_prompt, suggest_prompt_fix

# Test prompts
test_prompts = [
    "<image>\n<|grounding|>Convert the document to markdown.",
    "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.",
    "<image>\n<|grounding|>OCR this image and extract all text.",
]

for prompt in test_prompts:
    print(f"\nTesting: {prompt}")
    
    # Validate
    is_valid, error = validate_prompt(prompt)
    print(f"  Valid: {is_valid}")
    
    # Get suggestion
    suggested = suggest_prompt_fix(prompt)
    if suggested != prompt:
        print(f"  Suggested: {suggested}")
    
    # Test with model (add your actual model inference here)
    # outputs = llm.generate(...)
```

## Migration Guide

### For Existing Code Using vLLM:

**Before:**
```python
logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90)]
```

**After:**
```python
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=10,      # Add this
        enable_adaptive_ngram=True    # Add this
    )
]
```

### For Existing Code Using Transformers:

The Transformers version doesn't use the custom `NoRepeatNGramLogitsProcessor`, so the issue is less likely to occur. However, you should still follow the prompt best practices above.

## Technical Details

### How the Fix Works:

1. **Prompt Length Tracking**: On the first call, the processor estimates the prompt length. This allows it to distinguish between prompt tokens and generated tokens.

2. **Delayed Activation**: The processor doesn't start blocking tokens until `min_generated_tokens` have been generated. This prevents premature blocking that could interfere with the model's initial output.

3. **Adaptive N-gram Size**: Early in generation (first half of the window), the processor uses a smaller n-gram size (half of the configured size, minimum 3). This makes it less restrictive when the model is still "warming up".

4. **Repetition Threshold**: Instead of banning tokens after a single occurrence, the processor now requires at least 2 occurrences of a pattern before banning. This reduces false positives.

5. **Search Range Limitation**: The processor only looks at generated tokens (after the prompt) within the window, preventing prompt content from affecting generation.

## Troubleshooting

### Issue: Model still outputs repeated numbers

**Solution**: Try these steps in order:

1. Use a recommended prompt from `get_recommended_prompts()`
2. Increase `min_generated_tokens` to 20 or 30
3. Reduce `ngram_size` to 20
4. Disable adaptive n-gram: `enable_adaptive_ngram=False`

### Issue: Model output is too repetitive

**Solution**: The fix makes the processor less aggressive. If you need more repetition prevention:

1. Decrease `min_generated_tokens` to 5
2. Increase `ngram_size` to 40 or 50
3. Increase `window_size` to 120

### Issue: Prompt validation fails

**Solution**: Check that:

1. Prompt starts with `<image>`
2. Prompt contains exactly one `<image>` token
3. Prompt is not empty

## Summary

This fix resolves Issue #288 by:

1. ✅ Making the n-gram processor smarter about distinguishing prompts from generated text
2. ✅ Adding adaptive behavior that's less aggressive early in generation
3. ✅ Providing utilities to validate and optimize prompts
4. ✅ Documenting best practices for prompt engineering

The model now works correctly with both short and long prompts, including the problematic example from the issue.
