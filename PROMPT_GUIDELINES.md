# DeepSeek-OCR Prompt Guidelines

## Issue #288 Fix: Model Not Working with Modified Prompts

### Problem Description
When modifying the original prompt from:
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```
to:
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
```

The model would start outputting repeated numbers instead of proper OCR results.

### Root Cause
The issue was caused by the **N-gram repetition prevention logic** (`NoRepeatNGramLogitsProcessor`) being too aggressive with longer or modified prompts. The processor would ban legitimate tokens during generation, causing the model to fall back to generating unexpected output like repeated numbers.

### Solution
The fix includes:

1. **Enhanced NoRepeatNGramLogitsProcessor**: 
   - Added `min_generated_tokens` parameter to delay n-gram blocking until sufficient tokens are generated
   - Added `prompt_length` parameter to exclude the prompt from n-gram analysis
   - Modified logic to only ban tokens that appear multiple times (actual repetition)
   - Improved search range validation

2. **Prompt Validation Utilities**: 
   - Created `prompt_utils.py` with helper functions for prompt validation and sanitization
   - Added recommended prompt templates
   - Provided functions to create safe, well-formatted prompts

## Best Practices for Prompt Modification

### ✅ DO:

1. **Keep prompts concise and clear**
   ```python
   # Good
   prompt = "<image>\n<|grounding|>Convert the document to markdown."
   
   # Also good
   prompt = "<image>\n<|grounding|>Extract all text from this document."
   ```

2. **Use proper formatting**
   - Always include `<image>` token
   - Add newline after `<image>` token
   - Use appropriate task prefix (`<|grounding|>` for grounding tasks)
   - End with proper punctuation or space

3. **Use the prompt validation utilities**
   ```python
   from process.prompt_utils import validate_prompt, sanitize_prompt
   
   # Validate your prompt
   is_valid, error = validate_prompt(your_prompt)
   if not is_valid:
       print(f"Invalid prompt: {error}")
   
   # Or sanitize it automatically
   safe_prompt = sanitize_prompt(your_prompt)
   ```

4. **Adjust n-gram parameters for longer prompts**
   ```python
   from process.prompt_utils import get_recommended_ngram_params
   from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
   
   # Get recommended parameters based on your prompt
   params = get_recommended_ngram_params(your_prompt)
   
   # Create processor with recommended settings
   logits_processor = NoRepeatNGramLogitsProcessor(
       ngram_size=params['ngram_size'],
       window_size=params['window_size'],
       min_generated_tokens=params['min_generated_tokens'],
       whitelist_token_ids={128821, 128822}
   )
   ```

### ❌ DON'T:

1. **Don't make prompts excessively long**
   ```python
   # Bad - too long and complex
   prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters, make sure to preserve all formatting, and also check for any errors, and fix them if you find any, and also..."
   ```

2. **Don't remove essential tokens**
   ```python
   # Bad - missing <image> token
   prompt = "<|grounding|>Convert the document to markdown."
   
   # Bad - missing newline after <image>
   prompt = "<image><|grounding|>Convert the document to markdown."
   ```

3. **Don't use inconsistent formatting**
   ```python
   # Bad - inconsistent spacing
   prompt = "<image>    \n\n\n    <|grounding|>Convert the document to markdown."
   ```

## Recommended Prompt Templates

### Document OCR Tasks

```python
# Standard document to markdown conversion
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# OCR with specific focus
prompt = "<image>\n<|grounding|>OCR this image."

# Extract text without layout
prompt = "<image>\nFree OCR."
```

### Figure and Chart Tasks

```python
# Parse figures in documents
prompt = "<image>\nParse the figure."

# Describe charts
prompt = "<image>\nDescribe this chart in detail."
```

### General Vision Tasks

```python
# General image description
prompt = "<image>\nDescribe this image in detail."

# Locate specific elements
prompt = "<image>\nLocate <|ref|>specific_text<|/ref|> in the image."
```

## Using the Prompt Utilities

### Example 1: Create a Safe Prompt

```python
from process.prompt_utils import create_safe_prompt

# Create a safe prompt for document conversion
prompt = create_safe_prompt(
    base_instruction="Convert the document to markdown",
    task_type="grounding"
)
print(prompt)
# Output: "<image>\n<|grounding|>Convert the document to markdown. "
```

### Example 2: Validate Custom Prompts

```python
from process.prompt_utils import validate_prompt

custom_prompt = "<image>\n<|grounding|>Extract text with formatting."

is_valid, error = validate_prompt(custom_prompt)
if is_valid:
    print("Prompt is valid!")
else:
    print(f"Prompt validation failed: {error}")
```

### Example 3: Get Recommended Template

```python
from process.prompt_utils import get_prompt_template

# Get a recommended template
prompt = get_prompt_template("document_to_markdown")
print(prompt)
# Output: "<image>\n<|grounding|>Convert the document to markdown."
```

## Updated Usage Examples

### For vLLM Inference

```python
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.prompt_utils import create_safe_prompt, get_recommended_ngram_params

# Create a safe prompt
prompt = create_safe_prompt(
    base_instruction="Convert the document to markdown",
    task_type="grounding"
)

# Get recommended n-gram parameters
ngram_params = get_recommended_ngram_params(prompt)

# Create logits processor with proper parameters
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=ngram_params['ngram_size'],
        window_size=ngram_params['window_size'],
        min_generated_tokens=ngram_params['min_generated_tokens'],
        whitelist_token_ids={128821, 128822}
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

# Use with your model
outputs = llm.generate(inputs, sampling_params=sampling_params)
```

### For Transformers Inference

```python
from transformers import AutoModel, AutoTokenizer
from process.prompt_utils import create_safe_prompt

# Create a safe prompt
prompt = create_safe_prompt(
    base_instruction="Convert the document to markdown",
    task_type="grounding"
)

# Use with model
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True
)
```

## Troubleshooting

### Issue: Model outputs repeated numbers or gibberish

**Possible causes:**
1. Prompt is too long or complex
2. N-gram blocking is too aggressive
3. Prompt formatting is incorrect

**Solutions:**
1. Simplify your prompt
2. Use `get_recommended_ngram_params()` to adjust parameters
3. Use `sanitize_prompt()` to fix formatting issues
4. Increase `min_generated_tokens` parameter

### Issue: Model doesn't follow instructions

**Possible causes:**
1. Missing or incorrect task prefix (`<|grounding|>`)
2. Prompt doesn't match expected format

**Solutions:**
1. Use `create_safe_prompt()` to generate proper format
2. Check prompt with `validate_prompt()`
3. Use recommended templates from `get_prompt_template()`

## Migration Guide

If you have existing code with custom prompts:

### Before (may cause issues):
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90)]
```

### After (recommended):
```python
from process.prompt_utils import create_safe_prompt, get_recommended_ngram_params

# Option 1: Use a simpler, clearer prompt
prompt = create_safe_prompt(
    base_instruction="Convert the document to markdown",
    task_type="grounding"
)

# Option 2: If you need the custom instruction, use it but with proper parameters
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
ngram_params = get_recommended_ngram_params(prompt)

logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=ngram_params['ngram_size'],
        window_size=ngram_params['window_size'],
        min_generated_tokens=ngram_params['min_generated_tokens'],
        whitelist_token_ids={128821, 128822}
    )
]
```

## Additional Resources

- See `process/prompt_utils.py` for all available utility functions
- See `process/ngram_norepeat.py` for updated NoRepeatNGramLogitsProcessor implementation
- Check the main README.md for general usage examples

## Contributing

If you encounter issues with specific prompts, please report them with:
1. The exact prompt you used
2. The unexpected output
3. Your configuration (ngram_size, window_size, etc.)

This helps us improve the prompt handling and validation utilities.
