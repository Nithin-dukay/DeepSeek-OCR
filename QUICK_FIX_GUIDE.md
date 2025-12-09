# Quick Fix Guide for Issue #288

## TL;DR

If your model outputs repeated numbers or gibberish after modifying the prompt, use this quick fix:

## Quick Solution

### For vLLM Users:

```python
from process.prompt_utils import get_recommended_ngram_params
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Your custom prompt
prompt = "<image>\n<|grounding|>Your custom instruction here."

# Get recommended parameters
params = get_recommended_ngram_params(prompt)

# Create processor with proper settings
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=params['ngram_size'],
        window_size=params['window_size'],
        min_generated_tokens=params['min_generated_tokens'],
        whitelist_token_ids={128821, 128822}
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)
```

### For Transformers Users:

```python
from process.prompt_utils import create_safe_prompt

# Create a safe prompt
prompt = create_safe_prompt(
    base_instruction="Your instruction here",
    task_type="grounding"  # or "free_ocr" or "description"
)

# Use with model
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, ...)
```

## Recommended Prompts

Instead of custom prompts, use these tested templates:

```python
from process.prompt_utils import get_prompt_template

# Document to markdown
prompt = get_prompt_template("document_to_markdown")

# OCR image
prompt = get_prompt_template("ocr_image")

# Free OCR
prompt = get_prompt_template("free_ocr")
```

## What Changed?

The `NoRepeatNGramLogitsProcessor` now:
- Waits before blocking tokens (doesn't interfere with initial generation)
- Only analyzes generated tokens (ignores the prompt)
- Only bans tokens that actually repeat (not single occurrences)

## Need More Help?

- See `PROMPT_GUIDELINES.md` for detailed documentation
- See `ISSUE_288_FIX_SUMMARY.md` for technical details
- Run `python test_prompt_utils.py` to verify your setup
