# Repetition Prevention for HuggingFace Transformers

## Problem

The DeepSeek-OCR model can sometimes enter infinite loops generating repetitive patterns like:
```
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
```

This typically happens on certain document pages and can consume all available tokens without producing useful output.

## Solution

This directory includes `ngram_norepeat.py` which provides the `NoRepeatNGramLogitsProcessor` to prevent such repetition.

## Usage

### Option 1: Using the Updated run_dpsk_ocr.py

The `run_dpsk_ocr.py` script has been updated to include repetition prevention. Simply run it as usual:

```python
python run_dpsk_ocr.py
```

### Option 2: Manual Integration

If you're using the model directly in your own code, integrate the processor as follows:

```python
from transformers import AutoModel, AutoTokenizer
import torch
from ngram_norepeat import NoRepeatNGramLogitsProcessor

# Load model
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# Create repetition prevention processor
logits_processor = NoRepeatNGramLogitsProcessor(
    ngram_size=30,           # Size of patterns to track
    window_size=90,          # How far back to look
    whitelist_token_ids={128821, 128822},  # <td>, </td> tags
    min_ngram_size=2,        # Minimum pattern size
    max_consecutive_repeats=3  # Max consecutive identical tokens
)

# Use with generation (if model.infer supports it)
res = model.infer(
    tokenizer, 
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    image_file='your_image.jpg',
    output_path='output/',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    logits_processor=logits_processor  # Pass the processor
)
```

### Option 3: Using with model.generate()

If you're using the standard `model.generate()` method:

```python
from transformers import GenerationConfig
from ngram_norepeat import NoRepeatNGramLogitsProcessor

# Create processor
logits_processor = NoRepeatNGramLogitsProcessor(
    ngram_size=30,
    window_size=90,
    whitelist_token_ids={128821, 128822}
)

# Generate with repetition prevention
outputs = model.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    max_new_tokens=8192,
    temperature=0.0,
    logits_processor=[logits_processor],  # Pass as list
    do_sample=False
)
```

## Configuration Parameters

### ngram_size (default: 30)
- Size of n-grams to track for repetition
- Larger values catch longer repetitive patterns
- Recommended: 20-40 for document OCR

### window_size (default: 90)
- How many tokens back to search for repeated n-grams
- Larger values use more memory but catch distant repetitions
- Recommended: 50-100

### whitelist_token_ids (default: {128821, 128822})
- Token IDs that are allowed to repeat
- Useful for structural elements like table tags
- Common whitelist tokens:
  - 128821: `<td>` (table cell start)
  - 128822: `</td>` (table cell end)

### min_ngram_size (default: 2)
- Minimum pattern size for short pattern detection
- Helps catch patterns like ". . . ."
- Recommended: 2-3

### max_consecutive_repeats (default: 3)
- Maximum allowed consecutive identical tokens
- Prevents immediate repetition like "the the the the"
- Recommended: 3-5

## How It Works

The processor prevents repetition through three mechanisms:

1. **N-gram Blocking**: Tracks n-grams of size `ngram_size` within a `window_size` and prevents exact repetition

2. **Consecutive Token Detection**: Blocks tokens that have appeared consecutively more than `max_consecutive_repeats` times

3. **Short Pattern Detection**: Identifies and blocks short repetitive patterns (e.g., "A B A B A B")

## Troubleshooting

### Issue: Model still generates repetitive output

**Solution**: Increase `ngram_size` or decrease `max_consecutive_repeats`:
```python
logits_processor = NoRepeatNGramLogitsProcessor(
    ngram_size=40,  # Increased from 30
    max_consecutive_repeats=2  # Decreased from 3
)
```

### Issue: Model output seems constrained or unnatural

**Solution**: Decrease `ngram_size` or increase `window_size`:
```python
logits_processor = NoRepeatNGramLogitsProcessor(
    ngram_size=20,  # Decreased from 30
    window_size=120  # Increased from 90
)
```

### Issue: Table tags are being blocked

**Solution**: Add table-related token IDs to the whitelist:
```python
# Find token IDs
td_start = tokenizer.convert_tokens_to_ids('<td>')
td_end = tokenizer.convert_tokens_to_ids('</td>')
tr_start = tokenizer.convert_tokens_to_ids('<tr>')
tr_end = tokenizer.convert_tokens_to_ids('</tr>')

logits_processor = NoRepeatNGramLogitsProcessor(
    ngram_size=30,
    whitelist_token_ids={td_start, td_end, tr_start, tr_end}
)
```

## Note on Model Compatibility

The `model.infer()` method is defined in the remote model code (loaded via `trust_remote_code=True`). 

If the method doesn't support the `logits_processor` parameter, you may need to:

1. Use the vLLM inference path instead (see `../DeepSeek-OCR-vllm/`)
2. Modify the model's remote code to accept logits processors
3. Use the standard `model.generate()` method instead of `model.infer()`

## Related Files

- `ngram_norepeat.py`: The logits processor implementation
- `run_dpsk_ocr.py`: Example usage script
- `../DeepSeek-OCR-vllm/process/ngram_norepeat.py`: vLLM version (slightly different API)
- `../DeepSeek-OCR-vllm/process/repetition_detector.py`: Additional repetition detection utilities

## References

- GitHub Issue #250: https://github.com/deepseek-ai/DeepSeek-OCR/issues/250
- HuggingFace LogitsProcessor: https://huggingface.co/docs/transformers/internal/generation_utils#logitsprocessor
