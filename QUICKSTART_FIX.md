# Quick Start: Fix Hallucinations in DeepSeek-OCR

## Problem
Getting repetitive or incorrect text when using DeepSeek-OCR with Transformers/HuggingFace API? This is especially common with handwritten or ancient documents.

## Solution (3 Steps)

### Step 1: Copy the Processor File
Copy `DeepSeek-OCR-master/DeepSeek-OCR-hf/ngram_norepeat_hf.py` to your project directory.

### Step 2: Update Your Code
Add these 4 lines to your existing code:

```python
# Add this import at the top
from ngram_norepeat_hf import NoRepeatNGramLogitsProcessor, get_recommended_params

# Add these lines before model.infer()
params = get_recommended_params("handwritten")  # Choose: handwritten, printed, table, or general
logits_processor = NoRepeatNGramLogitsProcessor(**params)

# Add this parameter to your model.infer() call
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path,
    # ... your other parameters ...
    logits_processor=logits_processor  # <-- ADD THIS LINE
)
```

### Step 3: Done!
Run your code. Hallucinations should be significantly reduced or eliminated.

## Complete Example

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os
from ngram_norepeat_hf import NoRepeatNGramLogitsProcessor, get_recommended_params

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# Configure hallucination prevention
params = get_recommended_params("handwritten")
logits_processor = NoRepeatNGramLogitsProcessor(**params)

# Run OCR
prompt = "<image>\nFree OCR. "
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file='your_image.jpg', 
    output_path='./output',
    base_size=1024, 
    image_size=640, 
    crop_mode=True,
    logits_processor=logits_processor  # Prevents hallucinations!
)
```

## Document Types

Choose the right type for your document:

- **`"handwritten"`** - Ancient manuscripts, handwritten text (most aggressive)
- **`"printed"`** - Modern printed documents (balanced)
- **`"table"`** - Documents with tables (allows table tag repetition)
- **`"general"`** - Default for mixed content

## Still Having Issues?

### Increase strictness:
```python
params = get_recommended_params("handwritten")
params["ngram_size"] = 40  # Higher = more strict
```

### Decrease strictness:
```python
params = get_recommended_params("printed")
params["ngram_size"] = 20  # Lower = less strict
```

## Need More Help?

- **Detailed docs**: See `ISSUE_191_FIX.md`
- **Examples**: See `fix_hallucination_example.py`
- **Tests**: Run `python3 test_hallucination_fix.py`

## That's It!
Your OCR should now work without hallucinations. 🎉
