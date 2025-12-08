# DeepSeek-OCR Compression Feature Guide

## Overview

This guide explains how to use the compression evaluation feature in DeepSeek-OCR, which was introduced in the paper to study vision encoder efficiency from an LLM-centric viewpoint.

## Table of Contents

- [What is Compression Evaluation?](#what-is-compression-evaluation)
- [The Issue (GitHub #285)](#the-issue-github-285)
- [Solution](#solution)
- [Usage Examples](#usage-examples)
- [Fox Dataset Evaluation](#fox-dataset-evaluation)
- [Understanding Compression Metrics](#understanding-compression-metrics)
- [Troubleshooting](#troubleshooting)

## What is Compression Evaluation?

The compression feature measures how efficiently the vision encoder compresses visual information into tokens that the LLM can process. It calculates:

- **Image Tokens**: Number of vision tokens used to represent the input image
- **Output Tokens**: Number of text tokens in the generated output
- **Compression Ratio**: `output_tokens / image_tokens`

A higher compression ratio indicates that the model generates more text tokens per vision token, suggesting efficient visual information encoding.

## The Issue (GitHub #285)

### Problem Description

When using `model.infer()` with `test_compress=True`, the method returns `None` instead of the generated output text. This happens because the original implementation in the HuggingFace model file (`modeling_deepseekocr.py`) prints compression statistics but doesn't return the output.

```python
# This returns None (BUG)
res = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    image_file="image.jpg",
    output_path="./output",
    test_compress=True  # Prints stats but returns None
)
print(res)  # None
```

### Root Cause

The `infer` method has three conditional return paths:
1. `eval_mode=True` → Returns output ✓
2. `test_compress=True` → Prints stats, **no return statement** ✗
3. `save_results=True` → Saves files, **no return statement** ✗

## Solution

We provide a utility module (`compression_utils.py`) that patches the model's `infer` method to fix this issue.

### Installation

The compression utilities are included in the `DeepSeek-OCR-hf` directory:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
```

### Quick Fix

```python
from transformers import AutoModel, AutoTokenizer
from compression_utils import patch_infer_method
import torch

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

# Apply the patch
model = patch_infer_method(model)

# Now test_compress=True works correctly
res = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    image_file="your_image.jpg",
    output_path="./output",
    test_compress=True,
    save_results=True
)

print(res)  # Now returns the output text!
```

## Usage Examples

### Example 1: Basic Compression Evaluation

```python
from transformers import AutoModel, AutoTokenizer
from compression_utils import patch_infer_method
import torch

# Setup
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)
model = patch_infer_method(model)

# Run inference with compression stats
result = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    image_file="document.jpg",
    output_path="./output",
    base_size=1024,
    image_size=640,
    crop_mode=True,
    test_compress=True
)

# Output will show:
# ==================================================
# image size:  (1920, 1080)
# valid image tokens:  456
# output texts tokens (valid):  1234
# compression ratio:  2.71
# ==================================================

print(f"Generated text: {result[:200]}...")
```

### Example 2: Get Compression Stats as Dictionary

```python
# Get both output and stats in a structured format
result = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    image_file="document.jpg",
    output_path="./output",
    base_size=1024,
    image_size=640,
    crop_mode=True,
    test_compress=True,
    return_stats=True  # Returns dict with 'output' and 'stats'
)

print(f"Output: {result['output'][:200]}...")
print(f"Image tokens: {result['stats']['image_tokens']}")
print(f"Output tokens: {result['stats']['output_tokens']}")
print(f"Compression ratio: {result['stats']['compression_ratio']}")
```

### Example 3: Batch Evaluation

```python
from compression_utils import (
    patch_infer_method,
    evaluate_compression_batch,
    save_compression_results,
    print_compression_summary
)

# Setup model
model = patch_infer_method(model)

# Prepare image list
image_files = [
    "images/doc1.jpg",
    "images/doc2.jpg",
    "images/doc3.jpg"
]

# Run batch evaluation
results = evaluate_compression_batch(
    model,
    tokenizer,
    image_files,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    output_dir="./compression_results",
    base_size=1024,
    image_size=640,
    crop_mode=True
)

# Save results
save_compression_results(
    results,
    output_file="compression_results.json",
    csv_file="compression_results.csv"
)

# Print summary
print_compression_summary(results)
```

## Fox Dataset Evaluation

The Fox dataset is mentioned in the DeepSeek-OCR paper for compression studies. Here's how to evaluate it:

### Setup

1. Download the Fox dataset from [https://github.com/ucaslcl/Fox](https://github.com/ucaslcl/Fox)
2. Organize images in a directory structure

### Evaluation Script

```python
from transformers import AutoModel, AutoTokenizer
from compression_utils import (
    patch_infer_method,
    evaluate_compression_batch,
    save_compression_results,
    print_compression_summary
)
import torch
import glob

# Load model
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)
model = patch_infer_method(model)

# Get all images from Fox dataset
fox_images = glob.glob("path/to/fox/dataset/**/*.jpg", recursive=True)
fox_images.extend(glob.glob("path/to/fox/dataset/**/*.png", recursive=True))

print(f"Found {len(fox_images)} images in Fox dataset")

# Evaluate with different resolution modes
modes = [
    {"name": "Tiny", "base_size": 512, "image_size": 512, "crop_mode": False},
    {"name": "Small", "base_size": 640, "image_size": 640, "crop_mode": False},
    {"name": "Base", "base_size": 1024, "image_size": 1024, "crop_mode": False},
    {"name": "Large", "base_size": 1280, "image_size": 1280, "crop_mode": False},
    {"name": "Gundam", "base_size": 1024, "image_size": 640, "crop_mode": True},
]

for mode in modes:
    print(f"\n{'='*80}")
    print(f"Evaluating mode: {mode['name']}")
    print(f"{'='*80}")
    
    results = evaluate_compression_batch(
        model,
        tokenizer,
        fox_images,
        prompt="<image>\n<|grounding|>Convert the document to markdown.",
        output_dir=f"./fox_results/{mode['name']}",
        base_size=mode['base_size'],
        image_size=mode['image_size'],
        crop_mode=mode['crop_mode']
    )
    
    save_compression_results(
        results,
        output_file=f"fox_results_{mode['name']}.json",
        csv_file=f"fox_results_{mode['name']}.csv"
    )
    
    print_compression_summary(results)
```

### Using the Provided Script

We provide a ready-to-use script for Fox dataset evaluation:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_compression_eval.py --dataset_path /path/to/fox/dataset --output_dir ./fox_results
```

## Understanding Compression Metrics

### Image Token Calculation

The number of image tokens depends on the resolution mode:

| Mode | Base Size | Image Size | Crop Mode | Tokens (approx) |
|------|-----------|------------|-----------|-----------------|
| Tiny | 512 | 512 | False | 64 |
| Small | 640 | 640 | False | 100 |
| Base | 1024 | 1024 | False | 256 |
| Large | 1280 | 1280 | False | 400 |
| Gundam | 1024 | 640 | True | 256 + n×100 (dynamic) |

**Gundam mode** uses dynamic resolution:
- 1 global view at 1024×1024 (256 tokens)
- n local views at 640×640 (100 tokens each)
- Total tokens = 256 + n×100

### Compression Ratio Interpretation

- **Ratio < 1.0**: Model generates fewer text tokens than image tokens (under-compression)
- **Ratio = 1.0**: Equal number of text and image tokens
- **Ratio > 1.0**: Model generates more text tokens than image tokens (efficient compression)
- **Ratio > 2.0**: High compression efficiency (typical for document OCR)

### Example Output

```
==================================================
image size:  (2480, 3508)
valid image tokens:  556
output texts tokens (valid):  2847
compression ratio:  5.12
==================================================
```

This shows:
- Input image: 2480×3508 pixels
- Vision encoder used: 556 tokens
- Generated output: 2847 text tokens
- Compression ratio: 5.12 (very efficient - generates ~5 text tokens per image token)

## Troubleshooting

### Issue: `model.infer()` returns `None`

**Cause**: Using `test_compress=True` without the patch.

**Solution**: Apply the patch using `compression_utils.patch_infer_method(model)`

### Issue: `return_stats` parameter not recognized

**Cause**: Using the original model without the patch.

**Solution**: Make sure to patch the model first:
```python
from compression_utils import patch_infer_method
model = patch_infer_method(model)
```

### Issue: CUDA out of memory

**Cause**: Processing large images or using Large/Gundam modes.

**Solutions**:
1. Use smaller resolution modes (Tiny, Small, Base)
2. Reduce batch size in batch evaluation
3. Process images sequentially instead of in parallel
4. Use `torch.cuda.empty_cache()` between images

### Issue: Compression ratio seems incorrect

**Cause**: Different prompt types generate different amounts of text.

**Solution**: Use consistent prompts for fair comparison:
- Document OCR: `"<image>\n<|grounding|>Convert the document to markdown."`
- General OCR: `"<image>\n<|grounding|>OCR this image."`
- Free OCR: `"<image>\nFree OCR."`

### Issue: No compression statistics printed

**Cause**: `test_compress=False` or using `eval_mode=True`.

**Solution**: Set `test_compress=True` and `eval_mode=False`:
```python
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    test_compress=True,  # Enable compression stats
    eval_mode=False      # Don't use eval mode
)
```

## Alternative Workarounds (Without Patch)

If you cannot use the patch, here are alternative approaches:

### Workaround 1: Use `eval_mode=True`

```python
# Get output without compression stats
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    eval_mode=True  # Returns output, no compression stats
)
```

### Workaround 2: Use `save_results=True` and read files

```python
# Save results to files
model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path="./output",
    save_results=True,
    test_compress=True  # Prints stats but returns None
)

# Read the saved output file
with open("./output/result.txt", "r") as f:
    result = f.read()
```

### Workaround 3: Call twice (once for output, once for stats)

```python
# First call: get output
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    eval_mode=True
)

# Second call: print compression stats
model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    test_compress=True
)
```

## Additional Resources

- **Paper**: [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)
- **Model**: [HuggingFace - deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Fox Dataset**: [https://github.com/ucaslcl/Fox](https://github.com/ucaslcl/Fox)
- **GitHub Issue**: [#285 - Query about Compression Study Evaluation](https://github.com/deepseek-ai/DeepSeek-OCR/issues/285)

## Contributing

If you find issues or have improvements for the compression utilities, please:
1. Open an issue on GitHub
2. Submit a pull request with your changes
3. Include test cases and documentation

## License

This guide and the compression utilities are provided under the same license as DeepSeek-OCR.
