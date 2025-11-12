# DeepSeek-OCR Examples

This directory contains practical examples for using DeepSeek-OCR with different types of documents, with a focus on minimizing hallucinations and maximizing accuracy.

## Overview

These examples address common issues when processing handwritten and historical documents, particularly the hallucination problem reported in GitHub Issue #191.

## Examples

### 1. Ancient Portuguese Handwritten Documents
**File:** `example_handwritten_portuguese.py`

Demonstrates optimal settings for ancient Portuguese handwritten documents:
- Image preprocessing for historical documents
- Grounded OCR mode to prevent hallucinations
- Multi-pass validation for accuracy
- Language-specific prompts

**Usage:**
```bash
python example_handwritten_portuguese.py
```

**Key Features:**
- Aggressive contrast and sharpness enhancement
- Uses `<|grounding|>` mode instead of "Free OCR"
- Preserves original spelling and diacritics
- Runs validation pass for comparison

### 2. Batch Processing
**File:** `example_batch_processing.py`

Process multiple handwritten documents efficiently:
- Batch processing of entire directories
- Consistent preprocessing across all images
- Progress tracking and error handling
- Detailed summary reports

**Usage:**
```bash
python example_batch_processing.py
```

**Key Features:**
- Processes all images in a directory
- Saves individual results for each image
- Generates batch summary JSON
- Handles errors gracefully

### 3. Region-Based OCR
**File:** `example_region_based_ocr.py`

Process complex documents by dividing them into regions:
- Multiple region strategies (grid, horizontal, vertical)
- Reduces hallucinations on large documents
- Visual region mapping
- Combines results intelligently

**Usage:**
```bash
python example_region_based_ocr.py
```

**Key Features:**
- Grid-based region division (2x2, 3x3)
- Horizontal/vertical strip processing
- Region visualization
- Combined output from all regions

## Quick Start

### Prerequisites

```bash
pip install transformers torch pillow numpy scipy
```

### Basic Usage Pattern

All examples follow this pattern:

1. **Load the model:**
```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
```

2. **Preprocess the image:**
```python
from PIL import Image, ImageEnhance

img = Image.open('document.jpg')
if img.mode != 'RGB':
    img = img.convert('RGB')

# Enhance for handwritten documents
contrast = ImageEnhance.Contrast(img)
img = contrast.enhance(1.5)

sharpness = ImageEnhance.Sharpness(img)
img = sharpness.enhance(1.3)
```

3. **Run OCR with grounded mode:**
```python
# CRITICAL: Use grounded mode, NOT "Free OCR"
prompt = "<image>\n<|grounding|>Extract all text from this handwritten document."

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file='document.jpg',
    output_path='output',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)
```

## Common Issues and Solutions

### Issue: Hallucinations on Handwritten Documents

**Problem:** Model generates text that doesn't exist in the image.

**Solution:**
1. ✅ Use `<|grounding|>` mode instead of "Free OCR"
2. ✅ Preprocess images (enhance contrast and sharpness)
3. ✅ Use conservative parameters (base_size=1024, image_size=640)
4. ✅ Enable crop_mode for better accuracy
5. ✅ Run multi-pass validation

**Example:**
```python
# ❌ WRONG - Causes hallucinations
prompt = "<image>\nFree OCR."

# ✅ CORRECT - Minimizes hallucinations
prompt = "<image>\n<|grounding|>Extract all text from this handwritten document."
```

### Issue: Poor Quality Historical Documents

**Problem:** Faded or degraded text is not recognized.

**Solution:**
1. Aggressive preprocessing (contrast=1.8, sharpness=1.5)
2. Optional denoising for very noisy documents
3. Use larger model size (base_size=1280)
4. Enable multi-pass validation

See `example_handwritten_portuguese.py` for implementation.

### Issue: Large Complex Documents

**Problem:** Entire document processing produces inconsistent results.

**Solution:**
1. Use region-based processing
2. Divide document into manageable sections
3. Process each region independently
4. Combine results intelligently

See `example_region_based_ocr.py` for implementation.

## Parameter Guide

### Model Size Parameters

| Configuration | base_size | image_size | crop_mode | Use Case |
|--------------|-----------|------------|-----------|----------|
| Tiny | 512 | 512 | False | Small, simple documents |
| Small | 640 | 640 | False | Medium documents |
| Base | 1024 | 1024 | False | Standard documents |
| Large | 1280 | 1280 | False | High-resolution documents |
| **Gundam (Recommended)** | **1024** | **640** | **True** | **Handwritten documents** |

### Preprocessing Parameters

| Parameter | Range | Recommended for Handwritten | Description |
|-----------|-------|----------------------------|-------------|
| enhance_contrast | 1.0-2.5 | 1.5-1.8 | Contrast enhancement factor |
| enhance_sharpness | 1.0-2.0 | 1.3-1.5 | Sharpness enhancement factor |
| denoise | True/False | False (use carefully) | Apply median filter |
| max_size | pixels | 2048 | Maximum dimension |

### Prompt Templates

#### Handwritten Documents
```python
prompt = "<image>\n<|grounding|>Extract all text from this handwritten document with precise positioning."
```

#### Historical Documents
```python
prompt = "<image>\n<|grounding|>Extract all text from this historical document, preserving original spelling and diacritics."
```

#### Language-Specific
```python
prompt = "<image>\n<|grounding|>Extract all Portuguese text from this handwritten document."
```

#### Tables and Forms
```python
prompt = "<image>\n<|grounding|>Extract text preserving table structure and convert to markdown."
```

## Advanced Techniques

### Multi-Pass Validation

Run OCR multiple times with slightly different parameters and compare results:

```python
# Pass 1: Primary
result1 = model.infer(tokenizer, prompt=prompt1, ...)

# Pass 2: Validation
result2 = model.infer(tokenizer, prompt=prompt2, ...)

# Compare and choose most consistent result
```

### Adaptive Preprocessing

Analyze image quality and adjust preprocessing:

```python
import numpy as np

img_array = np.array(img)
contrast_score = np.std(img_array) / 128.0

if contrast_score < 0.5:
    # Low contrast - aggressive enhancement
    enhance_factor = 2.0
else:
    # Good contrast - moderate enhancement
    enhance_factor = 1.3
```

### Region-Based Processing

For large or complex documents:

```python
# Divide into regions
regions = [
    (0.0, 0.0, 0.5, 0.5),  # Top-left
    (0.5, 0.0, 1.0, 0.5),  # Top-right
    (0.0, 0.5, 0.5, 1.0),  # Bottom-left
    (0.5, 0.5, 1.0, 1.0),  # Bottom-right
]

# Process each region
for region in regions:
    region_img = img.crop(region)
    result = model.infer(...)
```

## Troubleshooting

### Model Loading Issues

If you encounter Flash Attention errors:
```python
# Remove flash attention
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True
)
```

### CUDA Out of Memory

Reduce model size or image resolution:
```python
# Use smaller configuration
base_size = 640
image_size = 640

# Or resize image before processing
max_size = 1024
if max(img.size) > max_size:
    ratio = max_size / max(img.size)
    new_size = tuple(int(dim * ratio) for dim in img.size)
    img = img.resize(new_size, Image.Resampling.LANCZOS)
```

### Poor Results on Handwritten Text

1. Check if using grounded mode (`<|grounding|>`)
2. Increase preprocessing enhancement
3. Try region-based processing
4. Enable multi-pass validation
5. Ensure image quality is adequate (>1000px)

## Additional Resources

- [Troubleshooting Guide](../TROUBLESHOOTING_HANDWRITTEN_OCR.md) - Comprehensive troubleshooting
- [Configuration Helper](../DeepSeek-OCR-master/DeepSeek-OCR-hf/ocr_config_helper.py) - Auto-configuration tool
- [Optimized Script](../DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py) - Production-ready script

## Contributing

If you have additional examples or improvements, please contribute:

1. Follow the existing code style
2. Include comprehensive comments
3. Add usage examples in docstrings
4. Test with various document types

## License

These examples are provided under the same license as the DeepSeek-OCR project.
