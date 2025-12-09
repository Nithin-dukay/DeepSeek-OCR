# DeepSeek-OCR vLLM Mode Selection Guide

This guide explains how to use dynamic mode selection with DeepSeek-OCR when deployed via vLLM.

## Overview

DeepSeek-OCR supports multiple modes that balance speed and quality:

| Mode | Resolution | Vision Tokens | Best For |
|------|-----------|---------------|----------|
| **Tiny** | 512×512 | 64 | Quick previews, simple text |
| **Small** | 640×640 | 100 | Simple documents, receipts |
| **Base** | 1024×1024 | 256 | Standard documents, forms |
| **Large** | 1280×1280 | 400 | High-quality documents |
| **Gundam** | Dynamic | 256 + n×100 | Large documents, complex layouts |

## Mode Selection Methods

### Method 1: Config-based (Traditional)

Edit `config.py` to set the mode globally:

```python
# Tiny mode
BASE_SIZE = 512
IMAGE_SIZE = 512
CROP_MODE = False

# Small mode
BASE_SIZE = 640
IMAGE_SIZE = 640
CROP_MODE = False

# Base mode
BASE_SIZE = 1024
IMAGE_SIZE = 1024
CROP_MODE = False

# Large mode
BASE_SIZE = 1280
IMAGE_SIZE = 1280
CROP_MODE = False

# Gundam mode (dynamic cropping)
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
```

Then run the standard scripts:
```bash
python run_dpsk_ocr_image.py
python run_dpsk_ocr_pdf.py
```

### Method 2: Dynamic Mode Selection (New!)

Use the new scripts with `--mode` parameter to select mode per request:

#### Image Processing

```bash
# Tiny mode (fastest, lowest quality)
python run_dpsk_ocr_image_with_mode.py \\
    --image your_image.jpg \\
    --mode tiny \\
    --output ./output \\
    --save-results

# Small mode
python run_dpsk_ocr_image_with_mode.py \\
    --image your_image.jpg \\
    --mode small \\
    --output ./output \\
    --save-results

# Base mode (balanced)
python run_dpsk_ocr_image_with_mode.py \\
    --image your_image.jpg \\
    --mode base \\
    --output ./output \\
    --save-results

# Large mode (high quality)
python run_dpsk_ocr_image_with_mode.py \\
    --image your_image.jpg \\
    --mode large \\
    --output ./output \\
    --save-results

# Gundam mode (dynamic, best for complex layouts)
python run_dpsk_ocr_image_with_mode.py \\
    --image your_image.jpg \\
    --mode gundam \\
    --output ./output \\
    --save-results
```

#### PDF Processing

```bash
# Process PDF with Gundam mode
python run_dpsk_ocr_pdf_with_mode.py \\
    --pdf your_document.pdf \\
    --mode gundam \\
    --output ./output

# Process PDF with Base mode
python run_dpsk_ocr_pdf_with_mode.py \\
    --pdf your_document.pdf \\
    --mode base \\
    --output ./output \\
    --max-concurrency 50 \\
    --num-workers 32
```

### Method 3: Programmatic API

Use the processor directly in your Python code:

```python
from process.image_process import DeepseekOCRProcessor
from PIL import Image

# Initialize processor
processor = DeepseekOCRProcessor()

# Load image
image = Image.open("your_image.jpg").convert('RGB')

# Process with Tiny mode
image_features_tiny = processor.tokenize_with_images(
    images=[image], 
    bos=True, 
    eos=True, 
    base_size=512,
    image_size=512,
    crop_mode=False
)

# Process with Small mode
image_features_small = processor.tokenize_with_images(
    images=[image], 
    bos=True, 
    eos=True, 
    base_size=640,
    image_size=640,
    crop_mode=False
)

# Process with Base mode
image_features_base = processor.tokenize_with_images(
    images=[image], 
    bos=True, 
    eos=True, 
    base_size=1024,
    image_size=1024,
    crop_mode=False
)

# Process with Large mode
image_features_large = processor.tokenize_with_images(
    images=[image], 
    bos=True, 
    eos=True, 
    base_size=1280,
    image_size=1280,
    crop_mode=False
)

# Process with Gundam mode (dynamic cropping)
image_features_gundam = processor.tokenize_with_images(
    images=[image], 
    bos=True, 
    eos=True, 
    base_size=1024,
    image_size=640,
    crop_mode=True
)

# Use with vLLM
from vllm import LLM, SamplingParams

llm = LLM(model="deepseek-ai/DeepSeek-OCR", ...)
sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)

request = {
    "prompt": "<image>\\n<|grounding|>Convert the document to markdown.",
    "multi_modal_data": {"image": image_features_gundam}
}

outputs = llm.generate([request], sampling_params)
print(outputs[0].outputs[0].text)
```

## Mode Selection Guidelines

### When to use each mode:

**Tiny Mode (512×512, 64 tokens)**
- ✅ Quick previews
- ✅ Simple text extraction
- ✅ Low-resolution images
- ✅ When speed is critical
- ❌ Complex layouts
- ❌ Small text

**Small Mode (640×640, 100 tokens)**
- ✅ Receipts
- ✅ Simple forms
- ✅ Business cards
- ✅ Good speed/quality balance
- ❌ Dense documents
- ❌ Complex tables

**Base Mode (1024×1024, 256 tokens)**
- ✅ Standard documents
- ✅ Forms with moderate complexity
- ✅ Most common use case
- ✅ Good balance of speed and quality
- ❌ Very large documents
- ❌ Extremely complex layouts

**Large Mode (1280×1280, 400 tokens)**
- ✅ High-quality documents
- ✅ Documents with small text
- ✅ Complex forms
- ✅ When quality is priority
- ❌ When speed is critical
- ❌ Simple documents (overkill)

**Gundam Mode (Dynamic, 256 + n×100 tokens)**
- ✅ Large documents (>1024px)
- ✅ Complex layouts
- ✅ Multi-column documents
- ✅ Documents with mixed content
- ✅ Best overall quality
- ❌ Small images (unnecessary overhead)
- ❌ When speed is critical

### Performance Comparison

Approximate processing times on A100-40G:

| Mode | Tokens/Image | Speed | Quality |
|------|-------------|-------|---------|
| Tiny | 64 | ~50ms | ⭐⭐ |
| Small | 100 | ~80ms | ⭐⭐⭐ |
| Base | 256 | ~150ms | ⭐⭐⭐⭐ |
| Large | 400 | ~250ms | ⭐⭐⭐⭐⭐ |
| Gundam | 256-1000+ | ~200-500ms | ⭐⭐⭐⭐⭐ |

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **Existing scripts work unchanged**: `run_dpsk_ocr_image.py` and `run_dpsk_ocr_pdf.py` continue to use `config.py` settings
2. **Default parameters**: If mode parameters are not specified, the processor falls back to `config.py` values
3. **Old parameter names**: The `cropping` parameter is still supported (mapped to `crop_mode`)

## Troubleshooting

### Issue: Mode parameters not taking effect

**Solution**: Make sure you're using the new scripts (`run_dpsk_ocr_image_with_mode.py`) or passing parameters explicitly in your code.

### Issue: Out of memory errors

**Solution**: Use a smaller mode (Tiny or Small) or reduce `max_concurrency` for batch processing.

### Issue: Poor quality results

**Solution**: Try a larger mode (Large or Gundam) for better quality, especially for complex documents.

### Issue: Slow processing

**Solution**: Use a smaller mode (Tiny or Small) for faster processing, or increase GPU memory allocation.

## Examples

### Example 1: Process multiple images with different modes

```python
from process.image_process import DeepseekOCRProcessor
from PIL import Image

processor = DeepseekOCRProcessor()

# Simple receipt - use Small mode
receipt = Image.open("receipt.jpg").convert('RGB')
receipt_features = processor.tokenize_with_images(
    images=[receipt], bos=True, eos=True,
    base_size=640, image_size=640, crop_mode=False
)

# Complex document - use Gundam mode
document = Image.open("complex_doc.jpg").convert('RGB')
doc_features = processor.tokenize_with_images(
    images=[document], bos=True, eos=True,
    base_size=1024, image_size=640, crop_mode=True
)
```

### Example 2: Batch processing with mode selection

```bash
# Process all PDFs in a directory with Gundam mode
for pdf in documents/*.pdf; do
    python run_dpsk_ocr_pdf_with_mode.py \\
        --pdf "$pdf" \\
        --mode gundam \\
        --output "./output/$(basename "$pdf" .pdf)"
done
```

### Example 3: Adaptive mode selection

```python
from PIL import Image
from process.image_process import DeepseekOCRProcessor

def select_mode_for_image(image):
    """Automatically select best mode based on image size."""
    width, height = image.size
    
    if width <= 640 and height <= 640:
        return {'base_size': 640, 'image_size': 640, 'crop_mode': False}
    elif width <= 1024 and height <= 1024:
        return {'base_size': 1024, 'image_size': 1024, 'crop_mode': False}
    else:
        # Use Gundam mode for large images
        return {'base_size': 1024, 'image_size': 640, 'crop_mode': True}

processor = DeepseekOCRProcessor()
image = Image.open("document.jpg").convert('RGB')
mode = select_mode_for_image(image)

features = processor.tokenize_with_images(
    images=[image], bos=True, eos=True, **mode
)
```

## FAQ

**Q: Can I mix different modes in the same batch?**
A: Currently, all images in a batch use the same mode. Process different modes separately.

**Q: Does mode selection affect the model weights?**
A: No, mode selection only affects image preprocessing. The same model weights are used.

**Q: Which mode should I use by default?**
A: Base mode (1024×1024) is recommended for most use cases. Use Gundam for complex/large documents.

**Q: Can I create custom modes?**
A: Yes! Just specify custom `base_size`, `image_size`, and `crop_mode` values. Recommended sizes are multiples of 16.

**Q: Does this work with the upstream vLLM version?**
A: This implementation is for the custom vLLM integration. For upstream vLLM, refer to the official documentation.

## Support

For issues or questions:
- GitHub Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- Reference Issue: #164 (Dynamic mode selection for vLLM)
