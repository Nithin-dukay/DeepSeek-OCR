# DeepSeek-OCR vLLM Mode Selection Guide

This guide explains how to use different OCR modes (Tiny, Small, Base, Large, Gundam) when deploying DeepSeek-OCR with vLLM.

## Available Modes

DeepSeek-OCR supports the following modes:

| Mode   | Resolution | Vision Tokens | Crop Mode | Use Case |
|--------|-----------|---------------|-----------|----------|
| **Tiny**   | 512×512   | 64            | No        | Fast processing, simple documents |
| **Small**  | 640×640   | 100           | No        | Balanced speed and quality |
| **Base**   | 1024×1024 | 256           | No        | Standard quality documents |
| **Large**  | 1280×1280 | 400           | No        | High-quality, detailed documents |
| **Gundam** | n×640×640 + 1×1024×1024 | Dynamic | Yes | Complex layouts, dynamic resolution |

## Method 1: Using config.py (Recommended for Fixed Mode)

Edit `config.py` and set the `MODE` variable:

```python
# In config.py
MODE = "gundam"  # Options: "tiny", "small", "base", "large", "gundam"
```

Then run your scripts normally:

```bash
python run_dpsk_ocr_image.py
python run_dpsk_ocr_pdf.py
python run_dpsk_ocr_eval_batch.py
```

## Method 2: Using Enhanced Scripts with Command-Line Arguments

For dynamic mode selection, use the enhanced scripts:

### For Image Processing

```bash
# Using Gundam mode (default)
python run_dpsk_ocr_image_with_mode.py --mode gundam --input image.jpg

# Using Small mode for faster processing
python run_dpsk_ocr_image_with_mode.py --mode small --input image.jpg

# Using Base mode with custom output directory
python run_dpsk_ocr_image_with_mode.py --mode base --input image.jpg --output ./results

# List all available modes
python run_dpsk_ocr_image_with_mode.py --list-modes
```

### For PDF Processing

```bash
# Using Gundam mode (default)
python run_dpsk_ocr_pdf_with_mode.py --mode gundam --input document.pdf

# Using Base mode for standard quality
python run_dpsk_ocr_pdf_with_mode.py --mode base --input document.pdf

# Using Large mode for high-quality documents
python run_dpsk_ocr_pdf_with_mode.py --mode large --input document.pdf --output ./pdf_results

# List all available modes
python run_dpsk_ocr_pdf_with_mode.py --list-modes
```

## Method 3: Programmatic Mode Selection

You can also use the mode configuration in your own Python scripts:

```python
from mode_config import OCRModeConfig, get_mode_params
from process.image_process import DeepseekOCRProcessor

# Method 1: Get full configuration
mode_config = OCRModeConfig.get_mode_config("gundam")
base_size = mode_config["base_size"]
image_size = mode_config["image_size"]
crop_mode = mode_config["crop_mode"]

# Method 2: Get parameters as tuple
base_size, image_size, crop_mode = get_mode_params("small")

# Use in image processing
processor = DeepseekOCRProcessor()
image_features = processor.tokenize_with_images(
    images=[image], 
    bos=True, 
    eos=True, 
    cropping=crop_mode
)

# List all available modes
OCRModeConfig.print_modes()
```

## Mode Selection Guidelines

### When to Use Each Mode

**Tiny Mode** (`tiny`)
- ✅ Simple text documents
- ✅ Fast processing required
- ✅ Limited GPU memory
- ❌ Complex layouts
- ❌ Small text

**Small Mode** (`small`)
- ✅ General purpose documents
- ✅ Good balance of speed and quality
- ✅ Moderate GPU memory
- ❌ Very detailed documents

**Base Mode** (`base`)
- ✅ Standard documents
- ✅ Good quality OCR
- ✅ Most common use case
- ❌ Very large or complex documents

**Large Mode** (`large`)
- ✅ High-quality documents
- ✅ Detailed text
- ✅ Professional documents
- ⚠️ Requires more GPU memory
- ⚠️ Slower processing

**Gundam Mode** (`gundam`) - **Recommended for Complex Documents**
- ✅ Complex layouts
- ✅ Mixed content (text, images, tables)
- ✅ Dynamic resolution adaptation
- ✅ Best quality for varied documents
- ⚠️ Highest GPU memory usage
- ⚠️ Slowest processing

## Performance Considerations

### GPU Memory Usage

| Mode   | Approximate VRAM | Batch Size Recommendation |
|--------|------------------|---------------------------|
| Tiny   | ~2-3 GB          | Large batches (50+)       |
| Small  | ~3-4 GB          | Medium batches (20-30)    |
| Base   | ~5-6 GB          | Small batches (10-15)     |
| Large  | ~8-10 GB         | Very small batches (5-8)  |
| Gundam | ~6-12 GB (varies)| Small batches (5-10)      |

### Processing Speed

Relative processing speed (Tiny = 1.0x baseline):

- **Tiny**: 1.0x (fastest)
- **Small**: 0.7x
- **Base**: 0.4x
- **Large**: 0.25x
- **Gundam**: 0.2-0.5x (varies with image complexity)

## Examples

### Example 1: Quick Document Scan

For quick scanning of simple documents:

```bash
python run_dpsk_ocr_image_with_mode.py --mode small --input receipt.jpg
```

### Example 2: High-Quality Academic Paper

For detailed academic papers with complex layouts:

```bash
python run_dpsk_ocr_pdf_with_mode.py --mode gundam --input research_paper.pdf
```

### Example 3: Batch Processing with Memory Constraints

For batch processing with limited GPU memory:

```bash
# Edit config.py
MODE = "small"
MAX_CONCURRENCY = 50

# Run batch processing
python run_dpsk_ocr_eval_batch.py
```

### Example 4: Custom Script with Mode Selection

```python
import argparse
from mode_config import get_mode_params
from process.image_process import DeepseekOCRProcessor
from PIL import Image

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='base', choices=['tiny', 'small', 'base', 'large', 'gundam'])
args = parser.parse_args()

# Get mode parameters
base_size, image_size, crop_mode = get_mode_params(args.mode)

print(f"Using {args.mode} mode: base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")

# Process image with selected mode
image = Image.open("document.jpg")
processor = DeepseekOCRProcessor()
features = processor.tokenize_with_images(
    images=[image],
    bos=True,
    eos=True,
    cropping=crop_mode
)
```

## Troubleshooting

### Issue: Out of Memory Error

**Solution**: Use a smaller mode or reduce batch size

```python
# In config.py
MODE = "small"  # or "tiny"
MAX_CONCURRENCY = 10  # Reduce from default
```

### Issue: Poor OCR Quality

**Solution**: Use a larger mode

```python
# In config.py
MODE = "large"  # or "gundam"
```

### Issue: Processing Too Slow

**Solution**: Use a faster mode or increase GPU memory utilization

```python
# In config.py
MODE = "small"  # Faster mode

# In your script
llm = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=0.95,  # Increase from 0.9
    ...
)
```

## Migration from Previous Versions

If you were previously manually setting `BASE_SIZE`, `IMAGE_SIZE`, and `CROP_MODE`:

**Before:**
```python
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
```

**After:**
```python
MODE = "gundam"  # Equivalent to the above settings
```

## API Reference

### OCRModeConfig Class

```python
from mode_config import OCRModeConfig

# Get mode configuration
config = OCRModeConfig.get_mode_config("gundam")
# Returns: {"base_size": 1024, "image_size": 640, "crop_mode": True, ...}

# Get individual parameters
base_size = OCRModeConfig.get_base_size("small")
image_size = OCRModeConfig.get_image_size("small")
crop_mode = OCRModeConfig.get_crop_mode("small")

# List all modes
modes = OCRModeConfig.list_modes()
# Returns: {"tiny": "Tiny mode: 512×512 (64 vision tokens)", ...}

# Print modes to console
OCRModeConfig.print_modes()
```

### Convenience Functions

```python
from mode_config import get_mode_params

# Get all parameters as tuple
base_size, image_size, crop_mode = get_mode_params("base")
```

## Additional Resources

- [DeepSeek-OCR GitHub Repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM Documentation](https://docs.vllm.ai/)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)

## Support

For issues or questions:
1. Check this guide first
2. Review the main README.md
3. Open an issue on GitHub
4. Join the Discord community

---

**Note**: This mode selection feature addresses GitHub Issue #164, enabling users to easily switch between different OCR modes when using vLLM deployment.
