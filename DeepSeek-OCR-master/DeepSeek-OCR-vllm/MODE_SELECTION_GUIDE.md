# DeepSeek-OCR Mode Selection Guide

This guide explains how to use different OCR modes when deploying DeepSeek-OCR with vLLM.

## Available Modes

DeepSeek-OCR supports five different modes, each optimized for different use cases:

| Mode | Resolution | Vision Tokens | Crop Mode | Best For |
|------|-----------|---------------|-----------|----------|
| **Tiny** | 512×512 | 64 | No | Quick previews, low-memory environments |
| **Small** | 640×640 | 100 | No | Fast processing, simple documents |
| **Base** | 1024×1024 | 256 | No | General purpose, balanced quality/speed |
| **Large** | 1280×1280 | 400 | No | High-quality single images |
| **Gundam** | 1024 base + 640 crops | Variable | Yes | Complex documents, best quality |

## How to Select a Mode

### Method 1: Configure in config.py (Recommended for Production)

Edit `config.py` and set the `MODE` parameter:

```python
# Set your preferred mode
MODE = 'Gundam'  # Options: Tiny, Small, Base, Large, Gundam
```

Then run your scripts normally:
```bash
python run_dpsk_ocr_image.py
python run_dpsk_ocr_pdf.py
python run_dpsk_ocr_eval_batch.py
```

### Method 2: Command-Line Arguments (Recommended for Testing)

Override the config.py settings using command-line arguments:

```bash
# Image processing with different modes
python run_dpsk_ocr_image.py --mode Tiny
python run_dpsk_ocr_image.py --mode Small
python run_dpsk_ocr_image.py --mode Base
python run_dpsk_ocr_image.py --mode Large
python run_dpsk_ocr_image.py --mode Gundam

# PDF processing with mode selection
python run_dpsk_ocr_pdf.py --mode Base --input document.pdf --output ./results

# Batch evaluation with mode selection
python run_dpsk_ocr_eval_batch.py --mode Small --input ./images --output ./results
```

## Mode Selection Guidelines

### When to use Tiny mode (512×512, 64 tokens)
- ✅ Quick testing and prototyping
- ✅ Low-memory GPU environments
- ✅ Simple text extraction from small images
- ❌ Not recommended for complex documents
- ❌ Not recommended for high-quality requirements

### When to use Small mode (640×640, 100 tokens)
- ✅ Fast batch processing
- ✅ Simple documents with clear text
- ✅ Resource-constrained environments
- ✅ Real-time applications
- ❌ Not ideal for dense or complex layouts

### When to use Base mode (1024×1024, 256 tokens)
- ✅ General-purpose OCR tasks
- ✅ Balanced quality and speed
- ✅ Standard document processing
- ✅ Good default choice for most use cases
- ⚠️ May miss fine details in very dense documents

### When to use Large mode (1280×1280, 400 tokens)
- ✅ High-quality single image processing
- ✅ Images with fine text details
- ✅ When quality is more important than speed
- ⚠️ Higher memory usage
- ⚠️ Slower processing

### When to use Gundam mode (Dynamic resolution)
- ✅ Complex multi-page documents
- ✅ Documents with mixed layouts
- ✅ Best quality requirements
- ✅ Production document processing
- ⚠️ Highest memory usage
- ⚠️ Slowest processing (but best quality)

## Performance Comparison

Approximate processing times on A100-40G (single image):

| Mode | Processing Time | Memory Usage | Quality |
|------|----------------|--------------|---------|
| Tiny | ~0.5s | ~2GB | ⭐⭐ |
| Small | ~0.8s | ~3GB | ⭐⭐⭐ |
| Base | ~1.5s | ~5GB | ⭐⭐⭐⭐ |
| Large | ~2.5s | ~8GB | ⭐⭐⭐⭐⭐ |
| Gundam | ~3-5s (varies) | ~10-15GB | ⭐⭐⭐⭐⭐⭐ |

*Note: Times are approximate and depend on image complexity and hardware.*

## Examples

### Example 1: Quick Testing with Tiny Mode
```bash
# Fast preview of OCR capabilities
python run_dpsk_ocr_image.py --mode Tiny --input sample.jpg --output ./preview
```

### Example 2: Production Document Processing with Gundam Mode
```bash
# Best quality for important documents
python run_dpsk_ocr_pdf.py --mode Gundam --input contract.pdf --output ./contracts
```

### Example 3: Batch Processing with Base Mode
```bash
# Balanced processing for large batches
python run_dpsk_ocr_eval_batch.py --mode Base --input ./dataset --output ./results
```

### Example 4: Memory-Constrained Environment
```bash
# Use Small mode when GPU memory is limited
python run_dpsk_ocr_pdf.py --mode Small --input document.pdf --output ./output
```

## Troubleshooting

### Out of Memory Errors
If you encounter OOM errors:
1. Try a smaller mode (Gundam → Large → Base → Small → Tiny)
2. Reduce `MAX_CONCURRENCY` in config.py
3. Reduce `gpu_memory_utilization` in the LLM initialization

### Quality Issues
If OCR quality is not satisfactory:
1. Try a larger mode (Tiny → Small → Base → Large → Gundam)
2. Ensure input images are high resolution
3. Use Gundam mode for complex documents

### Speed Issues
If processing is too slow:
1. Use a smaller mode for faster processing
2. Increase `MAX_CONCURRENCY` for batch processing
3. Consider using Base mode as a good balance

## Advanced: Custom Mode Configuration

For advanced users, you can manually configure mode parameters in `config.py`:

```python
# Option 1: Use predefined mode
MODE = 'Base'

# Option 2: Manual configuration (overrides MODE)
BASE_SIZE = 1024
IMAGE_SIZE = 1024
CROP_MODE = False
```

## Getting Help

To see all available command-line options:
```bash
python run_dpsk_ocr_image.py --help
python run_dpsk_ocr_pdf.py --help
python run_dpsk_ocr_eval_batch.py --help
```

To list all available modes programmatically:
```python
from modes import print_available_modes
print_available_modes()
```
