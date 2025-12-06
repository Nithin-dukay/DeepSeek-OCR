# DeepSeek-OCR vLLM Mode Selection Guide

## Overview

The vLLM implementation of DeepSeek-OCR now supports multiple inference modes, allowing you to choose the optimal balance between speed, memory usage, and OCR quality based on your specific needs.

## Available Modes

| Mode | Resolution | Vision Tokens | Crop Mode | Best For |
|------|-----------|---------------|-----------|----------|
| **tiny** | 512×512 | 64 | No | Quick processing, limited GPU memory, simple documents |
| **small** | 640×640 | 100 | No | Fast processing, low memory usage, general documents |
| **base** | 1024×1024 | 256 | No | Balanced performance, standard documents |
| **large** | 1280×1280 | 400 | No | High-quality OCR, complex layouts, sufficient GPU memory |
| **gundam** | 1024 + n×640×640 | Dynamic | Yes | Best quality, adaptive to image size, fine details (default) |

## Configuration Methods

### Method 1: Edit config.py (Persistent Setting)

Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

```python
# Set your preferred mode
MODE = 'base'  # Options: 'tiny', 'small', 'base', 'large', 'gundam'
```

This setting will be used by default for all inference scripts.

### Method 2: Command-Line Arguments (Runtime Override)

Override the mode at runtime without modifying config.py:

```bash
# Image inference
python run_dpsk_ocr_image.py --mode base

# PDF inference
python run_dpsk_ocr_pdf.py --mode small

# Batch evaluation
python run_dpsk_ocr_eval_batch.py --mode large
```

## Usage Examples

### Example 1: Quick Document Processing (Small Mode)

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Process a simple document quickly
python run_dpsk_ocr_image.py \
    --mode small \
    --input /path/to/document.jpg \
    --output /path/to/output \
    --prompt "<image>\n<|grounding|>Convert the document to markdown."
```

**Use case:** Simple documents, receipts, forms where speed is more important than capturing fine details.

### Example 2: High-Quality OCR (Large Mode)

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Process a complex document with high quality
python run_dpsk_ocr_image.py \
    --mode large \
    --input /path/to/complex_document.jpg \
    --output /path/to/output
```

**Use case:** Complex layouts, small fonts, documents requiring high accuracy.

### Example 3: Adaptive Processing (Gundam Mode - Default)

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Process with adaptive tiling for best quality
python run_dpsk_ocr_image.py \
    --mode gundam \
    --input /path/to/large_document.jpg \
    --output /path/to/output
```

**Use case:** Large documents, mixed content, when you want the best possible quality and can afford the processing time.

### Example 4: Batch Processing with Memory Constraints

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Process multiple images with limited GPU memory
python run_dpsk_ocr_eval_batch.py \
    --mode tiny \
    --input /path/to/images_directory \
    --output /path/to/output
```

**Use case:** Processing many documents on limited hardware, preliminary analysis.

### Example 5: PDF Processing

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Process a PDF with balanced settings
python run_dpsk_ocr_pdf.py \
    --mode base \
    --input /path/to/document.pdf \
    --output /path/to/output
```

**Use case:** Multi-page PDFs with standard formatting.

## Performance Comparison

### Speed (approximate, on A100-40G)

- **tiny**: ~3000 tokens/s
- **small**: ~2800 tokens/s
- **base**: ~2500 tokens/s
- **large**: ~2000 tokens/s
- **gundam**: ~2500 tokens/s (varies with image size)

### Memory Usage (approximate)

- **tiny**: ~2GB VRAM per image
- **small**: ~3GB VRAM per image
- **base**: ~6GB VRAM per image
- **large**: ~10GB VRAM per image
- **gundam**: ~8-15GB VRAM per image (varies with tiles)

### Quality

- **tiny**: Good for simple text
- **small**: Good for standard documents
- **base**: Very good for most documents
- **large**: Excellent for complex documents
- **gundam**: Best quality, especially for large/complex documents

## Mode Selection Guidelines

### Choose **tiny** when:
- ✓ Processing simple documents with large, clear text
- ✓ GPU memory is very limited (< 4GB)
- ✓ Speed is the top priority
- ✓ Doing preliminary analysis or testing

### Choose **small** when:
- ✓ Processing standard documents
- ✓ GPU memory is limited (4-8GB)
- ✓ Need a good balance of speed and quality
- ✓ Text is reasonably sized and clear

### Choose **base** when:
- ✓ Processing general documents
- ✓ GPU memory is adequate (8-16GB)
- ✓ Need good quality without excessive processing time
- ✓ This is the recommended starting point for most use cases

### Choose **large** when:
- ✓ Processing complex documents with small text
- ✓ GPU memory is sufficient (16GB+)
- ✓ Quality is more important than speed
- ✓ Documents have intricate layouts or fine details

### Choose **gundam** when:
- ✓ Need the absolute best quality
- ✓ Processing large documents or images
- ✓ Documents have varying levels of detail
- ✓ GPU memory is sufficient (16GB+)
- ✓ Can afford longer processing time

## Advanced Configuration

### Manual Override (Advanced Users)

After setting MODE in config.py, you can still manually override individual parameters:

```python
# In config.py
MODE = 'base'  # This sets BASE_SIZE=1024, IMAGE_SIZE=1024, CROP_MODE=False

# Manual override (uncomment to use)
# BASE_SIZE = 1280
# IMAGE_SIZE = 1024
# CROP_MODE = True
```

### Adjusting Concurrency

For batch processing, adjust `MAX_CONCURRENCY` in config.py based on your GPU memory:

```python
# For tiny/small modes with limited memory
MAX_CONCURRENCY = 50

# For base mode
MAX_CONCURRENCY = 100

# For large/gundam modes with ample memory
MAX_CONCURRENCY = 20
```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Switch to a smaller mode (large → base → small → tiny)
2. Reduce `MAX_CONCURRENCY` in config.py
3. Reduce `gpu_memory_utilization` in the inference scripts

### Poor Quality Results

If OCR quality is insufficient:
1. Switch to a larger mode (tiny → small → base → large → gundam)
2. Ensure input images are high resolution
3. Try gundam mode for adaptive processing

### Slow Processing

If processing is too slow:
1. Switch to a faster mode (gundam → large → base → small → tiny)
2. Increase `MAX_CONCURRENCY` if memory allows
3. Consider processing in batches

## FAQ

**Q: Which mode should I use by default?**  
A: Start with `base` mode for most documents. Use `gundam` if quality is critical, or `small` if speed is important.

**Q: Can I change modes without restarting?**  
A: Yes! Use command-line arguments to override the mode for each run without editing config.py.

**Q: Does mode affect the model weights?**  
A: No, all modes use the same model weights. They only change the input image resolution and processing strategy.

**Q: Can I use different modes for different images in batch processing?**  
A: Currently, one mode applies to all images in a batch. Process different image types separately with appropriate modes.

**Q: How do I know which mode was used?**  
A: The mode and its description are printed at the start of each inference run.

## Support

For issues or questions about mode selection:
- Check the main README.md for general setup
- Review this guide for mode-specific information
- Open an issue on GitHub with your mode configuration and error details
