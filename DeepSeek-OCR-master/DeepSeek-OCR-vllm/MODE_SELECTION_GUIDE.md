# DeepSeek-OCR vLLM Mode Selection Guide

## Overview

DeepSeek-OCR now supports dynamic mode selection when using vLLM deployment. This allows you to choose different processing modes based on your requirements for speed, quality, and GPU memory constraints.

## Available Modes

| Mode | Resolution | Vision Tokens | Crop Mode | Use Case |
|------|-----------|---------------|-----------|----------|
| **tiny** | 512×512 | 64 | No | Fast processing, low memory, simple documents |
| **small** | 640×640 | 100 | No | Balanced speed and quality for standard documents |
| **base** | 1024×1024 | 256 | No | High quality for complex documents |
| **large** | 1280×1280 | 400 | No | Maximum quality for detailed documents |
| **gundam** | n×640×640 + 1×1024×1024 | Dynamic | Yes | Dynamic resolution for large/complex documents (default) |

## Configuration Methods

### Method 1: Command-Line Arguments (Recommended)

Use command-line arguments to specify the mode when running scripts:

```bash
# Select a preset mode
python run_dpsk_ocr_image.py --mode base

# Override individual parameters
python run_dpsk_ocr_image.py --base-size 1024 --image-size 640 --crop-mode true

# Combine mode with other options
python run_dpsk_ocr_image.py --mode small --input image.jpg --output ./results
```

### Method 2: Edit config.py

Modify the `MODE` variable in `config.py`:

```python
# Default mode (can be overridden via command-line arguments)
MODE = 'base'  # Options: tiny, small, base, large, gundam
```

## Usage Examples

### Image Processing

```bash
# Fast processing with tiny mode
python run_dpsk_ocr_image.py --mode tiny --input document.jpg --output ./output

# High quality with base mode
python run_dpsk_ocr_image.py --mode base --input document.jpg --output ./output

# Dynamic resolution with gundam mode (default)
python run_dpsk_ocr_image.py --mode gundam --input large_document.jpg --output ./output
```

### PDF Processing

```bash
# Process PDF with small mode for speed
python run_dpsk_ocr_pdf.py --mode small --input document.pdf --output ./output

# Process PDF with base mode for quality
python run_dpsk_ocr_pdf.py --mode base --input document.pdf --output ./output

# Process PDF with custom parameters
python run_dpsk_ocr_pdf.py --base-size 1280 --image-size 1280 --crop-mode false \
    --input document.pdf --output ./output
```

### Batch Evaluation

```bash
# Evaluate with base mode
python run_dpsk_ocr_eval_batch.py --mode base --input ./images --output ./results

# Evaluate with gundam mode for dynamic resolution
python run_dpsk_ocr_eval_batch.py --mode gundam --input ./images --output ./results
```

## Command-Line Arguments Reference

### Mode Selection
- `--mode [tiny|small|base|large|gundam]`
  - Select a preset OCR mode
  - Example: `--mode base`

### Parameter Overrides
- `--base-size INT`
  - Override base size for image processing
  - Example: `--base-size 1024`

- `--image-size INT`
  - Override image size for cropping
  - Example: `--image-size 640`

- `--crop-mode [true|false]`
  - Enable or disable crop mode
  - Example: `--crop-mode true`

### Path Configuration
- `--input PATH`
  - Input file or directory path
  - Example: `--input /path/to/image.jpg`

- `--output PATH`
  - Output directory path
  - Example: `--output /path/to/results`

### Prompt Customization
- `--prompt TEXT`
  - Custom prompt for OCR
  - Example: `--prompt "<image>\nFree OCR."`

## Mode Selection Guidelines

### Choose **tiny** mode when:
- Processing simple documents with clear text
- GPU memory is limited
- Speed is the primary concern
- Processing large batches of simple documents

### Choose **small** mode when:
- Processing standard documents
- Balancing speed and quality
- GPU memory is moderate
- Good default for most use cases

### Choose **base** mode when:
- Processing complex documents with mixed content
- Quality is important
- GPU memory is sufficient
- Documents contain tables, formulas, or detailed layouts

### Choose **large** mode when:
- Processing highly detailed documents
- Maximum quality is required
- GPU memory is abundant
- Documents contain fine details or small text

### Choose **gundam** mode when:
- Processing documents with varying sizes
- Documents are large or have complex layouts
- Dynamic resolution is beneficial
- Default mode for best overall performance

## Performance Considerations

### GPU Memory Usage
- **tiny**: ~2-3 GB
- **small**: ~3-4 GB
- **base**: ~5-7 GB
- **large**: ~8-10 GB
- **gundam**: Variable (depends on document size)

### Processing Speed (approximate)
- **tiny**: Fastest (~3-5x faster than base)
- **small**: Fast (~2-3x faster than base)
- **base**: Standard (baseline)
- **large**: Slower (~0.6x base speed)
- **gundam**: Variable (depends on document complexity)

### Quality Trade-offs
- **tiny**: Good for simple text, may miss fine details
- **small**: Good balance for most documents
- **base**: High quality for complex documents
- **large**: Maximum quality, best for detailed content
- **gundam**: Adaptive quality based on document

## Troubleshooting

### Out of Memory Errors
If you encounter GPU memory errors:
1. Switch to a smaller mode (e.g., from `base` to `small`)
2. Reduce `MAX_CONCURRENCY` in config.py
3. Use `--crop-mode false` to disable dynamic cropping

### Poor Quality Results
If OCR quality is insufficient:
1. Switch to a larger mode (e.g., from `small` to `base`)
2. Enable crop mode: `--crop-mode true`
3. Use `gundam` mode for adaptive resolution

### Slow Processing
If processing is too slow:
1. Switch to a smaller mode (e.g., from `base` to `small`)
2. Increase `MAX_CONCURRENCY` in config.py (if GPU memory allows)
3. Use `tiny` mode for simple documents

## Examples by Document Type

### Simple Text Documents
```bash
python run_dpsk_ocr_image.py --mode small --input simple_text.jpg
```

### Documents with Tables
```bash
python run_dpsk_ocr_image.py --mode base --input table_document.jpg
```

### Mathematical Documents
```bash
python run_dpsk_ocr_image.py --mode base --input math_paper.jpg
```

### Large Multi-page PDFs
```bash
python run_dpsk_ocr_pdf.py --mode gundam --input large_document.pdf
```

### Scanned Documents
```bash
python run_dpsk_ocr_image.py --mode large --input scanned_page.jpg
```

## Migration from Previous Versions

If you were previously editing `config.py` to change modes, you can now use command-line arguments instead:

**Before:**
```python
# Edit config.py
BASE_SIZE = 1024
IMAGE_SIZE = 1024
CROP_MODE = False
```

**After:**
```bash
# Use command-line arguments
python run_dpsk_ocr_image.py --mode base
```

This approach is more flexible and doesn't require editing configuration files.

## Additional Resources

- See `config.py` for all available configuration options
- Check the main README.md for installation and setup instructions
- Refer to the DeepSeek-OCR paper for technical details on different modes
