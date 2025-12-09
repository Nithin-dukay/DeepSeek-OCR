# DeepSeek-OCR vLLM Mode Selection - Quick Reference

## 🚀 Quick Start

```bash
# Default mode (gundam)
python run_dpsk_ocr_image.py

# Select a specific mode
python run_dpsk_ocr_image.py --mode base
```

## 📋 Available Modes

| Mode | Command | Resolution | Best For |
|------|---------|-----------|----------|
| **tiny** | `--mode tiny` | 512×512 | Simple text, fast processing |
| **small** | `--mode small` | 640×640 | Standard documents, balanced |
| **base** | `--mode base` | 1024×1024 | Complex documents, tables |
| **large** | `--mode large` | 1280×1280 | Detailed content, high quality |
| **gundam** | `--mode gundam` | Dynamic | Large/complex documents (default) |

## 💡 Common Commands

### Image Processing
```bash
# Fast processing
python run_dpsk_ocr_image.py --mode tiny --input image.jpg

# High quality
python run_dpsk_ocr_image.py --mode base --input image.jpg

# Custom output
python run_dpsk_ocr_image.py --mode small --input image.jpg --output ./results
```

### PDF Processing
```bash
# Standard PDF
python run_dpsk_ocr_pdf.py --mode small --input doc.pdf

# High quality PDF
python run_dpsk_ocr_pdf.py --mode base --input doc.pdf --output ./results
```

### Batch Processing
```bash
# Batch evaluation
python run_dpsk_ocr_eval_batch.py --mode base --input ./images --output ./results
```

## 🔧 Advanced Options

```bash
# Custom parameters
python run_dpsk_ocr_image.py --base-size 1024 --image-size 640 --crop-mode true

# Custom prompt
python run_dpsk_ocr_image.py --mode base --prompt "<image>\nFree OCR."

# Full customization
python run_dpsk_ocr_pdf.py \
    --mode small \
    --input document.pdf \
    --output ./results \
    --prompt "<image>\n<|grounding|>Convert the document to markdown."
```

## 📊 Mode Selection Guide

**Choose tiny** → Speed is critical, simple documents  
**Choose small** → Balanced performance, most documents  
**Choose base** → Quality matters, complex layouts  
**Choose large** → Maximum quality, detailed content  
**Choose gundam** → Adaptive, large/variable documents  

## 🎯 Arguments Reference

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--mode` | string | Select preset mode | `--mode base` |
| `--base-size` | int | Override base size | `--base-size 1024` |
| `--image-size` | int | Override image size | `--image-size 640` |
| `--crop-mode` | bool | Enable crop mode | `--crop-mode true` |
| `--input` | path | Input file/directory | `--input image.jpg` |
| `--output` | path | Output directory | `--output ./results` |
| `--prompt` | text | Custom OCR prompt | `--prompt "<image>\nOCR."` |

## 🐛 Troubleshooting

**Out of memory?** → Use smaller mode (`--mode small` or `--mode tiny`)  
**Poor quality?** → Use larger mode (`--mode base` or `--mode large`)  
**Too slow?** → Use smaller mode or disable crop (`--crop-mode false`)  

## 📚 More Information

- See `MODE_SELECTION_GUIDE.md` for detailed documentation
- See `README.md` for installation and setup
- See `config.py` for all configuration options

## 🔄 Migration from Old Version

**Before (editing config.py):**
```python
BASE_SIZE = 1024
IMAGE_SIZE = 1024
CROP_MODE = False
```

**After (using command-line):**
```bash
python run_dpsk_ocr_image.py --mode base
```

---

**Need help?** Check the full documentation in `MODE_SELECTION_GUIDE.md`
