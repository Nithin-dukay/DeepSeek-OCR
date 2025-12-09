# Quick Start: Mode Selection

## 🚀 Quick Mode Selection

### Command Line (Easiest)

```bash
# Image processing
python run_dpsk_ocr_image_with_mode.py --image FILE --mode MODE

# PDF processing  
python run_dpsk_ocr_pdf_with_mode.py --pdf FILE --mode MODE
```

### Available Modes

| Mode | Command | Best For |
|------|---------|----------|
| 🏃 **tiny** | `--mode tiny` | Quick previews |
| 📄 **small** | `--mode small` | Receipts, simple docs |
| 📋 **base** | `--mode base` | Standard documents (default) |
| 📚 **large** | `--mode large` | High-quality documents |
| 🤖 **gundam** | `--mode gundam` | Complex/large documents |

## 📝 Examples

### Process an image with Gundam mode
```bash
python run_dpsk_ocr_image_with_mode.py \\
    --image document.jpg \\
    --mode gundam \\
    --output ./results \\
    --save-results
```

### Process a PDF with Base mode
```bash
python run_dpsk_ocr_pdf_with_mode.py \\
    --pdf report.pdf \\
    --mode base \\
    --output ./results
```

### Programmatic usage
```python
from process.image_process import DeepseekOCRProcessor
from PIL import Image

processor = DeepseekOCRProcessor()
image = Image.open("doc.jpg").convert('RGB')

# Gundam mode
features = processor.tokenize_with_images(
    images=[image], bos=True, eos=True,
    base_size=1024, image_size=640, crop_mode=True
)
```

## 🎯 Mode Selection Guide

**Choose based on your needs:**

- **Speed priority** → tiny or small
- **Quality priority** → large or gundam  
- **Balanced** → base (recommended default)
- **Large documents** → gundam
- **Simple text** → small or tiny

## 📊 Performance vs Quality

```
Quality:  tiny < small < base < large ≈ gundam
Speed:    gundam < large < base < small < tiny
Tokens:   tiny(64) < small(100) < base(256) < large(400) < gundam(256+)
```

## 🔧 Configuration

### Mode Parameters

```python
# Tiny
base_size=512, image_size=512, crop_mode=False

# Small  
base_size=640, image_size=640, crop_mode=False

# Base
base_size=1024, image_size=1024, crop_mode=False

# Large
base_size=1280, image_size=1280, crop_mode=False

# Gundam (dynamic)
base_size=1024, image_size=640, crop_mode=True
```

## 💡 Tips

1. **Start with base mode** - good balance for most documents
2. **Use gundam for complex layouts** - tables, multi-column, etc.
3. **Use small for receipts** - faster and sufficient quality
4. **Use large for small text** - better recognition of tiny fonts
5. **Batch similar documents** - use same mode for efficiency

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Use smaller mode (tiny/small) |
| Poor quality | Use larger mode (large/gundam) |
| Too slow | Use smaller mode or reduce concurrency |
| Text not detected | Try gundam mode with crop_mode=True |

## 📚 More Information

- Full guide: `MODE_SELECTION_GUIDE.md`
- Main README: `../README.md`
- GitHub Issue: #164
