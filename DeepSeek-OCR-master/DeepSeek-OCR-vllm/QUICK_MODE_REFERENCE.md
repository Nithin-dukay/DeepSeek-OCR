# Quick Mode Reference Card

## Mode Selection Cheat Sheet

```bash
# Fastest (limited memory)
python run_dpsk_ocr_image.py --mode tiny

# Fast (low memory)
python run_dpsk_ocr_image.py --mode small

# Balanced (recommended starting point)
python run_dpsk_ocr_image.py --mode base

# High quality (more memory)
python run_dpsk_ocr_image.py --mode large

# Best quality (adaptive, default)
python run_dpsk_ocr_image.py --mode gundam
```

## Mode Specifications

| Mode | Resolution | Tokens | Memory | Speed | Quality |
|------|-----------|--------|--------|-------|---------|
| tiny | 512×512 | 64 | ~2GB | ⚡⚡⚡⚡⚡ | ⭐⭐ |
| small | 640×640 | 100 | ~3GB | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| base | 1024×1024 | 256 | ~6GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| large | 1280×1280 | 400 | ~10GB | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| gundam | dynamic | varies | ~8-15GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |

## Quick Decision Tree

```
Need best quality? → gundam
Limited GPU memory (< 8GB)? → small or tiny
Standard documents? → base
Complex documents with ample memory? → large
Batch processing many images? → small or base
```

## Configuration

### Persistent (config.py)
```python
MODE = 'base'  # Change this line
```

### Runtime (command-line)
```bash
--mode tiny|small|base|large|gundam
```

## Common Commands

```bash
# Image with mode override
python run_dpsk_ocr_image.py --mode base --input image.jpg --output ./out

# PDF with mode override
python run_dpsk_ocr_pdf.py --mode small --input doc.pdf --output ./out

# Batch with mode override
python run_dpsk_ocr_eval_batch.py --mode gundam --input ./imgs --output ./out
```

## Troubleshooting

- **OOM Error?** → Use smaller mode (large→base→small→tiny)
- **Poor Quality?** → Use larger mode (tiny→small→base→large→gundam)
- **Too Slow?** → Use faster mode (gundam→large→base→small→tiny)

---
For detailed information, see MODE_SELECTION_GUIDE.md
