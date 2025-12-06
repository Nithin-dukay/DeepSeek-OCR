# Quick Start: Mode Selection

## 🚀 Quick Mode Selection

### Choose Your Mode

| Mode | When to Use | Command |
|------|-------------|---------|
| **Tiny** | Quick testing, low memory | `--mode Tiny` |
| **Small** | Fast processing, simple docs | `--mode Small` |
| **Base** | General purpose (default) | `--mode Base` |
| **Large** | High quality images | `--mode Large` |
| **Gundam** | Best quality, complex docs | `--mode Gundam` |

## 📝 Two Ways to Set Mode

### Method 1: Edit config.py (Persistent)
```python
MODE = 'Gundam'  # Change this line
```

### Method 2: Command Line (One-time)
```bash
python run_dpsk_ocr_image.py --mode Base
```

## 💡 Common Use Cases

### Fast Testing
```bash
python run_dpsk_ocr_image.py --mode Tiny --input test.jpg
```

### Production Documents
```bash
python run_dpsk_ocr_pdf.py --mode Gundam --input document.pdf --output ./results
```

### Batch Processing
```bash
python run_dpsk_ocr_eval_batch.py --mode Base --input ./images --output ./results
```

### Low Memory GPU
```bash
python run_dpsk_ocr_image.py --mode Small --input image.jpg
```

## 🔍 Check Available Modes
```bash
python modes.py
```

## 📚 Need More Help?
- Detailed guide: `MODE_SELECTION_GUIDE.md`
- Code examples: `python example_mode_usage.py`
- Command help: `python run_dpsk_ocr_image.py --help`

## ⚡ Performance Tips

- **Out of Memory?** → Use smaller mode (Gundam → Large → Base → Small → Tiny)
- **Too Slow?** → Use smaller mode or increase MAX_CONCURRENCY
- **Poor Quality?** → Use larger mode (Tiny → Small → Base → Large → Gundam)

## 🎯 Mode Recommendations

| Your Situation | Recommended Mode |
|----------------|------------------|
| Testing/Prototyping | Tiny or Small |
| Production (balanced) | Base |
| High quality needed | Large or Gundam |
| Limited GPU memory | Tiny or Small |
| Complex documents | Gundam |
| Simple text extraction | Small or Base |

---
**Quick Tip:** Start with `Base` mode for most use cases, then adjust based on your needs!
