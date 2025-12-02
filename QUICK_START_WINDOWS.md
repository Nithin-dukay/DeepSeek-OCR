# Quick Start Guide for Windows 11 Users

## TL;DR - Just Want It To Work?

Run this:
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr_windows.py
```

That's it! The script will automatically detect your hardware and configure everything.

---

## Step-by-Step Setup

### 1. Install Python (if not already installed)
- Download Python 3.11 from [python.org](https://www.python.org/downloads/)
- **Important:** Check "Add Python to PATH" during installation
- Python 3.13 may have compatibility issues - use 3.11 if possible

### 2. Install PyTorch

Open Command Prompt or PowerShell:

**If you have NVIDIA GPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**If you only have CPU:**
```bash
pip install torch torchvision torchaudio
```

### 3. Install Dependencies

```bash
cd path\to\DeepSeek-OCR
pip install -r requirements.txt
```

**Important:** Do NOT install flash-attn on Windows! Skip it if prompted.

### 4. Run the Model

```bash
cd DeepSeek-OCR-master\DeepSeek-OCR-hf
python run_dpsk_ocr_windows.py
```

### 5. Configure Your Image

Edit `run_dpsk_ocr_windows.py` and change:
```python
image_file = 'test.png'  # Change to your image path
output_path = 'output'   # Change to your output directory
```

---

## What Each Script Does

### `run_dpsk_ocr_windows.py` (Recommended)
- ✅ Automatically detects GPU or CPU
- ✅ Chooses best settings for your hardware
- ✅ Works with RTX 1060 and older GPUs
- ✅ Falls back to CPU if needed
- ✅ Provides helpful status messages

**Use this if:** You're on Windows and want it to "just work"

### `run_dpsk_ocr_cpu.py`
- ✅ Forces CPU-only mode
- ✅ Uses smaller image sizes for speed
- ✅ Good for testing without GPU

**Use this if:** You don't have a GPU or want to force CPU usage

### `run_dpsk_ocr.py` (Original)
- ❌ Requires flash-attention (doesn't work on Windows)
- ❌ Assumes CUDA is available
- ❌ Not recommended for Windows

**Use this if:** You're on Linux with a modern GPU and flash-attention installed

---

## Common Questions

### Q: Do I need flash-attention?
**A:** No! The model works perfectly without it. The Windows script doesn't use it.

### Q: Will it work with my RTX 1060?
**A:** Yes! The Windows script automatically handles older GPUs.

### Q: How slow is CPU mode?
**A:** 10-100x slower than GPU. A small image might take 2-5 minutes instead of 5-10 seconds.

### Q: Can I use Python 3.13?
**A:** It might work, but Python 3.11 is more stable. Downgrade if you have issues.

### Q: Why is my GPU not being detected?
**A:** Make sure you installed PyTorch with CUDA support (see step 2 above).

### Q: I'm getting errors!
**A:** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for solutions to common issues.

---

## Performance Tips

### For GPU Users:
- Close other GPU-intensive applications
- Use Gundam mode (default in Windows script) for best quality
- If you get "out of memory" errors, use smaller image sizes

### For CPU Users:
- Start with Tiny configuration (512x512)
- Close other applications to free RAM
- Be patient - it's slow but it works!
- Consider using a cloud GPU service for production

---

## Example Usage

### Basic OCR:
```python
prompt = "<image>\nFree OCR."
```

### Document to Markdown:
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```

### OCR with Layout:
```python
prompt = "<image>\n<|grounding|>OCR this image."
```

### Parse Figure:
```python
prompt = "<image>\nParse the figure."
```

---

## Need Help?

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Read the full [README.md](README.md)
3. Open an issue on [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
4. Join the [Discord](https://discord.gg/Tc7c45Zzu5)

---

## What's Different from the Original?

The original examples assume:
- ❌ Linux environment
- ❌ Modern GPU with flash-attention support
- ❌ CUDA always available

Our Windows scripts:
- ✅ Work on Windows 11
- ✅ Don't require flash-attention
- ✅ Auto-detect hardware
- ✅ Provide helpful error messages
- ✅ Work with older GPUs
- ✅ Have CPU fallback

---

## Success Checklist

Before running, make sure:
- [ ] Python 3.11 is installed
- [ ] PyTorch is installed (with CUDA if you have GPU)
- [ ] Dependencies are installed (`pip install -r requirements.txt`)
- [ ] You're in the correct directory (`DeepSeek-OCR-master/DeepSeek-OCR-hf`)
- [ ] Your image file exists and path is correct
- [ ] Output directory is writable

Then just run:
```bash
python run_dpsk_ocr_windows.py
```

Good luck! 🚀
