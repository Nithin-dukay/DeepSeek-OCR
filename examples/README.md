# DeepSeek-OCR Windows Examples

This directory contains example scripts for running DeepSeek-OCR on Windows 11, particularly for systems with limited GPU capabilities or without flash-attention support.

## Quick Start

### 1. Check Your System

First, run the system check script to understand your hardware capabilities:

```bash
python examples/check_system.py
```

This will tell you:
- Your GPU model and VRAM
- Whether flash-attention is supported
- Recommended settings for your system
- Which example script to use

### 2. Choose the Right Script

Based on your hardware:

#### For RTX 1060 or Similar GPUs (Recommended)
```bash
python examples/run_windows_gpu_sdpa.py
```
- Uses SDPA (Scaled Dot Product Attention)
- Best balance of speed and compatibility
- Works on all modern GPUs
- No flash-attention required

#### For Maximum Compatibility
```bash
python examples/run_windows_gpu_eager.py
```
- Uses eager attention mode
- Works on all GPUs
- Slightly slower than SDPA
- Most compatible option

#### For CPU-Only Testing
```bash
python examples/run_windows_cpu.py
```
- No GPU required
- Very slow (minutes per image)
- Good for testing installation
- Not recommended for production

## Script Details

### run_windows_gpu_sdpa.py
- **Attention Mode**: SDPA (Scaled Dot Product Attention)
- **Image Size**: 640x640 (Small mode)
- **Memory Usage**: ~4-5GB VRAM
- **Speed**: Fast
- **Compatibility**: All modern GPUs with PyTorch 2.0+

### run_windows_gpu_eager.py
- **Attention Mode**: Eager
- **Image Size**: 512x512 (Tiny mode)
- **Memory Usage**: ~3-4GB VRAM
- **Speed**: Medium
- **Compatibility**: All GPUs

### run_windows_cpu.py
- **Attention Mode**: Eager
- **Image Size**: 512x512 (Tiny mode)
- **Memory Usage**: System RAM
- **Speed**: Very slow
- **Compatibility**: Any system

### check_system.py
- Diagnostic tool to check your system configuration
- Provides recommendations based on your hardware
- No model loading required

## Customization

### Change Input Image

Edit the script and modify:
```python
image_file = 'test.png'  # Change to your image path
```

### Change Output Directory

Edit the script and modify:
```python
output_path = 'output'  # Change to your output directory
```

### Adjust Image Size

For more VRAM or better quality, modify:

**Tiny Mode (512x512)** - ~3-4GB VRAM:
```python
base_size = 512
image_size = 512
crop_mode = False
```

**Small Mode (640x640)** - ~4-5GB VRAM:
```python
base_size = 640
image_size = 640
crop_mode = False
```

**Base Mode (1024x1024)** - ~6-8GB VRAM:
```python
base_size = 1024
image_size = 1024
crop_mode = False
```

**Gundam Mode (Dynamic)** - ~8-12GB VRAM:
```python
base_size = 1024
image_size = 640
crop_mode = True
```

### Change Prompt

Different prompts for different use cases:

```python
# For documents (with layout preservation)
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# For general images
prompt = "<image>\n<|grounding|>OCR this image."

# For text-only extraction (no layout)
prompt = "<image>\nFree OCR."

# For figures in documents
prompt = "<image>\nParse the figure."

# For detailed image description
prompt = "<image>\nDescribe this image in detail."
```

## Troubleshooting

### Out of Memory Error

If you get CUDA out of memory:
1. Use a smaller mode (Tiny instead of Small)
2. Set `crop_mode = False`
3. Close other GPU applications
4. Try the CPU version

### Slow Performance

If inference is too slow:
1. Make sure you're using GPU mode (not CPU)
2. Try SDPA mode instead of eager
3. Verify CUDA is properly installed
4. Check GPU usage with Task Manager

### Import Errors

If you get import errors:
1. Make sure all dependencies are installed
2. Run: `pip install -r requirements.txt`
3. Check Python version (3.10 or 3.11 recommended)

### Model Download Issues

If model download fails:
1. Check your internet connection
2. Try using a VPN if Hugging Face is blocked
3. Download model manually and specify local path

## Performance Tips

1. **Use SDPA mode** for best balance of speed and compatibility
2. **Use float16** instead of bfloat16 on older GPUs
3. **Disable crop_mode** to save memory
4. **Close other applications** to free up VRAM
5. **Use smaller image sizes** if you have limited VRAM

## Requirements

- Python 3.10 or 3.11 (avoid 3.13)
- PyTorch 2.6.0 with CUDA 11.8 or 12.1
- Transformers 4.46.3
- 6GB+ VRAM for GPU mode (RTX 1060 or better)
- 16GB+ RAM for CPU mode

## Additional Resources

- [Windows Setup Guide](../WINDOWS_SETUP.md) - Detailed installation instructions
- [Main README](../README.md) - General project information
- [Hugging Face Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR) - Model card and documentation
