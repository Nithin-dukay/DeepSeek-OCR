# Windows 11 Setup Guide for DeepSeek-OCR

This guide addresses common issues when running DeepSeek-OCR on Windows 11, particularly with flash-attention compatibility and GPU limitations.

## Issue Overview

**Problem**: Flash-attention is difficult to install on Windows and may not be compatible with older GPUs like RTX 1060.

**Root Causes**:
1. Flash-attention requires specific CUDA compute capabilities (>= 7.5 for flash-attn 2.x)
2. RTX 1060 has compute capability 6.1, which is **not supported** by flash-attention 2.x
3. Flash-attention compilation on Windows is complex and often fails
4. Python 3.13 may have compatibility issues with some dependencies

## Solutions

### Solution 1: Use Eager Attention (Recommended for Windows)

Instead of flash-attention, use PyTorch's native "eager" attention implementation:

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Use 'eager' instead of 'flash_attention_2'
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='eager',  # Changed from 'flash_attention_2'
    trust_remote_code=True, 
    use_safetensors=True
)

# For RTX 1060 (6GB VRAM), use float16 instead of bfloat16
model = model.eval().cuda().to(torch.float16)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'test.png'
output_path = 'output'

# Use smaller settings for limited VRAM
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=512,      # Reduced from 1024
    image_size=512,     # Reduced from 640
    crop_mode=False,    # Disable cropping to save memory
    save_results=True, 
    test_compress=True
)
```

### Solution 2: Use SDPA (Scaled Dot Product Attention)

PyTorch 2.0+ includes optimized SDPA that works on most GPUs:

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Use 'sdpa' for better performance than eager
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='sdpa',  # Scaled Dot Product Attention
    trust_remote_code=True, 
    use_safetensors=True
)

model = model.eval().cuda().to(torch.float16)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'test.png'
output_path = 'output'

res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=640,
    image_size=640,
    crop_mode=False,
    save_results=True, 
    test_compress=True
)
```

### Solution 3: CPU-Only Mode (Slowest but Most Compatible)

For testing without GPU:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model without GPU
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='eager',
    trust_remote_code=True, 
    use_safetensors=True,
    torch_dtype=torch.float32  # CPU works better with float32
)

model = model.eval()  # No .cuda() call

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'test.png'
output_path = 'output'

# Use smallest settings for CPU
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=512,      # Tiny mode
    image_size=512,
    crop_mode=False,
    save_results=True, 
    test_compress=False  # Disable compression for speed
)
```

## Installation Instructions for Windows 11

### Prerequisites
- Python 3.10 or 3.11 (avoid 3.13 for better compatibility)
- CUDA 11.8 or 12.1 (if using GPU)
- Visual Studio 2019 or 2022 with C++ build tools (optional, for some packages)

### Step 1: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n deepseek-ocr python=3.11 -y
conda activate deepseek-ocr

# Or using venv
python -m venv deepseek-ocr-env
deepseek-ocr-env\Scripts\activate
```

### Step 2: Install PyTorch

For CUDA 11.8:
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Dependencies (Without Flash-Attention)

```bash
pip install transformers==4.46.3
pip install tokenizers==0.20.3
pip install PyMuPDF
pip install img2pdf
pip install einops
pip install easydict
pip install addict
pip install Pillow
pip install numpy
```

### Step 4: Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
```

## Memory Optimization for RTX 1060 (6GB VRAM)

The RTX 1060 has limited VRAM. Use these settings:

### Tiny Mode (Fastest, Lowest Quality)
```python
base_size = 512
image_size = 512
crop_mode = False
# Expected VRAM usage: ~3-4GB
```

### Small Mode (Balanced)
```python
base_size = 640
image_size = 640
crop_mode = False
# Expected VRAM usage: ~4-5GB
```

### If Out of Memory
1. Reduce `base_size` and `image_size`
2. Set `crop_mode = False`
3. Use `torch.float16` instead of `torch.bfloat16`
4. Close other GPU applications
5. Add gradient checkpointing (if available in model config)

## Troubleshooting

### Error: "CUDA out of memory"
- Reduce `base_size` and `image_size`
- Use CPU mode
- Close other applications using GPU

### Error: "flash_attn not found"
- Use `attn_implementation='eager'` or `'sdpa'` instead
- Do not install flash-attn on Windows with older GPUs

### Error: "bfloat16 not supported"
- Use `torch.float16` instead of `torch.bfloat16`
- RTX 1060 doesn't have native bfloat16 support

### Error: "trust_remote_code"
- This is required for DeepSeek-OCR custom model code
- Ensure you have `trust_remote_code=True` in `from_pretrained()`

### Slow Performance
- Use GPU mode with SDPA: `attn_implementation='sdpa'`
- Ensure CUDA is properly installed
- Use smaller image sizes
- Consider upgrading GPU for production use

## Performance Comparison

| Mode | Speed | Quality | VRAM | Windows Compatible |
|------|-------|---------|------|-------------------|
| flash_attention_2 | Fastest | High | Medium | ❌ (RTX 1060) |
| sdpa | Fast | High | Medium | ✅ |
| eager | Medium | High | Medium | ✅ |
| CPU | Slowest | High | N/A | ✅ |

## Recommended Configuration for RTX 1060

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='sdpa',  # Best balance for Windows
    trust_remote_code=True, 
    use_safetensors=True,
    torch_dtype=torch.float16,  # RTX 1060 compatible
    low_cpu_mem_usage=True       # Reduce CPU memory during loading
)

model = model.eval().cuda()

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'test.png'
output_path = 'output'

# Small mode for RTX 1060
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=640,
    image_size=640,
    crop_mode=False,
    save_results=True, 
    test_compress=True
)
```

## Additional Resources

- [PyTorch SDPA Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Transformers Attention Implementation](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2)
- [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus)

## Support

For issues specific to Windows, please report them with:
1. Windows version
2. GPU model and VRAM
3. Python version
4. PyTorch version
5. CUDA version
6. Full error traceback
