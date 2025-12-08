# Quick Fix for Windows 11 / RTX 1060 Users

## TL;DR - Just Make These Changes

### Change 1: Attention Mode
```python
# ❌ OLD (doesn't work)
_attn_implementation='flash_attention_2'

# ✅ NEW (works on Windows)
attn_implementation='sdpa'  # or 'eager'
```

### Change 2: Data Type
```python
# ❌ OLD (not supported on RTX 1060)
.to(torch.bfloat16)

# ✅ NEW (supported)
# For GPU:
model = AutoModel.from_pretrained(..., torch_dtype=torch.float16)
model = model.eval().cuda()

# For CPU:
model = AutoModel.from_pretrained(..., torch_dtype=torch.float32)
model = model.eval()
```

### Change 3: Image Size (for 6GB VRAM)
```python
# ❌ OLD (may cause out of memory)
base_size = 1024
image_size = 640
crop_mode = True

# ✅ NEW (fits in 6GB)
base_size = 640
image_size = 640
crop_mode = False
```

## Complete Working Example

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='sdpa',      # Changed
    trust_remote_code=True, 
    use_safetensors=True,
    torch_dtype=torch.float16,       # Changed
    low_cpu_mem_usage=True
)

model = model.eval().cuda()          # Changed

prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = 'test.png'
output_path = 'output'

res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=640,                   # Changed
    image_size=640,                  # Changed
    crop_mode=False,                 # Changed
    save_results=True, 
    test_compress=True
)
```

## Installation (Without Flash-Attention)

```bash
# 1. Create environment
conda create -n deepseek-ocr python=3.11 -y
conda activate deepseek-ocr

# 2. Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies (skip flash-attn)
pip install transformers==4.46.3 tokenizers==0.20.3
pip install PyMuPDF img2pdf einops easydict addict Pillow numpy
```

## Quick Test

```bash
# Check your system
python examples/check_system.py

# Run example
python examples/run_windows_gpu_sdpa.py
```

## Need More Help?

- **Full Guide**: See [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
- **Examples**: See [examples/README.md](examples/README.md)
- **Solution Details**: See [ISSUE_287_SOLUTION.md](ISSUE_287_SOLUTION.md)
