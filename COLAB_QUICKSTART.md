# Google Colab Quick Start Guide for DeepSeek-OCR

This guide helps you get DeepSeek-OCR running in Google Colab without encountering the `LlamaFlashAttention2` import error.

## Issue #7 Fix: Quick Setup

### Option 1: Use Compatible Versions (Recommended)

Simply install the correct versions at the start of your Colab notebook:

```python
# Cell 1: Install dependencies with compatible versions
!pip install -q transformers==4.46.3 tokenizers==0.20.3
!pip install -q torch torchvision
!pip install -q einops easydict addict Pillow numpy

# Optional: Install flash-attention for better performance
# Note: This may take several minutes to compile
!pip install -q flash-attn==2.7.3 --no-build-isolation

print("✓ Installation complete!")
```

```python
# Cell 2: Verify installation
import torch
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
```

```python
# Cell 3: Load and use the model
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',  # or 'eager' if flash-attn not installed
    trust_remote_code=True,
    use_safetensors=True
)

# Move to GPU and set to bfloat16
model = model.eval().cuda().to(torch.bfloat16)

print("✓ Model loaded successfully!")
```

```python
# Cell 4: Run inference
from PIL import Image
import requests
from io import BytesIO

# Example: Load an image from URL
image_url = "https://example.com/your-image.jpg"  # Replace with your image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Or upload from local file
# from google.colab import files
# uploaded = files.upload()
# image_file = list(uploaded.keys())[0]

# Run OCR
prompt = "<image>\\n<|grounding|>Convert the document to markdown."

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image,  # Can be PIL Image or file path
    output_path='./output',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)

print("OCR Result:")
print(res)
```

### Option 2: Use Compatibility Patch (If you must use newer transformers)

If you need to use `transformers>=4.57.0` for other dependencies:

```python
# Cell 1: Install dependencies
!pip install -q transformers>=4.57.0
!pip install -q torch torchvision
!pip install -q einops easydict addict Pillow numpy

# Download the compatibility patch
!wget -q https://raw.githubusercontent.com/deepseek-ai/DeepSeek-OCR/main/fix_transformers_compatibility.py

print("✓ Installation complete!")
```

```python
# Cell 2: Apply compatibility patch
from fix_transformers_compatibility import apply_transformers_compatibility_patch

# Apply patch BEFORE importing the model
apply_transformers_compatibility_patch()

print("✓ Compatibility patch applied!")
```

```python
# Cell 3: Load model (same as Option 1, Cell 3)
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',  # Use eager attention with newer transformers
    trust_remote_code=True,
    use_safetensors=True
)

model = model.eval().cuda().to(torch.bfloat16)

print("✓ Model loaded successfully!")
```

## Complete Colab Notebook Template

Here's a complete notebook you can copy and paste:

```python
# ============================================================
# DeepSeek-OCR in Google Colab - Complete Setup
# ============================================================

# Step 1: Install dependencies
print("Installing dependencies...")
!pip install -q transformers==4.46.3 tokenizers==0.20.3
!pip install -q torch torchvision
!pip install -q einops easydict addict Pillow numpy
print("✓ Dependencies installed")

# Step 2: Verify environment
import torch
import transformers
print(f"\\nPyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Step 3: Load model
from transformers import AutoModel, AutoTokenizer

model_name = 'deepseek-ai/DeepSeek-OCR'
print(f"\\nLoading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',  # Safe option for Colab
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

model = model.eval()
print("✓ Model loaded successfully!")

# Step 4: Upload and process image
from google.colab import files
from PIL import Image
import io

print("\\nPlease upload an image:")
uploaded = files.upload()

# Get the first uploaded file
image_file = list(uploaded.keys())[0]
image = Image.open(io.BytesIO(uploaded[image_file]))

print(f"Image uploaded: {image_file}")
print(f"Image size: {image.size}")

# Step 5: Run OCR
print("\\nRunning OCR...")
prompt = "<image>\\n<|grounding|>Convert the document to markdown."

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image,
    output_path='./output',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)

print("\\n" + "="*60)
print("OCR RESULT:")
print("="*60)
print(result)
print("="*60)
```

## Troubleshooting

### Error: "CUDA out of memory"

If you encounter memory errors:

1. Use a smaller model size:
```python
res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image,
    base_size=640,      # Reduced from 1024
    image_size=512,     # Reduced from 640
    crop_mode=False,    # Disable cropping
    save_results=True
)
```

2. Or use Colab Pro with more GPU memory

### Error: "ImportError: LlamaFlashAttention2"

This means you're using an incompatible transformers version. Solutions:

1. **Restart runtime and use transformers==4.46.3** (recommended)
2. Use the compatibility patch (Option 2 above)
3. Use `_attn_implementation='eager'` instead of `'flash_attention_2'`

### Error: "flash_attn not installed"

Flash Attention is optional. If you get this error:

1. Either install it: `!pip install flash-attn==2.7.3 --no-build-isolation`
2. Or use eager attention: `_attn_implementation='eager'`

### Model loading is slow

This is normal for the first time. The model is ~10GB and needs to be downloaded. Subsequent runs will use cached weights.

## Model Size Configurations

Choose based on your needs and available GPU memory:

| Configuration | base_size | image_size | crop_mode | GPU Memory | Use Case |
|--------------|-----------|------------|-----------|------------|----------|
| Tiny         | 512       | 512        | False     | ~4GB       | Quick tests |
| Small        | 640       | 640        | False     | ~6GB       | Simple documents |
| Base         | 1024      | 1024       | False     | ~10GB      | Standard documents |
| Large        | 1280      | 1280       | False     | ~16GB      | High quality |
| Gundam       | 1024      | 640        | True      | ~12GB      | Complex layouts |

## Supported Prompts

```python
# For documents with layout
prompt = "<image>\\n<|grounding|>Convert the document to markdown."

# For general images
prompt = "<image>\\n<|grounding|>OCR this image."

# For text-only extraction (no layout)
prompt = "<image>\\nFree OCR."

# For figures in documents
prompt = "<image>\\nParse the figure."

# For detailed image description
prompt = "<image>\\nDescribe this image in detail."

# For locating specific text
prompt = "<image>\\nLocate <|ref|>your text here<|/ref|> in the image."
```

## Additional Resources

- [Main README](README.md)
- [Issue #7 Fix Documentation](ISSUE_7_FIX.md)
- [DeepSeek-OCR Paper](DeepSeek_OCR_paper.pdf)
- [HuggingFace Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## Support

If you encounter issues:

1. Check that you're using `transformers==4.46.3`
2. Verify CUDA is available: `torch.cuda.is_available()`
3. Try with `_attn_implementation='eager'`
4. Reduce model size configuration
5. Open an issue on GitHub with your error message and environment details
