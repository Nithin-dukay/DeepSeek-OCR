# DeepSeek-OCR Quick Start Guide

Get DeepSeek-OCR running in 5 minutes!

## Prerequisites

- NVIDIA GPU with 16GB+ VRAM
- CUDA 11.8+
- Python 3.8+
- Linux OS

## Installation (3 Steps)

### 1. Install PyTorch
```bash
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Dependencies
```bash
pip install transformers==4.46.3 Pillow einops easydict addict PyMuPDF img2pdf
```

### 3. Verify Installation
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Basic Usage

### Simple OCR (Transformers API)

Create a file `ocr_test.py`:

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model
print("Loading model...")
model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True
).eval().cuda().to(torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True
)

# Run OCR
print("Running OCR...")
result = model.infer(
    tokenizer,
    prompt='<image>\n<|grounding|>Convert the document to markdown.',
    image_file='your_image.jpg',  # Change this to your image path
    output_path='./output',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True
)

print("Result:", result)
print("Output saved to: ./output/")
```

Run it:
```bash
python3 ocr_test.py
```

## Common Prompts

```python
# Document to Markdown (with layout)
prompt = '<image>\n<|grounding|>Convert the document to markdown.'

# Simple OCR (no layout)
prompt = '<image>\nFree OCR.'

# Parse figures/charts
prompt = '<image>\nParse the figure.'

# Detailed description
prompt = '<image>\nDescribe this image in detail.'
```

## Resolution Modes

Choose based on your needs:

| Mode | Config | Use Case |
|------|--------|----------|
| **Fast** | `base_size=640, image_size=640, crop_mode=False` | Simple documents |
| **Balanced** | `base_size=1024, image_size=640, crop_mode=True` | Most documents (recommended) |
| **High Quality** | `base_size=1280, image_size=1280, crop_mode=False` | Complex documents |

## Troubleshooting

### Out of Memory Error
```python
# Use smaller resolution
base_size=640
image_size=640
crop_mode=False
```

### Model Download Slow
```bash
# Use HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### CUDA Not Available
```bash
# Check CUDA installation
nvidia-smi
nvcc --version
```

## Next Steps

- 📖 Read the [full README](README.md) for advanced features
- 🚀 Try [vLLM inference](README.md#vllm-inference-recommended-for-production) for production (2500+ tokens/s)
- 📄 Process PDFs with `run_dpsk_ocr_pdf.py`
- 🎯 Batch processing with `run_dpsk_ocr_eval_batch.py`

## Production Setup (vLLM)

For high-throughput production use:

```bash
# 1. Install vLLM
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# 2. Configure
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
# Edit config.py: set INPUT_PATH and OUTPUT_PATH

# 3. Run
python run_dpsk_ocr_image.py  # For images
python run_dpsk_ocr_pdf.py    # For PDFs
```

## Support

- 💬 [Discord Community](https://discord.gg/Tc7c45Zzu5)
- 🐛 [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
- 📚 [Full Documentation](README.md)
- 🤗 [HuggingFace Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

---

**Ready to go?** Run your first OCR in the next 2 minutes! 🚀
