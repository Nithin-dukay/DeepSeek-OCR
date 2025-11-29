<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->


<div align="center">
  <img src="assets/logo.svg" width="60%" alt="DeepSeek AI" />
</div>


<hr>
<div align="center">
  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="Homepage" src="assets/badge.svg" />
  </a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>

<div align="center">

  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" />
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" />
  </a>

</div>

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c?logo=pytorch&logoColor=white" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.8%2B-76B900?logo=nvidia&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg" />
</div>

<br>

<p align="center">
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR"><b>📥 Model Download</b></a> |
  <a href="https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf"><b>📄 Paper Link</b></a> |
  <a href="https://arxiv.org/abs/2510.18234"><b>📄 Arxiv Paper Link</b></a> |
</p>

<h2>
<p align="center">
  <a href="">DeepSeek-OCR: Contexts Optical Compression</a>
</p>
</h2>

<p align="center">
<img src="assets/fig1.png" style="width: 1000px" align=center>
</p>
<p align="center">
<a href="">Explore the boundaries of visual-text compression.</a>       
</p>

---

<p align="center">
  <b>DeepSeek-OCR</b> is a state-of-the-art vision-language model for optical character recognition (OCR) and document understanding. 
  <br>
  It achieves <b>high-precision text extraction</b> with <b>layout preservation</b>, supporting multiple resolution modes and achieving up to <b>2500 tokens/s</b> on A100 GPUs.
</p>

<p align="center">
  🚀 <b>Production-Ready</b> • 📄 <b>Document Understanding</b> • 🎯 <b>Multi-Resolution</b> • ⚡ <b>High Performance</b>
</p>

---

## Release
- [2025/10/23]🚀🚀🚀 DeepSeek-OCR is now officially supported in upstream [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html#installing-vllm). Thanks to the [vLLM](https://github.com/vllm-project/vllm) team for their help.
- [2025/10/20]🚀🚀🚀 We release DeepSeek-OCR, a model to investigate the role of vision encoders from an LLM-centric viewpoint.

## Features

✨ **Key Capabilities**
- 🔍 **High-Precision OCR**: State-of-the-art text recognition accuracy
- 📄 **Document Understanding**: Layout-aware processing with structure preservation
- 🎯 **Multi-Resolution Support**: 5 resolution modes from 512×512 to 1280×1280
- 🚀 **Dynamic Resolution**: Gundam mode for adaptive processing of complex documents
- 📊 **Chart & Figure Parsing**: Specialized support for visual elements
- 🌐 **Multilingual**: Supports multiple languages including Chinese, English, and more
- ⚡ **High Performance**: Up to 2500 tokens/s on A100 GPU with vLLM
- 🔧 **Flexible APIs**: Both vLLM (production) and Transformers (simple) interfaces

## Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
- [Usage](#usage)
  - [vLLM Inference](#vllm-inference-recommended-for-production)
  - [Transformers Inference](#transformers-inference-simple-api)
- [Supported Resolution Modes](#supported-resolution-modes)
- [Prompt Examples](#prompt-examples)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)
- [FAQ](#faq)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)
- [Visualizations](#visualizations)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Quick Start

Get started with DeepSeek-OCR in minutes:

```bash
# 1. Install dependencies
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.3 Pillow einops easydict addict

# 2. Run OCR on an image (Python)
python3 << 'EOF'
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True).eval().cuda().to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)

result = model.infer(
    tokenizer,
    prompt='<image>\n<|grounding|>Convert the document to markdown.',
    image_file='your_image.jpg',
    output_path='./output',
    base_size=1024,
    image_size=640,
    crop_mode=True
)
print(result)
EOF
```

> 💡 **For production use** with high throughput (2500+ tokens/s), see [vLLM Inference](#vllm-inference-recommended-for-production).

## Installation

### Prerequisites
- **Environment**: CUDA 11.8 + PyTorch 2.6.0
- **Python**: 3.12.9 (or 3.8+)
- **GPU**: NVIDIA GPU with sufficient VRAM (A100-40G recommended for optimal performance)

### System Requirements

#### Minimum Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (e.g., V100, RTX 4090)
- **CUDA**: 11.8 or higher
- **RAM**: 32GB system memory
- **Storage**: 20GB free space for model weights
- **OS**: Linux (Ubuntu 20.04+, Amazon Linux 2023, etc.)

#### Recommended Requirements
- **GPU**: NVIDIA A100 (40GB) or H100
- **CUDA**: 11.8 or 12.1
- **RAM**: 64GB+ system memory
- **Storage**: 50GB+ SSD
- **OS**: Ubuntu 22.04 LTS

#### GPU Memory Usage by Mode

| Mode | Batch Size 1 | Batch Size 4 | Batch Size 8 |
|------|-------------|--------------|--------------|
| Tiny | ~8GB | ~12GB | ~16GB |
| Small | ~10GB | ~16GB | ~24GB |
| Base | ~14GB | ~24GB | ~40GB |
| Large | ~18GB | ~32GB | ~56GB |
| Gundam | ~16GB | ~28GB | ~48GB |

> **Note**: Memory usage varies based on image size and complexity. Use lower batch sizes or smaller modes if you encounter OOM errors.

### Quick Start

#### 1. Clone the Repository
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
```

#### 2. Set Up Python Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

**Option B: Using venv (For Amazon Linux 2023 / Sandbox Environments)**
```bash
python3 -m venv deepseek-ocr-env
source deepseek-ocr-env/bin/activate
```

> 📦 **Sandbox/Cloud Environment Note**: If you're running in a sandbox environment (Amazon Linux 2023, Vercel, etc.), use Option B with venv. Make sure Python 3.8+ is available. You may need to install pip first: `python3 -m ensurepip --upgrade`

#### 3. Install Dependencies

**Step 3.1: Upgrade pip**
```bash
pip install --upgrade pip
```

**Step 3.2: Install PyTorch**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

**Step 3.3: Install vLLM**

Download the vLLM 0.8.5 wheel file from [GitHub Releases](https://github.com/vllm-project/vllm/releases/tag/v0.8.5):
```bash
# Download the wheel file
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install vLLM
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

**Step 3.4: Install Project Requirements**
```bash
pip install -r requirements.txt
```

**Step 3.5: Install Flash Attention (Optional but Recommended)**
```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

> **Note**: If you encounter dependency conflicts between vLLM (requires `transformers>=4.51.1`) and the project requirements (`transformers==4.46.3`), you can safely ignore them. Both vLLM and Transformers inference methods can coexist in the same environment.

## Usage

### vLLM Inference (Recommended for Production)

vLLM provides high-throughput inference with batching support, achieving ~2500 tokens/s on an A100-40G GPU.

#### Configuration

Before running inference, configure the settings in `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

```python
# Model configuration
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # or your local model path

# Input/Output paths
INPUT_PATH = '/path/to/your/input'  # Set your input file/directory
OUTPUT_PATH = '/path/to/your/output'  # Set your output directory

# Resolution modes (choose one)
# Tiny: BASE_SIZE = 512, IMAGE_SIZE = 512, CROP_MODE = False
# Small: BASE_SIZE = 640, IMAGE_SIZE = 640, CROP_MODE = False
# Base: BASE_SIZE = 1024, IMAGE_SIZE = 1024, CROP_MODE = False
# Large: BASE_SIZE = 1280, IMAGE_SIZE = 1280, CROP_MODE = False
# Gundam (Dynamic): BASE_SIZE = 1024, IMAGE_SIZE = 640, CROP_MODE = True

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True

# Performance tuning
MAX_CONCURRENCY = 100  # Reduce if GPU memory is limited
NUM_WORKERS = 64       # Image preprocessing workers
MAX_CROPS = 6          # Max: 9; reduce to 6 for limited GPU memory

# Prompt template
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
```

#### Running Inference

Navigate to the vLLM directory:
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
```

**1. Image Inference (Streaming Output)**
```bash
python run_dpsk_ocr_image.py
```
Process single or multiple images with streaming output.

**2. PDF Inference (High Concurrency)**
```bash
python run_dpsk_ocr_pdf.py
```
Process PDF documents with concurrent page processing (~2500 tokens/s on A100-40G).

**3. Batch Evaluation (Benchmarks)**
```bash
python run_dpsk_ocr_eval_batch.py
```
Run batch evaluation on benchmark datasets like OmniDocBench.

**[2025/10/23] The version of upstream [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html#installing-vllm):**

```shell
uv venv
source .venv/bin/activate
# Until v0.11.1 release, you need to install vLLM from nightly build
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

# Create model instance
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

# Prepare batched input with your image file
image_1 = Image.open("path/to/your/image_1.png").convert("RGB")
image_2 = Image.open("path/to/your/image_2.png").convert("RGB")
prompt = "<image>\nFree OCR."

model_input = [
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_1}
    },
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_2}
    }
]

sampling_param = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit processor args
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )
# Generate output
model_outputs = llm.generate(model_input, sampling_param)

# Print output
for output in model_outputs:
    print(output.outputs[0].text)
```
### Transformers Inference (Simple API)

For simpler use cases or when you don't need high-throughput batching, use the Transformers API.

#### Python API

```python
from transformers import AutoModel, AutoTokenizer
import torch
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load model and tokenizer
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# Configure inference
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = 'path/to/your_image.jpg'
output_path = 'path/to/output/directory'

# Run inference
# Resolution modes:
# - Tiny: base_size=512, image_size=512, crop_mode=False
# - Small: base_size=640, image_size=640, crop_mode=False
# - Base: base_size=1024, image_size=1024, crop_mode=False
# - Large: base_size=1280, image_size=1280, crop_mode=False
# - Gundam: base_size=1024, image_size=640, crop_mode=True

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,      # Base resolution
    image_size=640,      # Crop resolution
    crop_mode=True,      # Enable dynamic cropping
    save_results=True,   # Save output to file
    test_compress=True   # Enable compression testing
)
```

#### Command Line

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

> **Note**: Edit `run_dpsk_ocr.py` to configure your input image path and output directory before running.
## Supported Resolution Modes

The model supports multiple resolution modes to balance speed and accuracy:

### Native Resolution Modes

| Mode | Resolution | Vision Tokens | Speed | Quality | Best For |
|------|-----------|---------------|-------|---------|----------|
| **Tiny** | 512×512 | 64 | ⚡⚡⚡⚡ | ⭐⭐ | Quick scans, simple text |
| **Small** | 640×640 | 100 | ⚡⚡⚡ | ⭐⭐⭐ | General documents |
| **Base** | 1024×1024 | 256 | ⚡⚡ | ⭐⭐⭐⭐ | Standard documents |
| **Large** | 1280×1280 | 400 | ⚡ | ⭐⭐⭐⭐⭐ | High-quality scans |

### Dynamic Resolution Mode

| Mode | Resolution | Vision Tokens | Description |
|------|-----------|---------------|-------------|
| **Gundam** | n×640×640 + 1×1024×1024 | Variable | Adaptive multi-scale processing for complex layouts |

> **💡 Tip**: Use **Gundam mode** for best results on documents with complex layouts, tables, or mixed content types.

## Prompt Examples

### Common Prompt Templates

```python
# Document to Markdown conversion (with layout preservation)
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# General OCR for images
prompt = "<image>\n<|grounding|>OCR this image."

# Free OCR without layout information
prompt = "<image>\nFree OCR."

# Parse figures and charts in documents
prompt = "<image>\nParse the figure."

# Detailed image description
prompt = "<image>\nDescribe this image in detail."

# Text localization (find specific text in image)
prompt = "<image>\nLocate <|ref|>先天下之忧而忧<|/ref|> in the image."
```

### Prompt Guidelines

- **`<image>`**: Required placeholder for image input
- **`<|grounding|>`**: Enables layout-aware processing (preserves document structure)
- **`<|ref|>...<|/ref|>`**: Marks reference text for localization tasks
- **Without special tokens**: Performs basic OCR without layout preservation

## Performance Optimization

### GPU Memory Management

If you encounter GPU memory issues, adjust these settings in `config.py`:

```python
MAX_CONCURRENCY = 50    # Reduce from 100
MAX_CROPS = 4           # Reduce from 6
NUM_WORKERS = 32        # Reduce from 64
```

### Throughput Benchmarks

| GPU | Mode | Throughput | Notes |
|-----|------|-----------|-------|
| A100-40G | PDF (vLLM) | ~2500 tokens/s | With concurrency |
| A100-40G | Image (vLLM) | ~1800 tokens/s | Streaming output |
| V100-32G | PDF (vLLM) | ~1500 tokens/s | Reduced concurrency |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Reduce batch size and crops
MAX_CONCURRENCY = 50
MAX_CROPS = 4
```

**2. Flash Attention Installation Fails**
```bash
# Solution: Install without flash attention (slower but works)
# Skip: pip install flash-attn==2.7.3 --no-build-isolation
# Model will fall back to standard attention
```

**3. Transformers Version Conflict**
```bash
# Solution: The warning can be safely ignored
# Both vLLM and Transformers work with transformers==4.46.3
```

**4. Model Download Issues**
```bash
# Solution: Use HuggingFace mirror or download manually
export HF_ENDPOINT=https://hf-mirror.com
# Or download from: https://huggingface.co/deepseek-ai/DeepSeek-OCR
```

## Project Structure

```
DeepSeek-OCR/
├── DeepSeek-OCR-master/
│   ├── DeepSeek-OCR-hf/          # Transformers-based inference
│   │   └── run_dpsk_ocr.py       # Simple inference script
│   └── DeepSeek-OCR-vllm/        # vLLM-based inference (production)
│       ├── config.py             # Configuration file (edit this!)
│       ├── deepseek_ocr.py       # Model implementation
│       ├── run_dpsk_ocr_image.py # Image inference
│       ├── run_dpsk_ocr_pdf.py   # PDF inference
│       ├── run_dpsk_ocr_eval_batch.py  # Batch evaluation
│       ├── deepencoder/          # Vision encoder modules
│       └── process/              # Image processing utilities
├── assets/                       # Documentation images
├── requirements.txt              # Python dependencies
├── DeepSeek_OCR_paper.pdf       # Research paper
└── README.md                     # This file
```

## Best Practices

### 1. **Choose the Right Mode**
- **Simple documents**: Use Small or Base mode
- **Complex layouts**: Use Gundam mode (dynamic resolution)
- **High-quality scans**: Use Large mode
- **Speed priority**: Use Tiny or Small mode

### 2. **Optimize for Your Hardware**
```python
# For GPUs with < 24GB VRAM
MAX_CONCURRENCY = 50
MAX_CROPS = 4
BASE_SIZE = 1024
IMAGE_SIZE = 640

# For GPUs with >= 40GB VRAM
MAX_CONCURRENCY = 100
MAX_CROPS = 6
BASE_SIZE = 1280
IMAGE_SIZE = 1024
```

### 3. **Batch Processing**
For processing multiple documents, use vLLM with batching:
```python
# Process multiple images efficiently
python run_dpsk_ocr_eval_batch.py
```

### 4. **Output Formats**
- **Markdown**: Best for documents with structure
- **Plain text**: Use "Free OCR" prompt for simple text extraction
- **JSON**: Parse structured data from tables and forms

## API Reference

### Model.infer() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | Tokenizer | Required | HuggingFace tokenizer instance |
| `prompt` | str | Required | Prompt template with `<image>` placeholder |
| `image_file` | str | Required | Path to input image file |
| `output_path` | str | Required | Directory for output files |
| `base_size` | int | 1024 | Base resolution (512/640/1024/1280) |
| `image_size` | int | 640 | Crop resolution for dynamic mode |
| `crop_mode` | bool | True | Enable dynamic resolution (Gundam mode) |
| `save_results` | bool | False | Save output to file |
| `test_compress` | bool | False | Enable compression testing |

## FAQ

### General Questions

**Q: What's the difference between vLLM and Transformers inference?**
- **vLLM**: High-throughput production inference with batching, ~2500 tokens/s on A100
- **Transformers**: Simple API for single-image processing, easier to use for prototyping

**Q: Which resolution mode should I use?**
- **Gundam mode** (dynamic): Best for most use cases, adapts to image complexity
- **Base/Large**: Fixed resolution, good for consistent document sizes
- **Tiny/Small**: Fast processing for simple documents

**Q: Can I run this on CPU?**
- Not recommended. The model requires GPU for reasonable performance. Minimum 16GB VRAM.

**Q: How do I process PDFs?**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
# Set INPUT_PATH to your PDF file in config.py
python run_dpsk_ocr_pdf.py
```

**Q: Does it support handwritten text?**
- The model is primarily trained on printed text. Handwriting recognition may have lower accuracy.

**Q: What languages are supported?**
- The model supports multiple languages including English, Chinese, and other major languages. Performance may vary by language.

### Technical Questions

**Q: How do I reduce GPU memory usage?**
```python
# In config.py
MAX_CONCURRENCY = 50  # Reduce concurrency
MAX_CROPS = 4         # Reduce crop count
BASE_SIZE = 640       # Use smaller resolution
```

**Q: Can I fine-tune the model?**
- Yes, the model can be fine-tuned. Refer to the [paper](DeepSeek_OCR_paper.pdf) for training details.

**Q: How do I use a local model path?**
```python
# In config.py or your script
MODEL_PATH = '/path/to/local/model'
```

**Q: What's the maximum image size?**
- No hard limit, but larger images consume more memory. Gundam mode automatically handles large images by cropping.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## Support

- 📧 **Issues**: [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
- 💬 **Discord**: [DeepSeek AI Community](https://discord.gg/Tc7c45Zzu5)
- 🐦 **Twitter**: [@deepseek_ai](https://twitter.com/deepseek_ai)
- 🤗 **HuggingFace**: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## Visualizations
<table>
<tr>
<td><img src="assets/show1.jpg" style="width: 500px"></td>
<td><img src="assets/show2.jpg" style="width: 500px"></td>
</tr>
<tr>
<td><img src="assets/show3.jpg" style="width: 500px"></td>
<td><img src="assets/show4.jpg" style="width: 500px"></td>
</tr>
</table>


## Acknowledgement

We would like to thank [Vary](https://github.com/Ucas-HaoranWei/Vary/), [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/), [MinerU](https://github.com/opendatalab/MinerU), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [OneChart](https://github.com/LingyvKong/OneChart), [Slow Perception](https://github.com/Ucas-HaoranWei/Slow-Perception) for their valuable models and ideas.

We also appreciate the benchmarks: [Fox](https://github.com/ucaslcl/Fox), [OminiDocBench](https://github.com/opendatalab/OmniDocBench).

## Citation

```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
