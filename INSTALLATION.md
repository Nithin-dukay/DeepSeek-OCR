# DeepSeek-OCR Installation Guide

This guide provides detailed installation instructions to fix the `ImportError: cannot import name 'GenerationMixin'` issue and ensure proper setup of DeepSeek-OCR.

## Table of Contents
- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Environment-Specific Guides](#environment-specific-guides)
- [Troubleshooting](#troubleshooting)
- [Version Compatibility](#version-compatibility)

---

## Quick Start

### Option 1: Using the Setup Script (Recommended)

```bash
# Clone the repository
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR

# Run the setup script
bash setup.sh
```

### Option 2: Manual Installation

```bash
# 1. Create and activate conda environment
conda create -n deepseek-ocr python=3.11 -y
conda activate deepseek-ocr

# 2. Install transformers FIRST (critical!)
pip install transformers>=4.51.1

# 3. Install PyTorch with CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install vLLM nightly (includes DeepSeek-OCR support)
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly

# 5. Install other dependencies
pip install PyMuPDF img2pdf einops easydict addict Pillow numpy

# 6. Optional: Install flash-attn for better performance
pip install flash-attn==2.7.3 --no-build-isolation
```

---

## Detailed Installation

### Step 1: Environment Setup

#### Using Conda (Recommended)
```bash
conda create -n deepseek-ocr python=3.11 -y
conda activate deepseek-ocr
```

#### Using venv
```bash
python3.11 -m venv deepseek-ocr-env
source deepseek-ocr-env/bin/activate  # On Windows: deepseek-ocr-env\Scripts\activate
```

### Step 2: Install Dependencies in Correct Order

**⚠️ CRITICAL: Installation order matters!**

#### 2.1 Install Transformers First
```bash
pip install transformers>=4.51.1 tokenizers>=0.20.3
```

**Why first?** vLLM requires transformers >= 4.51.1, and installing it first prevents version conflicts.

#### 2.2 Install PyTorch
Choose the appropriate command based on your CUDA version:

**CUDA 11.8:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

**CPU Only:**
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 2.3 Install vLLM

**Option A: vLLM Nightly (Recommended - includes DeepSeek-OCR support)**
```bash
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
```

**Option B: vLLM 0.8.5 (Stable)**
```bash
# Download the wheel file from https://github.com/vllm-project/vllm/releases/tag/v0.8.5
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

#### 2.4 Install Other Dependencies
```bash
pip install PyMuPDF img2pdf einops easydict addict Pillow numpy
```

#### 2.5 Install Flash Attention (Optional but Recommended)
```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

**Note:** Flash Attention installation may fail on some systems. This is optional and the model will work without it, but performance may be reduced.

---

## Environment-Specific Guides

### Kaggle Notebooks

```python
# Cell 1: Install dependencies in correct order
!pip install transformers>=4.51.1 -q
!pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 -q
!pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly -q
!pip install PyMuPDF img2pdf einops easydict addict Pillow numpy -q

# Optional: Flash Attention
try:
    !pip install flash-attn==2.7.3 --no-build-isolation -q
except:
    print("flash-attn installation skipped (optional)")

# Cell 2: Import and use
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

# Load model
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor],
    max_model_len=8192,
    trust_remote_code=True,
    dtype="bfloat16",
)

# Load and process image
image = Image.open("/kaggle/working/test_image.jpg").convert("RGB")
prompt = "<image>\n<|grounding|>Convert the document to markdown."

model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
    ),
    skip_special_tokens=False,
)

# Generate
outputs = llm.generate(model_input, sampling_params)
print(outputs[0].outputs[0].text)
```

### Google Colab

```python
# Cell 1: Setup
!pip install transformers>=4.51.1 -q
!pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 -q
!pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly -q
!pip install PyMuPDF img2pdf einops easydict addict Pillow numpy -q

# Restart runtime after installation
import os
os.kill(os.getpid(), 9)

# Cell 2: After restart, use the model
# (same code as Kaggle example above)
```

### Local Development

```bash
# Clone repository
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR

# Create environment
conda create -n deepseek-ocr python=3.11 -y
conda activate deepseek-ocr

# Install dependencies
pip install transformers>=4.51.1
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
pip install PyMuPDF img2pdf einops easydict addict Pillow numpy
pip install flash-attn==2.7.3 --no-build-isolation

# Run inference
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

---

## Troubleshooting

### Issue 1: ImportError: cannot import name 'GenerationMixin'

**Error:**
```
ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'
```

**Solution:**
This occurs when transformers version is too old. Ensure you install transformers >= 4.51.1 **before** installing vLLM.

```bash
pip uninstall transformers vllm -y
pip install transformers>=4.51.1
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
```

### Issue 2: vLLM requires transformers >= 4.51.1

**Error:**
```
ERROR: vllm 0.8.5+cu118 requires transformers>=4.51.1
```

**Solution:**
Install transformers first, then vLLM:
```bash
pip install transformers>=4.51.1
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
```

### Issue 3: CUDA version mismatch

**Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:**
Check your CUDA version and install the matching PyTorch:
```bash
# Check CUDA version
nvidia-smi

# For CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
```

### Issue 4: Flash Attention installation fails

**Error:**
```
ERROR: Failed building wheel for flash-attn
```

**Solution:**
Flash Attention is optional. Skip it if installation fails:
```bash
# The model will work without flash-attn, but may be slower
# If you want to try installing it:
pip install flash-attn==2.7.3 --no-build-isolation

# If it still fails, continue without it
```

### Issue 5: Out of Memory (OOM) errors

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**
Reduce memory usage:
```python
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    max_model_len=4096,  # Reduce from 8192
    gpu_memory_utilization=0.75,  # Reduce from 0.9
    max_num_seqs=50,  # Reduce concurrency
)
```

### Issue 6: Model not found

**Error:**
```
OSError: deepseek-ai/DeepSeek-OCR does not appear to be a valid model identifier
```

**Solution:**
Ensure you have internet connection and Hugging Face access:
```bash
# Login to Hugging Face (if required)
huggingface-cli login

# Or specify local model path
llm = LLM(model="/path/to/local/model")
```

---

## Version Compatibility

### Tested Configurations

| Python | Transformers | PyTorch | vLLM | CUDA | Status |
|--------|-------------|---------|------|------|--------|
| 3.11   | 4.51.1      | 2.6.0   | nightly | 11.8 | ✅ Working |
| 3.11   | 4.51.1      | 2.6.0   | 0.8.5 | 11.8 | ✅ Working |
| 3.10   | 4.51.1      | 2.6.0   | nightly | 12.1 | ✅ Working |
| 3.11   | 4.46.3      | 2.6.0   | nightly | 11.8 | ❌ ImportError |
| 3.9    | 4.51.1      | 2.6.0   | nightly | 11.8 | ⚠️ Not tested |

### Minimum Requirements

- **Python:** 3.10 or higher (3.11 recommended)
- **Transformers:** >= 4.51.1
- **PyTorch:** >= 2.6.0
- **vLLM:** >= 0.8.5 or nightly
- **CUDA:** 11.8 or 12.1 (for GPU)
- **GPU Memory:** 16GB+ recommended (A100, V100, T4)

### Recommended Configuration

```
Python: 3.11
transformers: 4.51.1
torch: 2.6.0
vLLM: nightly build
CUDA: 11.8
flash-attn: 2.7.3
GPU: A100 40GB or V100 32GB
```

---

## Additional Resources

- **GitHub Issues:** [DeepSeek-OCR Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
- **vLLM Documentation:** [vLLM DeepSeek-OCR Recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- **Hugging Face Model:** [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Paper:** [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check existing [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
2. Search [vLLM Discussions](https://github.com/vllm-project/vllm/discussions)
3. Join [DeepSeek Discord](https://discord.gg/Tc7c45Zzu5)
4. Create a new issue with:
   - Python version (`python --version`)
   - Package versions (`pip list | grep -E "transformers|torch|vllm"`)
   - Full error traceback
   - System information (`nvidia-smi` output)
