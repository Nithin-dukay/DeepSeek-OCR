# DeepSeek-OCR Installation Guide

This guide provides detailed installation instructions for DeepSeek-OCR with proper dependency management to avoid common errors like `ImportError: cannot import name 'GenerationMixin'`.

## Table of Contents
- [Version Compatibility Matrix](#version-compatibility-matrix)
- [Installation Methods](#installation-methods)
  - [Method 1: Local vLLM 0.8.5 (Recommended for Stability)](#method-1-local-vllm-085-recommended-for-stability)
  - [Method 2: Upstream vLLM Nightly (Latest Features)](#method-2-upstream-vllm-nightly-latest-features)
  - [Method 3: Transformers-only (HuggingFace)](#method-3-transformers-only-huggingface)
- [Platform-Specific Instructions](#platform-specific-instructions)
  - [Kaggle](#kaggle)
  - [Google Colab](#google-colab)
  - [Local Machine](#local-machine)
- [Troubleshooting](#troubleshooting)
- [Environment Validation](#environment-validation)

---

## Version Compatibility Matrix

### For vLLM 0.8.5 (Local - Recommended)
| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.8-3.12 | 3.12.9 recommended |
| CUDA | 11.8 | Required for GPU support |
| PyTorch | 2.6.0 | Must match CUDA version |
| transformers | 4.46.3 | **Critical: Do not use 4.51.1** |
| vLLM | 0.8.5+cu118 | From official whl file |
| flash-attn | 2.7.3 | Optional but recommended |

### For vLLM Nightly (Upstream)
| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.8-3.12 | Latest stable recommended |
| CUDA | 11.8+ or 12.1+ | Check vLLM compatibility |
| PyTorch | 2.4.0+ | Auto-installed with vLLM |
| transformers | >=4.51.1 | Auto-installed with vLLM |
| vLLM | nightly | From wheels.vllm.ai |
| flash-attn | Latest | Optional |

---

## Installation Methods

### Method 1: Local vLLM 0.8.5 (Recommended for Stability)

This method uses the tested vLLM 0.8.5 wheel file and is the most stable option.

#### Step 1: Create Environment
```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

#### Step 2: Install PyTorch (CUDA 11.8)
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

#### Step 3: Download and Install vLLM 0.8.5
```bash
# Download the wheel file
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install vLLM
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

#### Step 4: Install Project Requirements
```bash
cd DeepSeek-OCR
pip install -r requirements.txt
```

#### Step 5: Install Flash Attention (Optional but Recommended)
```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

**Note:** If flash-attn installation fails, you can skip it. The model will fall back to standard attention.

---

### Method 2: Upstream vLLM Nightly (Latest Features)

⚠️ **WARNING**: This method requires different dependency versions and may have compatibility issues.

#### Step 1: Create Environment
```bash
# Using uv (recommended by vLLM)
uv venv
source .venv/bin/activate

# OR using conda
conda create -n deepseek-ocr-nightly python=3.11 -y
conda activate deepseek-ocr-nightly
```

#### Step 2: Install vLLM Nightly
```bash
# Using uv
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# OR using pip
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
```

**Important:** vLLM nightly will automatically install compatible versions of PyTorch and transformers. Do NOT install transformers 4.46.3 in this environment.

#### Step 3: Install Additional Dependencies
```bash
pip install Pillow PyMuPDF img2pdf einops easydict addict numpy
```

#### Step 4: Install Flash Attention (Optional)
```bash
pip install flash-attn --no-build-isolation
```

---

### Method 3: Transformers-only (HuggingFace)

For inference without vLLM (slower but simpler).

#### Step 1: Create Environment
```bash
conda create -n deepseek-ocr-hf python=3.11 -y
conda activate deepseek-ocr-hf
```

#### Step 2: Install Dependencies
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.3
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

#### Step 3: Run Inference
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

---

## Platform-Specific Instructions

### Kaggle

Kaggle notebooks have specific constraints. Use this installation order:

```python
# Cell 1: Install PyTorch first
!pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 -q

# Cell 2: Install vLLM nightly
!pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly -q

# Cell 3: Install other dependencies
!pip install Pillow PyMuPDF img2pdf einops easydict addict numpy -q

# Cell 4: Optional - Flash Attention
try:
    !pip install flash-attn --no-build-isolation -q
except:
    print("Flash attention installation skipped (optional)")

# Cell 5: Verify installation
!python -c "import vllm; import torch; print(f'vLLM: {vllm.__version__}'); print(f'PyTorch: {torch.__version__}')"
```

**Critical Notes for Kaggle:**
- Do NOT install transformers separately - vLLM nightly includes it
- Run cells in order and restart kernel if you get import errors
- Kaggle T4 GPUs support bfloat16 dtype

### Google Colab

Similar to Kaggle, but with some differences:

```python
# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Install PyTorch
!pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 -q

# Cell 3: Install vLLM nightly
!pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly -q

# Cell 4: Install dependencies
!pip install Pillow PyMuPDF img2pdf einops easydict addict numpy -q

# Cell 5: Clone repository
!git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
%cd DeepSeek-OCR
```

### Local Machine

For local installation, use Method 1 (vLLM 0.8.5) for best stability. See automated setup scripts:
- `setup_vllm_local.sh` - For vLLM 0.8.5
- `setup_vllm_upstream.sh` - For vLLM nightly

---

## Troubleshooting

### Error: `ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'`

**Cause:** Version mismatch between transformers and vLLM.

**Solutions:**

1. **If using vLLM 0.8.5:**
   ```bash
   pip uninstall transformers -y
   pip install transformers==4.46.3
   ```

2. **If using vLLM nightly:**
   ```bash
   pip uninstall transformers -y
   pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly --force-reinstall
   ```

3. **Nuclear option (clean reinstall):**
   ```bash
   conda deactivate
   conda env remove -n deepseek-ocr
   # Start fresh with Method 1 or Method 2
   ```

### Error: `vllm 0.8.5+cu118 requires transformers>=4.51.1`

**Cause:** You're mixing vLLM 0.8.5 with incompatible transformers version.

**Solution:** This warning can be ignored when using vLLM 0.8.5 with transformers 4.46.3. The project is designed to work with this combination. If you want both vLLM and transformers in the same environment, use Method 2 (vLLM nightly) instead.

### Error: `CUDA out of memory`

**Solutions:**
1. Reduce `MAX_CROPS` in `config.py` (try 4 or 6 instead of 9)
2. Reduce `max_model_len` (try 4096 instead of 8192)
3. Increase `gpu_memory_utilization` to 0.9 (default 0.75)
4. Use smaller image sizes (640x640 instead of 1024x1024)

### Error: `flash_attn` installation fails

**Solution:** Flash attention is optional. Skip it and the model will use standard attention:
```bash
# Continue without flash-attn
# The model will automatically fall back to torch SDPA
```

### Error: `ModuleNotFoundError: No module named 'deepencoder'`

**Cause:** Running from wrong directory.

**Solution:**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

### Error: Model download is slow or fails

**Solutions:**
1. Use HuggingFace mirror:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. Pre-download model:
   ```python
   from transformers import AutoModel, AutoTokenizer
   model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
   tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
   ```

---

## Environment Validation

Before running inference, validate your environment:

```bash
python check_environment.py
```

This script will check:
- Python version
- CUDA availability
- Package versions
- Compatibility issues

Manual validation:
```python
import torch
import transformers
import vllm

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"vLLM: {vllm.__version__}")
```

Expected output for Method 1 (vLLM 0.8.5):
```
Python: 3.12.9
PyTorch: 2.6.0+cu118
CUDA Available: True
CUDA Version: 11.8
Transformers: 4.46.3
vLLM: 0.8.5+cu118
```

Expected output for Method 2 (vLLM nightly):
```
Python: 3.11.x
PyTorch: 2.4.0+ (varies)
CUDA Available: True
CUDA Version: 11.8 or 12.1
Transformers: 4.51.1+
vLLM: 0.x.x (nightly)
```

---

## Quick Start After Installation

### Using vLLM (Method 1 or 2)

```python
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

# Prepare input
image = Image.open("your_image.jpg").convert("RGB")
prompt = "<image>\n<|grounding|>Convert the document to markdown."

model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

# Generate
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

outputs = llm.generate(model_input, sampling_params)
print(outputs[0].outputs[0].text)
```

### Using Transformers (Method 3)

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = 'your_image.jpg'

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    base_size=1024,
    image_size=640,
    crop_mode=True
)
```

---

## Additional Resources

- [Official DeepSeek-OCR Repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM DeepSeek-OCR Recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [Issue Tracker](https://github.com/deepseek-ai/DeepSeek-OCR/issues)

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting](#troubleshooting) section
2. Run `python check_environment.py` to validate your setup
3. Search existing [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
4. Create a new issue with:
   - Your installation method (1, 2, or 3)
   - Output of `check_environment.py`
   - Full error traceback
   - Platform (Kaggle/Colab/Local)
