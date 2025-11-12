# Fix for GitHub Issue #237: ImportError with transformers.generation

## Problem Summary

Users were experiencing the following error when trying to use DeepSeek-OCR with vLLM and transformers 4.51.1:

```
ImportError: cannot import name 'GenerationMixin' from 'transformers.generation' 
(/usr/local/lib/python3.11/dist-packages/transformers/generation/__init__.py)
```

## Root Cause

The issue was caused by an **unused import** in `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py`:

```python
from transformers.generation.logits_process import _calc_banned_ngram_tokens
```

This import had two problems:
1. **It was never used** - The `NoRepeatNGramLogitsProcessor` class implements its own n-gram logic without calling this function
2. **It's a private function** - Functions starting with `_` are not part of the public API and can break between versions

## Solution Applied

### 1. Fixed the Import Issue

**File:** `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py`

**Changed from:**
```python
import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from typing import List, Set
```

**Changed to:**
```python
import torch
from transformers import LogitsProcessor
from typing import List, Set
```

### 2. Updated requirements.txt

**File:** `requirements.txt`

**Changed from:**
```
transformers==4.46.3
tokenizers==0.20.3
```

**Changed to:**
```
transformers>=4.46.3,<=4.51.1
tokenizers>=0.20.3
```

This allows compatibility with transformers versions from 4.46.3 up to 4.51.1.

## Installation Instructions

### Option 1: Using This Repository (Fixed Version)

```bash
# 1. Clone the repository
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR

# 2. Create conda environment
conda create -n deepseek-ocr python=3.11 -y
conda activate deepseek-ocr

# 3. Install PyTorch 2.6.0 with CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install transformers 4.51.1 (now compatible!)
pip install transformers==4.51.1

# 5. Install vLLM nightly
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly

# 6. Install other requirements
pip install -r requirements.txt

# 7. Optional: Install flash-attention
pip install flash-attn==2.7.3 --no-build-isolation
```

### Option 2: Using Upstream vLLM (Recommended for Production)

As of October 23, 2025, DeepSeek-OCR is officially supported in upstream vLLM. This is the **recommended approach** for production use:

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install vLLM nightly
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# 3. Install Pillow
pip install pillow
```

## Usage Examples

### Using the Fixed Repository Code

```python
from vllm import LLM, SamplingParams
from PIL import Image
import sys
sys.path.append('DeepSeek-OCR-master/DeepSeek-OCR-vllm')
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Load model
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    max_model_len=8192,
    trust_remote_code=True,
    dtype="bfloat16",
)

# Load image
image = Image.open("test_image.jpg").convert("RGB")
prompt = "<image>\n<|grounding|>Convert the document to markdown."

model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
)

# Generate
outputs = llm.generate(model_input, sampling_params)
print(outputs[0].outputs[0].text)
```

### Using Upstream vLLM (Recommended)

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

# Load image
image = Image.open("test_image.jpg").convert("RGB")
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
        whitelist_token_ids={128821, 128822},  # <td>, </td> tokens
    ),
    skip_special_tokens=False,
)

# Generate
outputs = llm.generate(model_input, sampling_params)
print(outputs[0].outputs[0].text)
```

## Kaggle-Specific Instructions

For Kaggle notebooks, use this installation order:

```python
# 1. Install transformers FIRST
!pip install transformers==4.51.1 -q

# 2. Install PyTorch
!pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 -q

# 3. Install vLLM nightly
!pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly -q

# 4. Install Pillow
!pip install pillow -q

# 5. Optional: flash-attention (may fail on Kaggle, that's okay)
try:
    !pip install flash-attn==2.7.3 --no-build-isolation -q
except:
    print("flash-attn skipped (optional)")
```

## Verification

To verify the fix works, run:

```python
# Test the import
from transformers import LogitsProcessor
print("✓ LogitsProcessor imported successfully")

# Test the fixed module
import sys
sys.path.append('DeepSeek-OCR-master/DeepSeek-OCR-vllm')
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
print("✓ NoRepeatNGramLogitsProcessor imported successfully")

# Test vLLM
from vllm import LLM, SamplingParams
print("✓ vLLM imported successfully")

print("\n✅ All imports successful! Issue #237 is fixed.")
```

## Technical Details

### Why This Fix Works

1. **Removed Unused Import**: The `_calc_banned_ngram_tokens` function was imported but never called in the code
2. **No Functionality Lost**: The `NoRepeatNGramLogitsProcessor` class has its own complete implementation
3. **Version Compatibility**: By removing the private function import, the code now works with any transformers version that has `LogitsProcessor`

### What Changed in transformers 4.51.1

- Transformers 4.51.1 is a patch release focused on Llama 4 fixes and torch 2.6.0 compatibility
- No breaking changes to the public generation API
- `GenerationMixin` is still properly exported
- The error was a red herring caused by the private function import

### Upstream vLLM Changes

The official vLLM repository (as of October 23, 2025) has:
- Native DeepSeek-OCR support built-in
- Rewritten n-gram processor that doesn't depend on transformers internals
- Better integration with vLLM's v1 sampling architecture
- More efficient batched processing

## Troubleshooting

### Issue: Still getting ImportError

**Solution:** Make sure you've pulled the latest code with the fix:
```bash
git pull origin main
```

### Issue: CUDA out of memory

**Solution:** Reduce `max_model_len` or use smaller batch sizes:
```python
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    max_model_len=4096,  # Reduced from 8192
    # ... other params
)
```

### Issue: flash-attention installation fails

**Solution:** This is optional. The model will work without it, just slightly slower:
```bash
# Skip flash-attention if it fails
pip install -r requirements.txt
# Continue without flash-attention
```

## Related Issues

- GitHub Issue #237: ImportError with transformers.generation
- vLLM PR: DeepSeek-OCR upstream support (October 23, 2025)
- Transformers 4.51.1 release notes

## Credits

- Fix implemented for GitHub Issue #237
- Thanks to the vLLM team for upstream DeepSeek-OCR support
- Thanks to the DeepSeek-AI team for the original model

## License

This fix is provided under the same license as the DeepSeek-OCR repository.
