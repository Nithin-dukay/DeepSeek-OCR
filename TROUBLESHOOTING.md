# DeepSeek-OCR Troubleshooting Guide

## Common Issues and Solutions

### Issue #302: `cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'`

#### Problem Description
When loading the DeepSeek-OCR model, you encounter the following error:
```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

This occurs because the model's HuggingFace code (`modeling_deepseekv2.py`) imports `LlamaFlashAttention2` and `LlamaAttention` from transformers, but these classes are not available in transformers 4.46.3 (and were removed in 4.47+).

#### Root Cause
The model code includes these imports:
```python
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2
)
```

These classes are used in the `ATTENTION_CLASSES` dictionary for fallback MHA (Multi-Head Attention) modes:
```python
ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,
    "mla_eager": DeepseekV2Attention,
    "mla_flash_attention_2": DeepseekV2FlashAttention2,
    "mha_eager": LlamaAttention,  # ← Fallback mode
    "mha_flash_attention_2": LlamaFlashAttention2  # ← Fallback mode
}
```

**Important**: The model primarily uses MLA (Multi-head Latent Attention), so the Llama attention classes are only used as fallbacks for MHA mode, which most users don't need.

---

## Solutions

### Solution 1: Use Eager Attention (Recommended - Quick Fix)

**This is the easiest and recommended solution for most users.**

Modify your model loading code to use `attn_implementation='eager'`:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='eager',  # ← Add this parameter
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)
```

**Why this works**: Using `attn_implementation='eager'` forces the model to use the DeepSeek's own attention implementation (MLA), bypassing the need for Llama attention classes entirely.

**Performance impact**: Minimal to none for typical use cases, as the model is designed to use MLA by default.

---

### Solution 2: Downgrade Transformers (Not Recommended)

You could try using an older version of transformers that includes `LlamaFlashAttention2`, but this is not recommended due to potential security and compatibility issues.

```bash
pip install transformers==4.45.0  # Not recommended
```

**Why not recommended**: 
- Older versions may have security vulnerabilities
- May conflict with other dependencies
- The eager attention solution (Solution 1) is cleaner and more maintainable

---

### Solution 3: Use vLLM (For Production/High-Performance Inference)

If you need high-performance inference, use the vLLM implementation which is officially supported:

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

# Prepare input
image = Image.open("path/to/your/image.png").convert("RGB")
prompt = "<image>\\nFree OCR."

model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

sampling_param = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
    ),
    skip_special_tokens=False,
)

# Generate output
model_outputs = llm.generate(model_input, sampling_param)
print(model_outputs[0].outputs[0].text)
```

**Advantages**:
- No import issues
- Better performance for batch inference
- Officially supported by DeepSeek team

---

### Solution 4: Create a Local Patched Model (Advanced)

For advanced users who need full control, you can download the model files and patch the imports:

1. Download the model from HuggingFace:
```bash
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
cd DeepSeek-OCR
```

2. Edit `modeling_deepseekv2.py` to add conditional imports:
```python
# Replace the original import
try:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaFlashAttention2
    )
    LLAMA_ATTENTION_AVAILABLE = True
except ImportError:
    LLAMA_ATTENTION_AVAILABLE = False
    LlamaAttention = None
    LlamaFlashAttention2 = None

# Update ATTENTION_CLASSES dictionary
ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
    "flash_attention_2": DeepseekV2FlashAttention2,
    "mla_eager": DeepseekV2Attention,
    "mla_flash_attention_2": DeepseekV2FlashAttention2,
}

# Only add Llama attention if available
if LLAMA_ATTENTION_AVAILABLE:
    ATTENTION_CLASSES["mha_eager"] = LlamaAttention
    ATTENTION_CLASSES["mha_flash_attention_2"] = LlamaFlashAttention2
```

3. Load from local path:
```python
model = AutoModel.from_pretrained(
    "./DeepSeek-OCR",  # Local path
    trust_remote_code=True,
    use_safetensors=True
)
```

---

## Verification

After applying any solution, verify the fix works:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        attn_implementation='eager',
        trust_remote_code=True, 
        use_safetensors=True
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
```

---

## Related Issues

- GitHub Issue #302: https://github.com/deepseek-ai/DeepSeek-OCR/issues/302
- GitHub Issue #161: Transformers Version Conflict When Deploying with vLLM
- GitHub Issue #182: Incorrect Requirements - transformers 4.51.2 Incompatible
- HuggingFace Discussion #38: Make compatible with newer transformers

---

## Additional Help

If you continue to experience issues:

1. Check your transformers version: `pip show transformers`
2. Ensure you're using the latest code from this repository
3. Try the vLLM implementation for production use
4. Open a new issue on GitHub with:
   - Your transformers version
   - Complete error traceback
   - Minimal code to reproduce the issue

---

## Summary

**For most users**: Use Solution 1 (eager attention) - it's simple, effective, and has no performance impact.

**For production**: Use Solution 3 (vLLM) - it's officially supported and optimized for performance.

**For advanced users**: Use Solution 4 (local patching) - gives you full control over the model code.
