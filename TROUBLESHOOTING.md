# Troubleshooting Guide

## Issue #302: LlamaFlashAttention2 Import Error

### Problem Description

When trying to load the DeepSeek-OCR model using HuggingFace transformers, you may encounter the following error:

```python
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

### Root Cause

The `LlamaFlashAttention2` class was removed from the transformers library before version 4.46 as part of an attention mechanism refactoring. However, the DeepSeek-OCR model's custom code (`modeling_deepseekv2.py`) that gets downloaded when using `trust_remote_code=True` still attempts to import this removed class.

**Timeline:**
- **transformers < 4.45**: `LlamaFlashAttention2` available
- **transformers 4.45.2**: Last version with `LlamaFlashAttention2`
- **transformers 4.46+**: `LlamaFlashAttention2` removed

### Solutions

#### Solution 1: Use Compatible Transformers Version (Recommended)

Install transformers version 4.45.2 which still includes `LlamaFlashAttention2`:

```bash
pip install transformers==4.45.2
```

This is the version specified in the updated `requirements.txt` file.

#### Solution 2: Use Eager Attention

Modify your model loading code to use eager attention instead of flash_attention_2:

```python
from transformers import AutoModel, AutoTokenizer

model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Use 'eager' instead of 'flash_attention_2'
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='eager',  # Changed from 'flash_attention_2'
    trust_remote_code=True, 
    use_safetensors=True
)
```

**Note:** Eager attention may be slightly slower than flash_attention_2 but is compatible with all transformers versions.

#### Solution 3: Use Pre-Patched Model

Use a community-patched version of the model that's compatible with newer transformers:

```python
model_name = "prithivMLmods/DeepSeek-OCR-Latest-BF16.I64"
```

### Verification

After applying any of the above solutions, verify the fix by running:

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
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")
```

### Performance Considerations

- **flash_attention_2**: Faster inference, requires specific transformers version (≤4.45.2)
- **eager**: Slightly slower but more compatible across transformers versions
- **Impact**: For most use cases, the performance difference is negligible

### Related Issues

- GitHub Issue #302: cannot import name 'LlamaFlashAttention2'
- GitHub Issue #182: Incorrect Requirements - transformers 4.51.2 Incompatible
- HuggingFace Discussion #38: Make compatible with newer transformers

### Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [DeepSeek-OCR Model Card](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Flash Attention Documentation](https://github.com/Dao-AILab/flash-attention)

### Still Having Issues?

If you continue to experience problems after trying these solutions:

1. Check your transformers version: `pip show transformers`
2. Ensure you have the correct dependencies: `pip install -r requirements.txt`
3. Clear your HuggingFace cache: `rm -rf ~/.cache/huggingface/`
4. Try reinstalling in a fresh virtual environment
5. Open a new issue on GitHub with your error logs and environment details
