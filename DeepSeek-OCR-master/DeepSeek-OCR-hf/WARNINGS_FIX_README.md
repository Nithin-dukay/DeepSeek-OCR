# DeepSeek-OCR Warnings Fix - Issue #65

## Overview

This document explains the fixes implemented to resolve the warnings that appear when using DeepSeek-OCR with Transformers inference on NVIDIA T4 instances (or similar hardware without Flash Attention 2 support).

## Warnings Addressed

### 1. Model Type Mismatch Warning
**Original Warning:**
```
You are using a model of type deepseek_vl_v2 to instantiate a model of type DeepseekOCR. 
This is not supported for all configurations of models and can yield errors.
```

**Root Cause:** The model's `config.json` on Hugging Face Hub has `model_type = "deepseek_vl_v2"`, but the actual model class is `DeepseekOCRForCausalLM`. This mismatch causes Transformers to issue a warning.

**Fix:** Created a custom configuration class (`DeepSeekOCRConfig`) that properly registers the model type as `"deepseek_ocr"` and a wrapper model class (`DeepSeekOCRForCausalLM`) that handles the model loading with proper type registration.

### 2. Uninitialized Weights Warning
**Original Warning:**
```
Some weights of DeepseekOCRForCausalLM were not initialized from the model checkpoint at cache_models 
and are newly initialized: ['model.vision_model.embeddings.position_ids']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

**Root Cause:** The `position_ids` buffer in the vision model is not saved in the checkpoint (it's a computed buffer, not a learned parameter). This is expected behavior but Transformers issues a warning.

**Fix:** The warning is suppressed in the model wrapper as it's a false positive - `position_ids` is automatically computed during forward pass and doesn't need to be loaded from the checkpoint.

### 3. Generation Configuration Warnings
**Original Warnings:**
```
`do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in 
sample-based generation modes. You should set `do_sample=True` or unset `temperature`.

The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior.

Setting `pad_token_id` to `eos_token_id`:None for open-end generation.

The attention mask is not set and cannot be inferred from input because pad token is same as eos token.
```

**Root Cause:** The generation configuration has conflicting or missing parameters:
- `temperature` is set when `do_sample=False` (temperature is only used for sampling)
- `pad_token_id` is not properly configured
- Attention masks are not being set correctly

**Fix:** The model wrapper's `generate()` and `infer()` methods now:
- Remove `temperature` parameter when `do_sample=False`
- Automatically set `pad_token_id` to `eos_token_id` if not provided
- Suppress expected warnings about attention masks (which are handled internally by the model)

### 4. Deprecation Warnings
**Original Warnings:**
```
The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` model input instead.

`get_max_cache()` is deprecated for all Cache classes. Use `get_max_cache_shape()` instead.

The attention layers in this model are transitioning from computing the RoPE embeddings internally through 
`position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings`.
```

**Root Cause:** These are deprecation warnings from Transformers library about upcoming API changes in the underlying model implementation.

**Fix:** These warnings are suppressed as they are informational and don't affect functionality. The actual fixes need to be made in the upstream model implementation on Hugging Face Hub, which is outside the scope of this repository.

## Implementation

### Files Created

1. **`configuration_deepseek_ocr.py`**
   - Custom configuration class that properly registers the model type
   - Handles conversion from `deepseek_vl_v2` to `deepseek_ocr` model type
   - Sets proper default values for generation parameters

2. **`modeling_deepseek_ocr.py`**
   - Wrapper model class that loads the actual DeepSeek-OCR model with `trust_remote_code=True`
   - Suppresses expected/harmless warnings during model loading and inference
   - Provides proper `generate()` and `infer()` methods with fixed parameters
   - Delegates all other method calls to the underlying model transparently

3. **`run_dpsk_ocr.py`** (Updated)
   - Uses the new wrapper classes instead of direct `AutoModel.from_pretrained()`
   - Demonstrates proper usage with clean output

4. **`WARNINGS_FIX_README.md`** (This file)
   - Comprehensive documentation of the issues and fixes

### Files Modified

- **`run_dpsk_ocr.py`**: Updated to use the custom model wrapper

## Usage

### Basic Usage (Recommended)

```python
from transformers import AutoTokenizer
import torch
from modeling_deepseek_ocr import DeepSeekOCRForCausalLM

# Load model and tokenizer
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Use the custom wrapper - this eliminates warnings
model = DeepSeekOCRForCausalLM.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',  # or 'eager' if FA2 not available
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

model = model.eval()

# Run inference
prompt = "<image>\\n<|grounding|>Convert the document to markdown."
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file='your_image.jpg',
    output_path='output/',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)
```

### Alternative: Direct AutoModel Usage (Not Recommended)

If you prefer to use `AutoModel.from_pretrained()` directly, you can suppress warnings manually:

```python
import warnings
from transformers import AutoModel, AutoTokenizer

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*model of type.*")
warnings.filterwarnings("ignore", message=".*not initialized from.*")
warnings.filterwarnings("ignore", message=".*TRAIN this model.*")
warnings.filterwarnings("ignore", message=".*do_sample.*temperature.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*seen_tokens.*")
warnings.filterwarnings("ignore", message=".*get_max_cache.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")

# Load model
model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
```

However, this approach only suppresses warnings without fixing the underlying issues.

## Resolution Modes

DeepSeek-OCR supports multiple resolution modes:

| Mode | base_size | image_size | crop_mode | Vision Tokens |
|------|-----------|------------|-----------|---------------|
| Tiny | 512 | 512 | False | 64 |
| Small | 640 | 640 | False | 100 |
| Base | 1024 | 1024 | False | 256 |
| Large | 1280 | 1280 | False | 400 |
| Gundam (Dynamic) | 1024 | 640 | True | Variable (256 + n×100) |

## Hardware Compatibility

### Flash Attention 2 Support

Flash Attention 2 requires:
- NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer)
- Examples: A100, A10, RTX 3090, RTX 4090, etc.

**NVIDIA T4 (Compute Capability 7.5):**
- Does NOT support Flash Attention 2
- Use `_attn_implementation='eager'` or omit the parameter
- The model will fall back to standard attention implementation

### Installation Notes

For systems without Flash Attention 2 support:

```bash
# Install PyTorch with CUDA support
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install Transformers and dependencies
pip install transformers==4.46.3 tokenizers==0.20.3
pip install -r requirements.txt

# DO NOT install flash-attn on T4 or older GPUs
```

For systems with Flash Attention 2 support:

```bash
# Install PyTorch with CUDA support
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install Transformers and dependencies
pip install transformers==4.46.3 tokenizers==0.20.3
pip install -r requirements.txt

# Install Flash Attention 2
pip install flash-attn==2.7.3 --no-build-isolation
```

## Testing

To verify the fixes work correctly:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

You should see:
- ✅ No model type mismatch warnings
- ✅ No uninitialized weights warnings
- ✅ No generation configuration warnings
- ✅ Deprecation warnings are suppressed (informational only)
- ✅ Model loads and runs successfully

## Technical Details

### Why These Warnings Occur

1. **Model Type Mismatch**: DeepSeek-OCR is based on DeepSeek-VL-V2 architecture but has custom modifications. The Hugging Face Hub model uses `model_type = "deepseek_vl_v2"` in its config, but the actual implementation is a custom class.

2. **Position IDs**: Vision transformers compute position embeddings dynamically. The `position_ids` buffer is not a learned parameter and doesn't need to be in the checkpoint.

3. **Generation Config**: The default generation configuration from the base model has parameters that conflict with typical OCR inference patterns.

4. **Deprecations**: Transformers library is evolving, and some internal APIs are being updated. These warnings are informational and don't affect current functionality.

### Why Suppressing Is Safe

The warnings being suppressed are either:
- **False positives**: Like the uninitialized `position_ids` warning
- **Configuration mismatches**: That are corrected by our wrapper
- **Informational**: Like deprecation warnings that don't affect current functionality

The actual model functionality is not affected, and the suppression is done in a targeted way to only hide expected warnings while allowing genuine errors to surface.

## Troubleshooting

### Issue: Import Error for Custom Classes

**Error:**
```python
ModuleNotFoundError: No module named 'modeling_deepseek_ocr'
```

**Solution:**
Make sure you're running the script from the correct directory:
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

Or add the directory to Python path:
```python
import sys
sys.path.insert(0, '/path/to/DeepSeek-OCR-master/DeepSeek-OCR-hf')
```

### Issue: CUDA Out of Memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**
Use a smaller resolution mode or enable gradient checkpointing:
```python
# Use smaller resolution
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file='image.jpg',
    base_size=640,  # Reduced from 1024
    image_size=640,
    crop_mode=False,  # Disable cropping
    ...
)
```

### Issue: Flash Attention Not Available

**Warning:**
```
Flash Attention is not available, falling back to eager attention
```

**Solution:**
This is expected on T4 GPUs. Either:
1. Explicitly use eager attention:
   ```python
   model = DeepSeekOCRForCausalLM.from_pretrained(
       model_name,
       _attn_implementation='eager',
       ...
   )
   ```
2. Or omit the `_attn_implementation` parameter entirely

## Contributing

If you encounter additional warnings or issues not covered by this fix, please:

1. Open an issue on GitHub with:
   - Full warning/error message
   - Your hardware configuration
   - Python and package versions
   - Minimal reproduction code

2. Check if the issue is in the upstream model by testing with the original code

## References

- [DeepSeek-OCR GitHub Repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR Hugging Face Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

## License

This fix is provided under the same license as the DeepSeek-OCR project.
