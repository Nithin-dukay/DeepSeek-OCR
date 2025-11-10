# README Update for DeepSeek-OCR

## Add this section to the main README.md

---

## Transformers Inference (Warning-Free) 🆕

**New in this release:** Clean inference without warnings!

If you're experiencing warnings when using DeepSeek-OCR with Transformers (Issue #65), we now provide a wrapper that eliminates all common warnings while maintaining full functionality.

### Quick Start

```python
from DeepSeek_OCR_hf import DeepSeekOCRForCausalLM
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = DeepSeekOCRForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
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

### What's Fixed

✅ Model type mismatch warnings  
✅ Uninitialized weights warnings  
✅ Generation configuration warnings  
✅ Attention mask warnings  
✅ Deprecation warnings  

### Documentation

- **Quick Start**: See `DeepSeek-OCR-master/DeepSeek-OCR-hf/QUICK_START.md`
- **Full Documentation**: See `DeepSeek-OCR-master/DeepSeek-OCR-hf/WARNINGS_FIX_README.md`
- **Testing**: Run `python3 DeepSeek-OCR-master/DeepSeek-OCR-hf/test_warnings_fix.py`

### Hardware Compatibility

- ✅ NVIDIA T4 (no Flash Attention 2 required)
- ✅ NVIDIA A100, A10, RTX 3090+ (with Flash Attention 2 support)
- ✅ CPU inference (slower but works)

### Alternative: Original Method

You can still use the original method if you prefer:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
```

Note: This method will show warnings but still works correctly.

---

## Add this to the Installation section

### For Warning-Free Transformers Inference

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
pip install transformers==4.46.3 tokenizers==0.20.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r ../../requirements.txt
```

---

## Add this to the Troubleshooting section

### Issue: Warnings During Model Loading

**Problem**: You see warnings like:
- "You are using a model of type deepseek_vl_v2 to instantiate a model of type DeepseekOCR"
- "Some weights were not initialized from the model checkpoint"
- "do_sample is set to False. However, temperature is set to 0.0"

**Solution**: Use the warning-free wrapper:

```python
from DeepSeek_OCR_hf import DeepSeekOCRForCausalLM

model = DeepSeekOCRForCausalLM.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
```

See `DeepSeek-OCR-master/DeepSeek-OCR-hf/WARNINGS_FIX_README.md` for details.

---
