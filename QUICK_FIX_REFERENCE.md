# Quick Fix Reference - Issue #7

## The Error
```
ImportError: cannot import name 'LlamaFlashAttention2' from 'transformers.models.llama.modeling_llama'
```

## Quick Solutions

### 🎯 Solution 1: Install Correct Version (RECOMMENDED)
```bash
pip install transformers==4.46.3 tokenizers==0.20.3
```

### 🔧 Solution 2: Use Compatibility Patch
```python
from fix_transformers_compatibility import apply_transformers_compatibility_patch
apply_transformers_compatibility_patch()

# Then load model normally
from transformers import AutoModel
model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
```

### ⚡ Solution 3: Use Eager Attention
```python
model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    _attn_implementation='eager',  # Instead of 'flash_attention_2'
    trust_remote_code=True
)
```

## Google Colab Quick Start
```python
# Cell 1: Install
!pip install -q transformers==4.46.3 tokenizers==0.20.3
!pip install -q torch einops easydict addict Pillow numpy

# Cell 2: Load Model
from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'deepseek-ai/DeepSeek-OCR',
    _attn_implementation='eager',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Cell 3: Run OCR
from google.colab import files
uploaded = files.upload()
image_file = list(uploaded.keys())[0]

result = model.infer(
    tokenizer,
    prompt="<image>\\n<|grounding|>Convert the document to markdown.",
    image_file=image_file,
    base_size=1024,
    image_size=640,
    crop_mode=True
)
print(result)
```

## Test Your Fix
```bash
python test_fix.py
```

## More Information
- **Detailed Guide:** [ISSUE_7_FIX.md](ISSUE_7_FIX.md)
- **Colab Guide:** [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)
- **Summary:** [ISSUE_7_SUMMARY.md](ISSUE_7_SUMMARY.md)

## Why This Happens
- DeepSeek-OCR uses `trust_remote_code=True`
- Remote code imports `LlamaFlashAttention2`
- This class was removed in transformers >= 4.47.0
- Solution: Use compatible version or apply patch

## Which Solution Should I Use?

| Situation | Recommended Solution |
|-----------|---------------------|
| Fresh install | Solution 1 (transformers==4.46.3) |
| Google Colab | Solution 1 + Colab guide |
| Version conflict | Solution 2 (compatibility patch) |
| Quick test | Solution 3 (eager attention) |
| Production | Solution 1 (transformers==4.46.3) |

## Still Having Issues?

1. Check your transformers version: `pip show transformers`
2. Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Run test script: `python test_fix.py`
4. See full documentation: [ISSUE_7_FIX.md](ISSUE_7_FIX.md)
5. Open GitHub issue with error details

---
**Last Updated:** 2025-11-10  
**Status:** ✅ Fixed  
**Affects:** transformers >= 4.47.0
