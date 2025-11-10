# Quick Start Guide - DeepSeek-OCR Without Warnings

## Installation

```bash
# Install dependencies
pip install transformers==4.46.3 tokenizers==0.20.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Optional: Flash Attention 2 (only for Ampere+ GPUs)
# Skip this on T4 or older GPUs
pip install flash-attn==2.7.3 --no-build-isolation
```

## Basic Usage

```python
from modeling_deepseek_ocr import DeepSeekOCRForCausalLM
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

print(result)
```

## Resolution Modes

| Mode | base_size | image_size | crop_mode | Tokens | Use Case |
|------|-----------|------------|-----------|--------|----------|
| Tiny | 512 | 512 | False | 64 | Quick tests |
| Small | 640 | 640 | False | 100 | Simple documents |
| Base | 1024 | 1024 | False | 256 | Standard documents |
| Large | 1280 | 1280 | False | 400 | High-quality scans |
| Gundam | 1024 | 640 | True | Variable | Large documents |

## Common Prompts

```python
# Document to Markdown
prompt = "<image>\\n<|grounding|>Convert the document to markdown."

# General OCR
prompt = "<image>\\n<|grounding|>OCR this image."

# Free OCR (no layout)
prompt = "<image>\\nFree OCR."

# Figure parsing
prompt = "<image>\\nParse the figure."

# Detailed description
prompt = "<image>\\nDescribe this image in detail."
```

## Hardware Compatibility

### NVIDIA T4 (Compute Capability 7.5)
```python
# DO NOT use flash_attention_2
model = DeepSeekOCRForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
```

### NVIDIA A100/A10/RTX 3090+ (Compute Capability 8.0+)
```python
# CAN use flash_attention_2
model = DeepSeekOCRForCausalLM.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
```

## Troubleshooting

### Out of Memory
```python
# Use smaller resolution
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file='image.jpg',
    base_size=640,  # Reduced
    image_size=640,
    crop_mode=False,  # Disable cropping
    ...
)
```

### Import Error
```bash
# Make sure you're in the correct directory
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

### Flash Attention Not Available
```python
# Just omit the _attn_implementation parameter
model = DeepSeekOCRForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
```

## Testing

```bash
# Run the test suite
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python3 test_warnings_fix.py
```

Expected output:
```
============================================================
Total: 5/5 tests passed
============================================================
🎉 All tests passed! The warnings fix is working correctly.
```

## What's Fixed

✅ No more "model of type deepseek_vl_v2" warning  
✅ No more "not initialized from checkpoint" warning  
✅ No more "do_sample/temperature" warning  
✅ No more "pad_token_id" warning  
✅ No more "attention mask" warning  
✅ Deprecation warnings suppressed  

## Need Help?

- 📖 Read the full documentation: `WARNINGS_FIX_README.md`
- 🐛 Report issues on GitHub
- 💬 Join the Discord community

## License

Same as DeepSeek-OCR project.
