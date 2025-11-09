# DeepSeek-OCR Quick Start Guide

## Fix for ImportError: cannot import name 'GenerationMixin'

If you're seeing this error, it means transformers was installed in the wrong order. Follow these steps:

### 1. Clean Installation (Recommended)

```bash
# Uninstall conflicting packages
pip uninstall transformers vllm -y

# Install in correct order
pip install transformers>=4.51.1
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
pip install PyMuPDF img2pdf einops easydict addict Pillow numpy
```

### 2. Quick Test

```python
from vllm import LLM, SamplingParams
from PIL import Image

# Load model
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    max_model_len=8192,
    trust_remote_code=True,
    dtype="bfloat16",
)

# Test with an image
image = Image.open("your_image.jpg").convert("RGB")
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

outputs = llm.generate(model_input, sampling_params)
print(outputs[0].outputs[0].text)
```

### 3. For Kaggle/Colab Users

Use this exact order in your notebook:

```python
# Cell 1: Install (run once)
!pip install transformers>=4.51.1 -q
!pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 -q
!pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly -q
!pip install PyMuPDF img2pdf einops easydict addict Pillow numpy -q

# Cell 2: Import and use (after installation completes)
from vllm import LLM, SamplingParams
from PIL import Image

# Your code here...
```

## Need More Help?

- **Detailed Installation:** See [INSTALLATION.md](INSTALLATION.md)
- **Full Documentation:** See [README.md](README.md)
- **Issues:** [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
- **Discord:** [DeepSeek AI Discord](https://discord.gg/Tc7c45Zzu5)

## Key Points to Remember

✅ **DO:**
- Install transformers >= 4.51.1 FIRST
- Use vLLM nightly build for best compatibility
- Follow the installation order exactly

❌ **DON'T:**
- Install vLLM before transformers
- Use transformers < 4.51.1
- Skip the installation order

## Version Requirements

| Package | Minimum Version | Recommended |
|---------|----------------|-------------|
| Python | 3.10 | 3.11 |
| transformers | 4.51.1 | 4.51.1+ |
| torch | 2.6.0 | 2.6.0 |
| vLLM | 0.8.5 | nightly |
| CUDA | 11.8 | 11.8 or 12.1 |
