# DeepSeek-OCR vLLM Serve Guide

## Fix for GitHub Issue #244

This guide provides solutions for the `pydantic_core.ValidationError` when using `vllm serve` with DeepSeek-OCR.

## Problem

When running:
```bash
vllm serve deepseek-ai/DeepSeek-OCR --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --no-enable-prefix-caching --mm-processor-cache-gb 0
```

You may encounter:
```
pydantic_core.ValidationError: 1 validation error for ModelConfig
Value error, Model architectures 'DeepseekOCRForCausallM' are not supported for now
```

## Root Cause

The custom `DeepseekOCRForCausalLM` model class needs to be registered with vLLM before it can be used with the `vllm serve` command.

## Solutions

### Solution 1: Use Python API (Recommended)

Instead of using `vllm serve` CLI, use the Python API with proper model registration:

```python
import sys
import os

# Add the DeepSeek-OCR-vllm directory to Python path
sys.path.insert(0, '/path/to/DeepSeek-OCR-master/DeepSeek-OCR-vllm')

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from PIL import Image

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Create model instance
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    trust_remote_code=True,
    logits_processors=[NoRepeatNGramLogitsProcessor]
)

# Prepare input
image = Image.open("path/to/your/image.png").convert("RGB")
prompt = "<image>\\nFree OCR."

model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

# Configure sampling
sampling_param = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # <td>, </td>
    ),
    skip_special_tokens=False,
)

# Generate output
model_outputs = llm.generate(model_input, sampling_param)

# Print results
for output in model_outputs:
    print(output.outputs[0].text)
```

### Solution 2: Use Provided Scripts

The repository includes pre-configured scripts that handle model registration:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# For images
python run_dpsk_ocr_image.py

# For PDFs
python run_dpsk_ocr_pdf.py

# For batch evaluation
python run_dpsk_ocr_eval_batch.py
```

Make sure to configure `config.py` first:
```python
# config.py
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
INPUT_PATH = 'path/to/your/input'
OUTPUT_PATH = 'path/to/output'
PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'
```

### Solution 3: Create a Custom vLLM Server Script

Create a file `serve_deepseek_ocr.py`:

```python
#!/usr/bin/env python3
"""
Custom vLLM server for DeepSeek-OCR with proper model registration.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Now import and run vLLM serve
from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    # Configure engine
    engine_args = AsyncEngineArgs(
        model=args.model,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        trust_remote_code=True,
    )
    
    print(f"Starting DeepSeek-OCR server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    
    # Start server
    run_server(engine_args, host=args.host, port=args.port)
```

Run it:
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python serve_deepseek_ocr.py --model deepseek-ai/DeepSeek-OCR --port 8000
```

## Installation Requirements

Make sure you have all dependencies installed:

```bash
# Install vLLM (version 0.8.5 or later)
pip install vllm>=0.8.5

# Install other requirements
pip install -r requirements.txt

# Install flash attention
pip install flash-attn==2.7.3 --no-build-isolation
```

## Troubleshooting

### Issue: "Model architectures 'DeepseekOCRForCausallM' are not supported"

**Cause**: Typo in model architecture name or model not registered.

**Solution**: Use one of the solutions above to properly register the model before serving.

### Issue: "Module not found: deepseek_ocr"

**Cause**: Python can't find the DeepSeek-OCR-vllm module.

**Solution**: Make sure you're running from the correct directory or add it to PYTHONPATH:
```bash
export PYTHONPATH="/path/to/DeepSeek-OCR-master/DeepSeek-OCR-vllm:$PYTHONPATH"
```

### Issue: CUDA out of memory

**Solution**: Reduce GPU memory utilization or use smaller image sizes:
```python
# In config.py
BASE_SIZE = 640  # Instead of 1024
IMAGE_SIZE = 512  # Instead of 640
MAX_CROPS = 4    # Instead of 6
```

## API Usage Example

Once the server is running, you can use it via HTTP API:

```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
image = Image.open("document.png")
buffered = io.BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "deepseek-ai/DeepSeek-OCR",
        "prompt": "<image>\\nFree OCR.",
        "images": [img_str],
        "max_tokens": 8192,
        "temperature": 0.0
    }
)

print(response.json()["choices"][0]["text"])
```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [DeepSeek-OCR HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

## Support

For issues and questions:
- GitHub Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- Discord: https://discord.gg/Tc7c45Zzu5
