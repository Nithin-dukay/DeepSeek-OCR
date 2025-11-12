# Table Recognition with DeepSeek-OCR

This guide provides detailed information on using DeepSeek-OCR for table recognition and conversion to HTML or Markdown format.

## Quick Start

### Basic Usage

```python
from PIL import Image

# For vLLM
prompt = "<image>\n<|grounding|>Convert the table to HTML."

# Load your table image
image = Image.open("table.jpg").convert('RGB')

# Process with DeepSeek-OCR (see full example below)
```

### Using the Example Script

```bash
# Convert a table image to HTML
python table_to_html_example.py --image path/to/table.jpg --output result.html

# Use transformers backend instead of vLLM
python table_to_html_example.py --image table.jpg --output result.html --backend transformers
```

## Prompts for Table Recognition

### HTML Output
```python
# Basic table to HTML
prompt = "<image>\n<|grounding|>Convert the table to HTML."

# Multiple tables in one image
prompt = "<image>\n<|grounding|>Extract and convert all tables to HTML."
```

### Markdown Output
```python
# Table to Markdown
prompt = "<image>\n<|grounding|>Convert the table to markdown."

# Document with tables to Markdown
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```

## Important Configuration

### Whitelist Tokens for HTML Tables

When converting tables to HTML, you **must** whitelist the `<td>` and `</td>` tokens to allow their repetition:

```python
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Token IDs for table elements
# 128821 = <td>
# 128822 = </td>

logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,           # Size of n-gram to check for repetition
        window_size=90,          # Window size for checking
        whitelist_token_ids={128821, 128822}  # Allow <td>, </td> repetition
    )
]
```

### Sampling Parameters

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0.0,              # Deterministic output
    max_tokens=8192,              # Maximum output length
    logits_processors=logits_processors,
    skip_special_tokens=False,    # Keep HTML tags in output
)
```

## Complete Examples

### vLLM Example

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from PIL import Image
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from deepseek_ocr import DeepseekOCRForCausalLM

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Initialize model
llm = LLM(
    model='deepseek-ai/DeepSeek-OCR',
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.75,
)

# Configure for table recognition
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822}  # <td>, </td>
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

# Process image
image = Image.open("table.jpg").convert('RGB')
image_features = DeepseekOCRProcessor().tokenize_with_images(
    images=[image],
    bos=True,
    eos=True,
    cropping=True
)

# Generate HTML
request = {
    "prompt": "<image>\n<|grounding|>Convert the table to HTML.",
    "multi_modal_data": {"image": image_features}
}

outputs = llm.generate([request], sampling_params=sampling_params)
html_output = outputs[0].outputs[0].text

print(html_output)
```

### Transformers Example

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# Run inference
prompt = "<image>\n<|grounding|>Convert the table to HTML."
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file="table.jpg",
    output_path="output",
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True
)

print(result)
```

## Tips for Best Results

1. **Use the `<|grounding|>` tag**: This enables structured output mode for better table recognition
2. **Enable cropping**: Set `crop_mode=True` for large or high-resolution images
3. **Whitelist table tokens**: Always whitelist `<td>` and `</td>` tokens (128821, 128822) for HTML output
4. **Adjust n-gram parameters**: 
   - Use `ngram_size=30` and `window_size=90` for most tables
   - For very large tables, you may increase `window_size`
5. **Image quality**: Higher resolution images generally produce better results
6. **Table complexity**: The model handles:
   - Simple tables with rows and columns
   - Tables with merged cells
   - Tables with headers
   - Nested tables (with appropriate prompts)

## Troubleshooting

### Issue: Repetitive output or truncated tables

**Solution**: Ensure you've whitelisted the table tokens:
```python
whitelist_token_ids={128821, 128822}  # <td>, </td>
```

### Issue: Missing table structure

**Solution**: Use the `<|grounding|>` tag in your prompt:
```python
prompt = "<image>\n<|grounding|>Convert the table to HTML."
```

### Issue: Poor quality output

**Solutions**:
- Increase image resolution
- Enable dynamic cropping: `crop_mode=True`
- Use appropriate base_size: `base_size=1024` for most cases
- Ensure good image quality (clear text, good contrast)

## Token IDs Reference

| Token | Token ID | Purpose |
|-------|----------|---------|
| `<td>` | 128821 | Table cell opening tag |
| `</td>` | 128822 | Table cell closing tag |
| `<tr>` | - | Table row opening tag |
| `</tr>` | - | Table row closing tag |

## Additional Resources

- Main README: [README.md](README.md)
- Example script: [table_to_html_example.py](table_to_html_example.py)
- vLLM configuration: [DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py](DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py)
- Paper: [DeepSeek_OCR_paper.pdf](DeepSeek_OCR_paper.pdf)

## Support

For issues and questions:
- GitHub Issues: [https://github.com/deepseek-ai/DeepSeek-OCR/issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
- Discord: [https://discord.gg/Tc7c45Zzu5](https://discord.gg/Tc7c45Zzu5)
