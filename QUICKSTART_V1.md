# Quick Start Guide: DeepSeek-OCR with vLLM 0.11.0 v1 Engine

## Installation

```bash
# Install vLLM 0.11.0
pip install vllm==0.11.0

# Install dependencies
pip install PyMuPDF img2pdf einops easydict addict Pillow

# Install flash attention (optional but recommended)
pip install flash_attn==2.8.1 --no-build-isolation
```

## Configuration

Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

```python
# Set your model path
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'  # or local path

# Set input/output paths
INPUT_PATH = '/path/to/your/image.jpg'
OUTPUT_PATH = '/path/to/output/directory'

# Choose your prompt
PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'
# or
# PROMPT = '<image>\\nFree OCR.'
```

## Usage

### Basic Image OCR

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

### What's Different in v1?

The v1 engine changes are **transparent to end users**. The script automatically:

1. ✅ Enables v1 engine via environment variable
2. ✅ Registers the v1-compatible logits processor
3. ✅ Processes images in the correct order
4. ✅ Passes parameters via the new `extra_args` mechanism

### Switching Between v0 and v1

To use the **legacy v0 engine** (if you have vLLM < 0.11.0):

Edit `run_dpsk_ocr_image.py` line 9:

```python
# For v0 engine (vLLM < 0.11.0)
os.environ['VLLM_USE_V1'] = '0'

# For v1 engine (vLLM >= 0.11.0)
os.environ['VLLM_USE_V1'] = '1'
```

## Example Output

```
=====================
BASE:  torch.Size([1, 256, 1280])
PATCHES:  torch.Size([6, 100, 1280])
=====================

# Document Title

This is the OCR output from your document...

===============save results:===============
```

Results are saved to your `OUTPUT_PATH`:
- `result.mmd` - Markdown output with images replaced
- `result_ori.mmd` - Original output with bounding boxes
- `result_with_boxes.jpg` - Image with detected regions highlighted
- `images/` - Extracted image regions

## Supported Modes

The model supports multiple resolution modes (configured in `config.py`):

| Mode | BASE_SIZE | IMAGE_SIZE | CROP_MODE | Vision Tokens |
|------|-----------|------------|-----------|---------------|
| Tiny | 512 | 512 | False | 64 |
| Small | 640 | 640 | False | 100 |
| Base | 1024 | 1024 | False | 256 |
| Large | 1280 | 1280 | False | 400 |
| Gundam | 1024 | 640 | True | Variable |

**Recommended:** Gundam mode (default) for best quality/performance balance.

## Common Prompts

```python
# Document to Markdown
PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'

# General OCR
PROMPT = '<image>\\nFree OCR.'

# OCR with layout
PROMPT = '<image>\\n<|grounding|>OCR this image.'

# Parse figures
PROMPT = '<image>\\nParse the figure.'

# Detailed description
PROMPT = '<image>\\nDescribe this image in detail.'

# Locate specific text
PROMPT = '<image>\\nLocate <|ref|>your text here<|/ref|> in the image.'
```

## Performance Tips

1. **GPU Memory:** Adjust `gpu_memory_utilization` in the script (default: 0.75)
2. **Batch Processing:** For multiple images, see `run_dpsk_ocr_pdf.py` or `run_dpsk_ocr_eval_batch.py`
3. **Crop Mode:** Enable for large documents (better quality, more tokens)
4. **Max Crops:** Reduce `MAX_CROPS` in config.py if you have limited GPU memory

## Troubleshooting

### "CUDA out of memory"
- Reduce `gpu_memory_utilization` to 0.5 or 0.6
- Reduce `MAX_CROPS` in config.py
- Use a smaller resolution mode (Small or Base instead of Gundam)

### "Module not found: vllm.v1"
- Ensure vLLM 0.11.0 or later is installed: `pip install --upgrade vllm==0.11.0`

### "No output generated"
- Check that `INPUT_PATH` and `OUTPUT_PATH` are set correctly
- Verify the image file exists and is readable
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### Slow inference
- Enable flash attention: `pip install flash_attn==2.8.1 --no-build-isolation`
- Reduce image resolution or disable crop mode
- Ensure `enforce_eager=False` in engine args (default)

## Advanced Usage

### Custom N-gram Parameters

Edit the `extra_args` in `run_dpsk_ocr_image.py`:

```python
extra_args={
    "ngram_size": 30,      # Size of n-grams to check (default: 30)
    "window_size": 90,     # Window for checking repetitions (default: 90)
    "whitelist_token_ids": {128821, 128822}  # Tokens to exclude from filtering
}
```

### Custom Logits Processor

Create your own adapter in `process/` directory:

```python
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

class MyCustomAdaptor(AdapterLogitsProcessor):
    def is_argmax_invariant(self) -> bool:
        return True
    
    def new_req_logits_processor(self, params):
        # Your custom processor logic
        return MyCustomProcessor(**params.extra_args)
```

Register it in engine args:
```python
logits_processors=["process.my_module:MyCustomAdaptor"]
```

## Getting Help

- **Documentation:** See `VLLM_V1_MIGRATION_GUIDE.md` for detailed technical information
- **Changes:** See `CHANGES_SUMMARY.md` for a summary of all modifications
- **Issues:** Report bugs on GitHub: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- **Discord:** Join the DeepSeek AI community

## What's Next?

- Try different prompts for various use cases
- Process PDFs with `run_dpsk_ocr_pdf.py`
- Batch process images with `run_dpsk_ocr_eval_batch.py`
- Integrate into your own applications using the provided examples

---

**Note:** All changes are backward compatible. If you need to use vLLM < 0.11.0, simply set `VLLM_USE_V1='0'` and the code will work with the legacy v0 engine.
