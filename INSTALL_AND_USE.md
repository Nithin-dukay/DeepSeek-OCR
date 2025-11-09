# Installation and Usage Guide - Issue #191 Fix

## Prerequisites

- Python 3.8+
- CUDA-capable GPU
- PyTorch 2.0+
- Transformers 4.46+

## Installation

### Step 1: Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install Transformers and other dependencies
pip install transformers==4.46.3 tokenizers==0.20.3
pip install Pillow numpy einops easydict addict

# Install flash-attention (optional but recommended)
pip install flash-attn==2.7.3 --no-build-isolation
```

### Step 2: Download the Fix Files

Download these files to your project directory:
- `transformers_logits_processor.py` (required)
- `quick_fix_issue_191.py` (for quick fix)
- `run_dpsk_ocr_fixed.py` (for CLI tool)

Or clone the repository:
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
# Copy the fix files to this directory
```

## Quick Start (3 Options)

### Option 1: Quick Fix (Easiest)

**Best for**: Minimal changes to existing code

1. Copy `transformers_logits_processor.py` and `quick_fix_issue_191.py` to your project
2. Modify your existing code:

```python
# Add this import
from quick_fix_issue_191 import infer_with_fix

# Replace this:
# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, ...)

# With this:
res = infer_with_fix(model, tokenizer, prompt=prompt, image_file=image_file,
                     base_size=1024, image_size=640, crop_mode=True,
                     ngram_size=30, window_size=90)
```

3. Run your script as normal

**Complete example**:
```python
from transformers import AutoModel, AutoTokenizer
import torch
import os
from quick_fix_issue_191 import infer_with_fix

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)

# Run OCR with fix
res = infer_with_fix(
    model, tokenizer,
    prompt="<image>\n<|grounding|>OCR this image. ",
    image_file='your_image.jpg',
    output_path='./output',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    ngram_size=40,      # Increase for difficult documents
    window_size=120     # Increase for long documents
)

print(res)
```

### Option 2: CLI Tool (Most Convenient)

**Best for**: Command-line usage, batch processing

1. Copy `transformers_logits_processor.py` and `run_dpsk_ocr_fixed.py` to your project
2. Run from command line:

```bash
# Basic usage
python run_dpsk_ocr_fixed.py --image your_image.jpg --output ./results

# For ancient/handwritten documents
python run_dpsk_ocr_fixed.py \
    --image ancient_document.jpg \
    --preset large \
    --prompt ocr_image \
    --ngram 40 \
    --window 120 \
    --output ./results

# For modern documents
python run_dpsk_ocr_fixed.py \
    --image modern_doc.jpg \
    --preset gundam \
    --prompt markdown \
    --output ./results

# See all options
python run_dpsk_ocr_fixed.py --help
```

### Option 3: Custom Integration (Most Flexible)

**Best for**: Custom pipelines, advanced usage

1. Copy `transformers_logits_processor.py` to your project
2. Import and use in your code:

```python
from transformers import AutoModel, AutoTokenizer
import torch
from transformers_logits_processor import create_anti_hallucination_processors

# Load model
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)

# Create processors
processors = create_anti_hallucination_processors(
    mode="standard",
    ngram_size=30,
    window_size=90,
    whitelist_token_ids={128821, 128822}
)

# Patch model.generate to use processors
original_generate = model.generate

def patched_generate(*args, **kwargs):
    kwargs['logits_processor'] = kwargs.get('logits_processor', []) + processors
    kwargs.setdefault('max_new_tokens', 8192)
    kwargs.setdefault('temperature', 0.0)
    return original_generate(*args, **kwargs)

model.generate = patched_generate

# Now use model.infer() as normal
res = model.infer(tokenizer, prompt="<image>\nFree OCR. ",
                  image_file='your_image.jpg', output_path='./output')
```

## Configuration Guide

### Resolution Presets

| Preset | Command | Resolution | Use Case |
|--------|---------|------------|----------|
| Tiny | `--preset tiny` | 512×512 | Testing, simple docs |
| Small | `--preset small` | 640×640 | Balanced |
| Base | `--preset base` | 1024×1024 | Standard quality |
| Large | `--preset large` | 1280×1280 | Best quality, handwritten |
| Gundam | `--preset gundam` | Dynamic | Large documents |

### Prompt Templates

| Template | Command | Use Case |
|----------|---------|----------|
| free_ocr | `--prompt free_ocr` | No layout preservation |
| markdown | `--prompt markdown` | Structured documents |
| ocr_image | `--prompt ocr_image` | Handwritten, non-standard |
| parse_figure | `--prompt parse_figure` | Charts and figures |
| describe | `--prompt describe` | General description |

### Anti-Hallucination Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `--ngram` | 20-50 | 30 | N-gram size for blocking |
| `--window` | 60-150 | 90 | Sliding window size |
| `--mode` | standard/adaptive | standard | Processor mode |

**Tuning guide**:
- **Increase ngram** (35-40) for: Ancient documents, handwritten text, ambiguous characters
- **Increase window** (100-120) for: Long documents, complex layouts
- **Use adaptive mode** for: Very long documents, uncertain optimal settings

## Common Use Cases

### Ancient Handwritten Documents

```bash
python run_dpsk_ocr_fixed.py \
    --image ancient_manuscript.jpg \
    --preset large \
    --prompt ocr_image \
    --ngram 40 \
    --window 120 \
    --output ./results
```

**Why these settings**:
- `large`: Maximum resolution for detail
- `ocr_image`: Better for non-standard layouts
- `ngram 40`: Higher tolerance for ambiguous characters
- `window 120`: Larger context for complex patterns

### Modern Printed Documents

```bash
python run_dpsk_ocr_fixed.py \
    --image modern_doc.jpg \
    --preset gundam \
    --prompt markdown \
    --ngram 30 \
    --window 90 \
    --output ./results
```

**Why these settings**:
- `gundam`: Efficient dynamic resolution
- `markdown`: Preserves structure
- Standard parameters work well

### Low-Quality Scans

```bash
python run_dpsk_ocr_fixed.py \
    --image poor_scan.jpg \
    --preset large \
    --prompt ocr_image \
    --ngram 35 \
    --window 100 \
    --output ./results
```

**Why these settings**:
- `large`: Compensates for quality loss
- `ngram 35`: Slightly higher tolerance for noise
- `window 100`: More context for disambiguation

### Tables and Structured Data

```bash
python run_dpsk_ocr_fixed.py \
    --image table.jpg \
    --preset base \
    --prompt markdown \
    --ngram 25 \
    --window 90 \
    --output ./results
```

**Why these settings**:
- `base`: Sufficient for structured data
- `markdown`: Preserves table structure
- `ngram 25`: Lower to allow cell repetition (whitelisted)

## Troubleshooting

### Problem: Still getting hallucination

**Solutions**:
1. Increase `--ngram` to 40-50
2. Increase `--window` to 120-150
3. Use `--preset large` for higher resolution
4. Try `--prompt ocr_image` instead of `free_ocr`
5. Preprocess image (enhance contrast, remove noise)
6. Use `--mode adaptive`

### Problem: Output seems constrained or incomplete

**Solutions**:
1. Decrease `--ngram` to 20-25
2. Decrease `--window` to 60-80
3. Check if valid repetition is being blocked
4. Try `--mode standard` instead of adaptive

### Problem: Out of memory

**Solutions**:
1. Use smaller preset: `--preset small` or `--preset base`
2. Reduce window: `--window 60`
3. Use `--no-crop` flag
4. Process smaller images

### Problem: Slow performance

**Solutions**:
1. Use smaller preset: `--preset small`
2. Reduce window: `--window 60`
3. Use `--mode standard` (avoid adaptive)
4. Ensure CUDA is properly configured

### Problem: Import errors

**Solutions**:
```bash
# Ensure all files are in the same directory
ls transformers_logits_processor.py
ls quick_fix_issue_191.py  # or run_dpsk_ocr_fixed.py

# Check Python path
python -c "import sys; print(sys.path)"

# Add current directory to path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Testing Your Installation

### Quick Test

```python
# test_installation.py
from transformers_logits_processor import create_anti_hallucination_processors
import torch

# Test processor creation
processors = create_anti_hallucination_processors(mode="standard")
print("✓ Processor created successfully")

# Test with dummy data
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
scores = torch.zeros((1, 100))
result = processors[0](input_ids, scores)
print("✓ Processor works correctly")

print("\n✅ Installation successful!")
```

Run:
```bash
python test_installation.py
```

### Full Test

```bash
# Run the comprehensive test suite
python test_fix.py
```

Expected output:
```
================================================================================
Testing NoRepeatNGramLogitsProcessor for Issue #191 Fix
================================================================================
...
✅ All tests passed! The fix is working correctly.
```

## Getting Help

1. **Documentation**:
   - `README_FIX_191.md` - Overview and quick start
   - `FIX_GUIDE.md` - Comprehensive guide
   - `ISSUE_191_ANALYSIS.md` - Technical details

2. **Examples**:
   - `quick_fix_issue_191.py` - Minimal example
   - `run_dpsk_ocr_fixed.py` - Full-featured example

3. **Testing**:
   - `test_fix.py` - Test suite
   - `verify_fix.py` - Verification script

4. **Support**:
   - Check troubleshooting section above
   - Review documentation files
   - Report issues on GitHub

## Next Steps

1. ✅ Install dependencies
2. ✅ Download fix files
3. ✅ Choose integration method
4. ✅ Test with your documents
5. ✅ Adjust parameters as needed
6. ✅ Provide feedback

---

**Quick Reference Card**

```bash
# Ancient handwritten documents
python run_dpsk_ocr_fixed.py --image doc.jpg --preset large --prompt ocr_image --ngram 40 --window 120

# Modern documents
python run_dpsk_ocr_fixed.py --image doc.jpg --preset gundam --prompt markdown

# Low quality scans
python run_dpsk_ocr_fixed.py --image doc.jpg --preset large --ngram 35 --window 100

# Tables
python run_dpsk_ocr_fixed.py --image table.jpg --preset base --prompt markdown --ngram 25
```

---

**Status**: Ready to use ✅  
**Last Updated**: 2025-11-09
