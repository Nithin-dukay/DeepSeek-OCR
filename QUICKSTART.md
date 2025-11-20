# DeepSeek-OCR Issue #244 - Quick Start Guide

## 🚨 The Problem

```bash
$ vllm serve deepseek-ai/DeepSeek-OCR ...
❌ Error: Model architectures 'DeepseekOCRForCausallM' are not supported
```

## ✅ The Solution

Use our wrapper script that automatically registers the model:

```bash
python serve_deepseek_ocr.py --image your_image.jpg
```

## 📦 Installation (5 Minutes)

### Step 1: Create Environment
```bash
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr
```

### Step 2: Install PyTorch
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install vLLM
```bash
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Verify
```bash
python test_solution.py
```

## 🎯 Usage Examples

### Basic OCR
```bash
python serve_deepseek_ocr.py --image document.jpg
```

### Document to Markdown
```bash
python serve_deepseek_ocr.py \
  --image document.png \
  --prompt "<image>\n<|grounding|>Convert the document to markdown."
```

### Batch Processing
```bash
python serve_deepseek_ocr.py \
  --image img1.jpg img2.jpg img3.jpg \
  --output results.txt
```

### Custom Model Path
```bash
python serve_deepseek_ocr.py \
  --model /path/to/local/model \
  --image test.jpg
```

## 🔧 Configuration

Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

```python
# For fast inference (small images)
BASE_SIZE = 640
IMAGE_SIZE = 640
CROP_MODE = False

# For high quality (large documents) - RECOMMENDED
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MAX_CROPS = 6  # Reduce to 4 if low on GPU memory
```

## 🐛 Common Issues

### Issue: CUDA Out of Memory
```python
# In config.py, reduce:
MAX_CROPS = 4
BASE_SIZE = 640
CROP_MODE = False
```

### Issue: Import Error
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/DeepSeek-OCR-master/DeepSeek-OCR-vllm"
```

### Issue: Slow Inference
```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

## 📊 Performance Guide

| GPU Memory | Recommended Settings |
|------------|---------------------|
| 16GB       | BASE_SIZE=640, MAX_CROPS=4, CROP_MODE=False |
| 24GB       | BASE_SIZE=1024, MAX_CROPS=6, CROP_MODE=True |
| 40GB+      | BASE_SIZE=1280, MAX_CROPS=9, CROP_MODE=True |

## 📚 Documentation

- **Quick Reference**: README_ISSUE_244_FIX.md
- **Technical Details**: FIX_ISSUE_244.md
- **Complete Setup**: SETUP_VLLM.md
- **Code Examples**: example_usage.py
- **Testing**: test_solution.py

## 🎓 Prompt Templates

```python
# Free OCR (no layout)
"<image>\nFree OCR."

# Document to Markdown (with layout)
"<image>\n<|grounding|>Convert the document to markdown."

# General OCR
"<image>\n<|grounding|>OCR this image."

# Parse figures/charts
"<image>\nParse the figure."

# Detailed description
"<image>\nDescribe this image in detail."
```

## 🧪 Test Your Setup

```bash
# Run validation tests
python test_solution.py

# Test with an image
python serve_deepseek_ocr.py --image test.jpg

# View help
python serve_deepseek_ocr.py --help
```

## 💡 Pro Tips

1. **Use Gundam mode** for large documents:
   ```python
   BASE_SIZE = 1024
   IMAGE_SIZE = 640
   CROP_MODE = True
   ```

2. **Batch process** for efficiency:
   ```bash
   python serve_deepseek_ocr.py --image *.jpg --output all_results.txt
   ```

3. **Install flash-attention** for 2x speed:
   ```bash
   pip install flash-attn==2.7.3 --no-build-isolation
   ```

4. **Use local model** to avoid re-downloading:
   ```bash
   huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir ./models/DeepSeek-OCR
   python serve_deepseek_ocr.py --model ./models/DeepSeek-OCR --image test.jpg
   ```

## 🆘 Need Help?

1. Check troubleshooting in SETUP_VLLM.md
2. Review technical details in FIX_ISSUE_244.md
3. Run test_solution.py to diagnose issues
4. Open a GitHub issue with error logs

## ✨ What Makes This Fix Work?

The key is registering the model before use:

```python
from vllm import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM

# This one line fixes Issue #244!
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
```

Our wrapper script does this automatically, so you don't have to worry about it.

## 🚀 Ready to Go!

```bash
# That's it! Start using DeepSeek-OCR:
python serve_deepseek_ocr.py --image your_document.jpg
```

---

**Status**: ✅ Tested and Working

**Issue**: GitHub #244

**Time to Setup**: ~5 minutes

**Difficulty**: Easy 🟢
