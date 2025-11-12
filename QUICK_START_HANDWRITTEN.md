# Quick Start Guide: Fixing Hallucinations in Handwritten OCR

## 🚨 Problem: Hallucinations on Handwritten Documents (Issue #191)

If you're experiencing hallucinations when using DeepSeek-OCR on handwritten or historical documents, follow this quick guide.

## ✅ Quick Fix (3 Steps)

### Step 1: Install Dependencies

```bash
pip install transformers torch pillow numpy scipy
```

### Step 2: Analyze Your Image

```bash
python DeepSeek-OCR-master/DeepSeek-OCR-hf/ocr_config_helper.py --image your_document.jpg
```

This will analyze your image and recommend optimal settings.

### Step 3: Run Optimized OCR

Use the command recommended by the helper, or use this template:

```bash
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --image your_document.jpg \
    --output results/ \
    --doc-type handwritten \
    --preprocess
```

**For ancient Portuguese documents specifically:**

```bash
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --image ancient_portuguese.jpg \
    --output results/ \
    --doc-type historical \
    --language Portuguese \
    --preprocess \
    --multi-pass
```

## 🔑 Key Changes to Fix Hallucinations

### ❌ WRONG (Causes Hallucinations)

```python
prompt = "<image>\nFree OCR."
```

### ✅ CORRECT (Minimizes Hallucinations)

```python
prompt = "<image>\n<|grounding|>Extract all text from this handwritten document."
```

## 📊 Comparison

| Aspect | Before (Free OCR) | After (Grounded OCR) |
|--------|-------------------|----------------------|
| Hallucinations | ❌ Frequent | ✅ Minimal |
| Accuracy | ❌ Inconsistent | ✅ High |
| Layout Awareness | ❌ None | ✅ Preserved |
| Historical Documents | ❌ Poor | ✅ Good |

## 🛠️ Available Tools

### 1. Configuration Helper (Recommended for First-Time Users)

```bash
# Interactive mode - walks you through the process
python DeepSeek-OCR-master/DeepSeek-OCR-hf/ocr_config_helper.py --interactive

# Analyze specific image
python DeepSeek-OCR-master/DeepSeek-OCR-hf/ocr_config_helper.py --image document.jpg

# List all available presets
python DeepSeek-OCR-master/DeepSeek-OCR-hf/ocr_config_helper.py --list-presets
```

### 2. Optimized Inference Script

```bash
# Single image
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --image document.jpg \
    --output results/ \
    --preprocess

# Batch processing (entire directory)
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --batch-dir images/ \
    --output results/ \
    --doc-type handwritten \
    --preprocess

# With validation (for critical documents)
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --image important.jpg \
    --output results/ \
    --multi-pass \
    --preprocess
```

### 3. Example Scripts

```bash
# Ancient Portuguese documents
python examples/example_handwritten_portuguese.py

# Batch processing
python examples/example_batch_processing.py

# Region-based OCR for complex layouts
python examples/example_region_based_ocr.py
```

## 📚 Documentation

- **[Troubleshooting Guide](TROUBLESHOOTING_HANDWRITTEN_OCR.md)** - Comprehensive guide with detailed explanations
- **[Solution Summary](SOLUTION_SUMMARY.md)** - Technical overview of the solution
- **[Examples README](examples/README.md)** - Detailed examples documentation

## 🎯 Common Use Cases

### Ancient/Historical Documents

```bash
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --image ancient_document.jpg \
    --output results/ \
    --doc-type historical \
    --preprocess \
    --multi-pass
```

### Modern Handwritten Notes

```bash
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --image handwritten_notes.jpg \
    --output results/ \
    --doc-type handwritten \
    --preprocess
```

### Mixed Content (Printed + Handwritten)

```bash
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --image mixed_document.jpg \
    --output results/ \
    --doc-type mixed \
    --preprocess
```

### Forms and Tables

```bash
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --image form.jpg \
    --output results/ \
    --doc-type form \
    --base-size 1280 \
    --image-size 1280 \
    --no-crop
```

## ⚙️ Optimal Parameters for Handwritten Documents

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--doc-type` | `handwritten` or `historical` | Document type |
| `--preprocess` | Enable | Enhance contrast and sharpness |
| `--base-size` | `1024` | Model base size |
| `--image-size` | `640` | Processing image size |
| `--crop-mode` | Enable (default) | Gundam mode for accuracy |
| `--multi-pass` | Enable for critical docs | Run validation pass |

## 🔍 Troubleshooting

### Still Getting Hallucinations?

1. ✅ Verify you're using `<|grounding|>` mode (not "Free OCR")
2. ✅ Enable preprocessing with `--preprocess`
3. ✅ Try `--multi-pass` for validation
4. ✅ Use `--doc-type historical` for old documents
5. ✅ Check image quality (should be >1000px on longest side)

### Poor Quality Images?

```bash
# Use aggressive preprocessing for historical documents
python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_handwritten.py \
    --image faded_document.jpg \
    --output results/ \
    --doc-type historical \
    --preprocess \
    --multi-pass
```

### Large Complex Documents?

```bash
# Use region-based processing
python examples/example_region_based_ocr.py
```

## 💡 Pro Tips

1. **Always use the configuration helper first** - It analyzes your image and recommends optimal settings
2. **Enable preprocessing for handwritten documents** - Significantly improves accuracy
3. **Use multi-pass validation for critical documents** - Compare results for consistency
4. **Specify language when known** - Helps the model focus on the correct character set
5. **Process large documents in regions** - Reduces hallucinations on complex layouts

## 📞 Getting Help

If you continue to experience issues:

1. Run the configuration helper to get recommendations
2. Review the [Troubleshooting Guide](TROUBLESHOOTING_HANDWRITTEN_OCR.md)
3. Check the [Examples](examples/) for similar use cases
4. Report issues with:
   - Image characteristics (resolution, quality)
   - Exact command used
   - Example output showing the issue

## 🎉 Expected Results

After following this guide, you should see:

- ✅ **Minimal to no hallucinations**
- ✅ **Consistent, reproducible results**
- ✅ **Better handling of faded/historical text**
- ✅ **Preserved layout and spatial information**
- ✅ **Improved accuracy on handwritten content**

---

**Quick Reference:** For ancient Portuguese handwritten documents (Issue #191), use:

```bash
python DeepSeek-OCR-master/DeepSeek-OCR-hf/ocr_config_helper.py --interactive
```

Then follow the recommended command. That's it! 🚀
