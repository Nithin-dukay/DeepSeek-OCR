# Quick Fix Guide for Issue #191: Hallucination During OCR

**Problem**: DeepSeek-OCR is hallucinating (repeating text) on your documents?

**Solution**: Use the improved inference script with anti-hallucination mechanisms.

## 🚀 Quick Fix (30 seconds)

### Step 1: Navigate to the HF directory
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
```

### Step 2: Run the improved script
```bash
python run_dpsk_ocr_improved.py \
    --image_file YOUR_IMAGE.jpg \
    --output_path ./output
```

**That's it!** Your OCR results will be in `./output/result.txt`

---

## 📋 Document Type Specific Commands

### For Handwritten Documents (Ancient manuscripts, handwritten notes)
```bash
python run_dpsk_ocr_improved.py \
    --image_file your_handwritten.jpg \
    --output_path ./output \
    --ngram_size 20 \
    --window_size 60
```

### For Printed Documents (Books, papers, reports)
```bash
python run_dpsk_ocr_improved.py \
    --image_file your_printed.jpg \
    --output_path ./output \
    --prompt_type grounding
```

### For Documents with Tables
```bash
python run_dpsk_ocr_improved.py \
    --image_file your_table.jpg \
    --output_path ./output \
    --prompt_type grounding \
    --ngram_size 40
```

---

## 🔧 Still Having Issues?

### If you're still seeing repetitions:
```bash
# Use more aggressive settings
python run_dpsk_ocr_improved.py \
    --image_file your_image.jpg \
    --output_path ./output \
    --ngram_size 15 \
    --window_size 50
```

### If you're running out of memory:
```bash
# Use smaller image size
python run_dpsk_ocr_improved.py \
    --image_file your_image.jpg \
    --output_path ./output \
    --base_size 640 \
    --no_crop_mode
```

---

## 🎯 Alternative: Use vLLM (Recommended for Production)

The vLLM implementation has anti-hallucination built-in and is faster:

```bash
cd ../DeepSeek-OCR-vllm

# Edit config.py to set your paths:
# INPUT_PATH = 'your_image.jpg'
# OUTPUT_PATH = './output'

python run_dpsk_ocr_image.py
```

---

## 📚 Need More Help?

- **Detailed Documentation**: See [HALLUCINATION_FIX.md](HALLUCINATION_FIX.md)
- **Usage Examples**: See [DeepSeek-OCR-master/DeepSeek-OCR-hf/USAGE_EXAMPLES.md](DeepSeek-OCR-master/DeepSeek-OCR-hf/USAGE_EXAMPLES.md)
- **Full Solution Summary**: See [ISSUE_191_SOLUTION_SUMMARY.md](ISSUE_191_SOLUTION_SUMMARY.md)

---

## ✅ What This Fix Does

- ✅ Prevents repetitive text generation (hallucinations)
- ✅ Uses greedy decoding for deterministic results
- ✅ Configurable for different document types
- ✅ Works with handwritten, printed, and degraded documents
- ✅ Maintains table structure and formatting

---

## 🆚 Before vs After

### Before (Original Script)
```
The quick brown fox jumps over the lazy dog. The quick brown fox jumps 
over the lazy dog. The quick brown fox jumps over the lazy dog. The 
quick brown fox jumps over the lazy dog. [HALLUCINATION CONTINUES...]
```

### After (Improved Script)
```
The quick brown fox jumps over the lazy dog. The dog was sleeping under 
a tree. [CORRECT OUTPUT CONTINUES...]
```

---

**Questions?** Open an issue on GitHub or check the documentation files listed above.

**Working?** Great! Consider using the vLLM implementation for even better performance.
