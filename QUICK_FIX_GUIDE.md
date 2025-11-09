# Quick Fix Guide for Issue #110

## 🚨 Error Message

```
ERROR 10-23 18:21:12 [core_client.py:597] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

## ⚡ Quick Fix (5 minutes)

### Option 1: Use Stable vLLM (RECOMMENDED)

```bash
# 1. Create environment
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# 2. Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Download and install vLLM 0.8.5
# Download from: https://github.com/vllm-project/vllm/releases/tag/v0.8.5
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# 4. Install dependencies
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation

# 5. Run your script
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

### Option 2: Use Fixed Scripts (2 minutes)

```bash
# 1. Navigate to vLLM directory
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# 2. Use the fixed script instead
python run_dpsk_ocr_image_fixed.py
```

### Option 3: Quick Patch (1 minute)

Edit your run script and comment out this line:

```python
# os.environ['VLLM_USE_V1'] = '0'  # Comment this out
```

## 🔍 Verify Fix

```bash
# Check your environment
python3 test_vllm_compatibility.py

# Should show all green checkmarks ✓
```

## 📚 More Information

- **Detailed Fix:** See `ISSUE_110_FIX.md`
- **Troubleshooting:** See `TROUBLESHOOTING.md`
- **Full Solution:** See `SOLUTION_SUMMARY.md`

## 💡 Why This Happens

vLLM nightly builds use v1 architecture, but DeepSeek-OCR is configured for v0. This creates a conflict that crashes the engine.

## ✅ Success Indicators

- No "Engine core proc died" errors
- Model loads successfully
- Inference produces output
- No CUDA errors

## ❌ Still Having Issues?

1. Check vLLM version: `python3 -c "import vllm; print(vllm.__version__)"`
2. Run diagnostics: `python3 test_vllm_compatibility.py`
3. See `TROUBLESHOOTING.md` for more help
4. Open a GitHub issue with your environment details

---

**Quick Tip:** Always use vLLM 0.8.5 for production. It's tested and stable!
