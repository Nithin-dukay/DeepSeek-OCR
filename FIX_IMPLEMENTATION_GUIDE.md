# GitHub Issue #299 Fix Implementation Guide

## Quick Start

If you're experiencing Triton/CUDA illegal memory access errors with DeepSeek OCR on vLLM 0.11.2, follow these steps:

### Option 1: Use the Fixed Scripts (Recommended)

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# For single image processing
python run_dpsk_ocr_image_fixed.py

# For batch processing
python run_dpsk_ocr_eval_batch_fixed.py
```

### Option 2: Use Safe Configuration

```bash
# Copy the safe config
cp config_safe.py config.py

# Then run your scripts normally
python run_dpsk_ocr_image.py
```

### Option 3: Upgrade to vLLM Nightly (Best Long-term Solution)

```bash
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

---

## What Was Fixed

### Root Cause

The error occurs in vLLM's **fused_moe (Mixture of Experts) Triton kernel** when:
1. Processing images with complex features (mixed handwritten/digital text)
2. Large batch sizes or token counts
3. Specific tensor shapes that trigger integer overflow in pointer arithmetic

The error manifests as:
```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

Or sometimes:
```
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED
```

### The Fix

The solution involves multiple layers of protection:

#### 1. **Engine Configuration Changes**
- **`enforce_eager=True`**: Disables CUDA graphs, which bypasses the buggy Triton kernel path
- **`gpu_memory_utilization=0.75`**: Reduced from default 0.9 to prevent memory pressure
- **`enable_prefix_caching=False`**: Reduces memory usage
- **`block_size=256`**: Better memory alignment

#### 2. **Processing Parameters**
- **`MAX_CROPS=4`**: Reduced from 6 to limit workload on MoE kernel
- **`MAX_CONCURRENCY=50`**: Reduced from 100 for safer batch processing

#### 3. **Error Handling**
- Automatic retry with progressively safer parameters
- Graceful degradation (reduce max_model_len, GPU memory)
- Clear error messages and suggestions

---

## Files Created

### 1. `run_dpsk_ocr_image_fixed.py`
Enhanced version of the original script with:
- Automatic retry logic (3 attempts with different parameters)
- Better error messages
- CUDA cache clearing between retries
- Progress indicators

**Usage:**
```bash
# Set INPUT_PATH and OUTPUT_PATH in config.py first
python run_dpsk_ocr_image_fixed.py
```

### 2. `run_dpsk_ocr_eval_batch_fixed.py`
Batch processing script with:
- Sequential processing (one image at a time)
- Per-image error handling
- Detailed statistics and logging
- JSON summary of results

**Usage:**
```bash
# Set INPUT_PATH (directory) and OUTPUT_PATH in config.py
python run_dpsk_ocr_eval_batch_fixed.py
```

### 3. `config_safe.py`
Conservative configuration with:
- Reduced MAX_CROPS (4 instead of 6)
- Reduced MAX_CONCURRENCY (50 instead of 100)
- Documented safe parameters
- Troubleshooting guide

**Usage:**
```bash
cp config_safe.py config.py
```

### 4. `ISSUE_299_FIX.md`
Comprehensive documentation including:
- Problem analysis
- Multiple solution approaches
- Root cause explanation
- Prevention strategies

---

## Testing the Fix

### Test 1: Single Problematic Image

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Edit config.py to set:
# INPUT_PATH = '/path/to/problematic/image.jpg'
# OUTPUT_PATH = '/path/to/output'

python run_dpsk_ocr_image_fixed.py
```

**Expected behavior:**
- First attempt with `enforce_eager=True`
- If it fails, automatic retry with reduced parameters
- Clear error messages if all attempts fail

### Test 2: Batch Processing

```bash
# Edit config.py to set:
# INPUT_PATH = '/path/to/image/directory'
# OUTPUT_PATH = '/path/to/output'

python run_dpsk_ocr_eval_batch_fixed.py
```

**Expected behavior:**
- Processes images sequentially
- Continues even if some images fail
- Generates `processing_summary.json` with statistics

### Test 3: Monitor GPU Memory

```bash
# In one terminal
watch -n 1 nvidia-smi

# In another terminal
python run_dpsk_ocr_image_fixed.py
```

**What to look for:**
- GPU memory should stay below 80% utilization
- No "out of memory" errors
- Memory should be freed between retries

---

## Troubleshooting

### Issue: Still getting illegal memory access errors

**Solutions:**
1. Further reduce MAX_CROPS in config.py:
   ```python
   MAX_CROPS = 2  # or even 1
   ```

2. Use smaller image sizes:
   ```python
   BASE_SIZE = 640
   IMAGE_SIZE = 640
   CROP_MODE = False
   ```

3. Reduce max_model_len in the script:
   ```python
   max_model_len = 2048  # instead of 8192
   ```

### Issue: Out of memory errors

**Solutions:**
1. Reduce GPU memory utilization:
   ```python
   gpu_memory_utilization = 0.5  # instead of 0.75
   ```

2. Reduce max_model_len:
   ```python
   max_model_len = 4096  # instead of 8192
   ```

3. Process images at lower resolution

### Issue: Slow processing

**Solutions:**
1. If using `enforce_eager=True`, processing will be slower but more stable
2. For production, upgrade to vLLM nightly which has kernel fixes
3. Consider using the HuggingFace Transformers backend for problematic images:
   ```bash
   cd ../DeepSeek-OCR-hf
   python run_dpsk_ocr.py
   ```

---

## Performance Comparison

| Configuration | Speed | Stability | Memory Usage |
|--------------|-------|-----------|--------------|
| Original (CUDA graphs) | Fast | Unstable | High |
| Fixed (enforce_eager) | Medium | Stable | Medium |
| HF Transformers | Slow | Very Stable | Low |
| vLLM Nightly | Fast | Stable | Medium |

---

## Migration Path

### For Development/Testing
Use the fixed scripts immediately:
```bash
python run_dpsk_ocr_image_fixed.py
```

### For Production
1. **Short-term:** Use fixed scripts with `enforce_eager=True`
2. **Medium-term:** Upgrade to vLLM nightly build
3. **Long-term:** Wait for vLLM stable release with MoE kernel fixes

---

## Environment Variables Reference

```bash
# Required for CUDA 11.8
export TRITON_PTXAS_PATH=/usr/local/cuda-11.8/bin/ptxas

# Use V0 engine (more stable)
export VLLM_USE_V1=0

# Better error messages
export CUDA_LAUNCH_BLOCKING=1

# For debugging Triton issues
export TRITON_DEBUG=1

# Set visible GPU
export CUDA_VISIBLE_DEVICES=0
```

---

## Code Changes Summary

### Key Modifications

1. **AsyncEngineArgs parameters:**
   ```python
   enforce_eager=True,              # NEW: Disable CUDA graphs
   gpu_memory_utilization=0.75,     # CHANGED: from 0.9
   enable_prefix_caching=False,     # NEW: Reduce memory
   block_size=256,                  # NEW: Better alignment
   ```

2. **Retry logic:**
   ```python
   async def stream_generate(..., retry_count=0, max_retries=2):
       try:
           # Process image
       except RuntimeError as e:
           if "illegal memory access" in str(e):
               if retry_count < max_retries:
                   # Retry with safer parameters
   ```

3. **Configuration:**
   ```python
   MAX_CROPS = 4              # CHANGED: from 6
   MAX_CONCURRENCY = 50       # CHANGED: from 100
   ```

---

## Additional Resources

- **vLLM Issue #5938:** Illegal memory access for MoE kernel
- **vLLM PR #13693:** BugFix for illegal memory access
- **DeepSeek-OCR Docs:** https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html

---

## Support

If you continue to experience issues:

1. **Collect debug information:**
   ```bash
   CUDA_LAUNCH_BLOCKING=1 TRITON_DEBUG=1 python run_dpsk_ocr_image_fixed.py 2>&1 | tee debug.log
   ```

2. **Check GPU info:**
   ```bash
   nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
   ```

3. **Report with:**
   - GPU model and driver version
   - vLLM version (`pip show vllm`)
   - Image characteristics (size, complexity)
   - Full error traceback from debug.log

---

## Success Criteria

The fix is working correctly if:
- ✅ No "illegal memory access" errors
- ✅ No "CUBLAS_STATUS_EXECUTION_FAILED" errors
- ✅ Images process successfully (may be slower with enforce_eager)
- ✅ GPU memory stays below 80%
- ✅ Consistent results across multiple runs

---

## License

This fix is provided as-is for the DeepSeek-OCR project. Follow the original project's license terms.
