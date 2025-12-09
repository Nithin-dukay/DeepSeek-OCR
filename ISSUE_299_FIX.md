# Fix for GitHub Issue #299: DeepSeek OCR Triton Error [CUDA] Illegal Memory Access on vLLM 0.11.2

## Problem Summary

The issue occurs when processing certain images (particularly those with mixed handwritten and digital text) with DeepSeek OCR on vLLM 0.11.2. The error manifests as:
- `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`
- Sometimes: `RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`

The root cause is in the **fused_moe (Mixture of Experts) Triton kernel** which can experience integer overflow or illegal memory access when processing certain workloads, particularly with:
- Large batch sizes or token counts
- Complex image features (mixed handwritten/digital text)
- Specific tensor shapes that trigger edge cases in the MoE kernel

## Solution Overview

The fix involves multiple approaches:

### 1. **Environment Configuration** (Immediate Workaround)
### 2. **Code Modifications** (Robust Fix)
### 3. **Runtime Parameters** (Optimization)

---

## Implementation

### Fix 1: Environment Variables and Configuration

Create or update the startup script to include proper environment variables and safer defaults.

**File: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image_fixed.py`**

Key changes:
- Set `VLLM_USE_V1=0` to use the more stable v0 engine
- Add `CUDA_LAUNCH_BLOCKING=1` for debugging
- Reduce `gpu_memory_utilization` to 0.75 (from default 0.9)
- Enable `enforce_eager=True` to disable CUDA graphs (workaround for Triton kernel issues)
- Reduce `max_model_len` if needed
- Add `block_size=256` for better memory alignment

### Fix 2: Update config.py with Safer Defaults

**File: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`**

Reduce `MAX_CROPS` from 6 to 4 or lower if GPU memory is limited. This reduces the workload on the MoE kernel.

### Fix 3: Add Error Handling and Retry Logic

Create a wrapper script that handles failures gracefully and can retry with different parameters.

### Fix 4: Batch Processing with Smaller Chunks

For batch processing, ensure images are processed in smaller batches to avoid overwhelming the MoE kernel.

---

## Detailed Code Changes

### 1. Create Fixed Runtime Script

This script includes all the necessary environment variables and safer parameters.

### 2. Update Configuration

Lower the maximum crops and concurrency to reduce memory pressure.

### 3. Add Fallback Mechanism

Implement a fallback that tries with `enforce_eager=True` if the first attempt fails.

---

## Testing the Fix

1. **Test with problematic images first:**
   ```bash
   python run_dpsk_ocr_image_fixed.py
   ```

2. **Monitor GPU memory:**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Check for errors in logs:**
   - Look for "illegal memory access" errors
   - Monitor CUDA out-of-memory errors

---

## Alternative Solutions

### Option A: Upgrade to vLLM Nightly (Recommended for Production)

```bash
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

The nightly builds include fixes for MoE kernel issues.

### Option B: Use Transformers Backend (Fallback)

If vLLM continues to have issues, fall back to the HuggingFace Transformers implementation:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

### Option C: Reduce Image Resolution

For problematic images, reduce the resolution before processing:
- Use `BASE_SIZE = 640` instead of `1024`
- Set `CROP_MODE = False` for simpler images

---

## Root Cause Analysis

The Triton kernel in vLLM's fused_moe implementation has an integer overflow issue when:

1. **Token count is high:** Mixed handwritten/digital text creates more complex features
2. **Batch size is large:** Multiple images or crops processed together
3. **Tensor shapes trigger edge cases:** Specific combinations of dimensions cause overflow in pointer arithmetic

The error occurs in:
```
File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py", line 693, in invoke_fused_moe_kernel
    fused_moe_kernel[grid](...)
```

The `off_experts * stride_be` calculation can overflow, leading to illegal memory access.

---

## Prevention

To prevent this issue in the future:

1. **Always use `enforce_eager=True`** for production until vLLM fixes the Triton kernel
2. **Monitor GPU memory utilization** and keep it below 80%
3. **Process images in smaller batches** (MAX_CONCURRENCY ≤ 50)
4. **Reduce MAX_CROPS** for complex images
5. **Update vLLM regularly** to get the latest bug fixes

---

## References

- vLLM Issue #5938: Illegal memory access for MoE kernel with large workloads
- vLLM PR #13693: BugFix for illegal memory access on H20
- DeepSeek-OCR Issue #240: RTX 5090 compatibility
- vLLM Issue #30044: CUDA illegal memory access during CUDA graph capture

---

## Support

If the issue persists after applying these fixes:

1. Collect debug information:
   ```bash
   CUDA_LAUNCH_BLOCKING=1 TRITON_DEBUG=1 python run_dpsk_ocr_image_fixed.py 2>&1 | tee debug.log
   ```

2. Check GPU compatibility:
   ```bash
   nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
   ```

3. Report to vLLM with:
   - GPU model and driver version
   - vLLM version
   - Image characteristics (size, complexity)
   - Full error traceback
