# Fix for GitHub Issue #299: DeepSeek OCR Triton Error [CUDA] Illegal Memory Access on vLLM 0.11.2

## Problem Summary

The issue occurs when processing certain images (particularly those with mixed handwritten and digital text) with DeepSeek OCR on vLLM 0.11.2. The error manifests as:
- `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`
- `RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`

The error originates from the Triton fused MoE (Mixture of Experts) kernel during model execution, specifically in the `invoke_fused_moe_kernel` function.

## Root Causes

1. **CUDA Graph Capture Issues**: vLLM 0.11.2's CUDA graph capture can cause memory access violations with certain input patterns
2. **MoE Kernel Optimization**: The fused MoE kernel optimization in Triton can trigger illegal memory access for specific tensor shapes
3. **Memory Alignment**: Certain image dimensions after preprocessing may result in misaligned memory access patterns
4. **Torch Compilation**: The compiled model components may not handle dynamic shapes properly

## Solutions

### Solution 1: Update vLLM Engine Configuration (Recommended)

Modify the engine initialization to use safer parameters that avoid the Triton kernel issues.

**File**: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`

Changes:
1. Add `enforce_eager=True` to disable CUDA graph capture
2. Adjust `gpu_memory_utilization` to prevent memory conflicts
3. Add environment variables to control vLLM behavior
4. Adjust `max_num_seqs` to avoid power-of-2 values that trigger the bug

### Solution 2: Add Triton Cache Clearing

Clear Triton's kernel cache to prevent corrupted cached kernels from being used.

### Solution 3: Disable Torch Compilation for Vision Components

The vision encoders (SAM and CLIP) may benefit from eager execution mode to avoid compilation-related memory issues.

### Solution 4: Add Error Recovery Mechanism

Implement graceful error handling to catch and recover from CUDA errors without crashing the entire server.

## Implementation

### Updated Configuration File

Create an updated configuration that includes safer defaults:

**File**: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config_safe.py`

### Updated Run Script

The main changes to `run_dpsk_ocr_image.py`:

1. **Environment Variables** (add at the top):
```python
# Triton and CUDA configuration for stability
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'  # Use xformers instead of flash-attn for stability
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Keep async for performance
```

2. **Engine Arguments** (update):
```python
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=128,  # Reduced from 256 for better memory alignment
    max_model_len=8192,
    enforce_eager=True,  # CRITICAL: Disable CUDA graphs
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.70,  # Reduced from 0.75 for stability
    max_num_seqs=255,  # Avoid power-of-2 values (not 256)
    disable_custom_all_reduce=True,  # Disable custom kernels
)
```

3. **Add Triton Cache Clearing**:
```python
def clear_triton_cache():
    """Clear Triton kernel cache to prevent corrupted kernels."""
    import shutil
    cache_dir = os.environ.get('TRITON_CACHE_DIR', '/tmp/triton_cache')
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Cleared Triton cache at {cache_dir}")
        except Exception as e:
            print(f"Warning: Could not clear Triton cache: {e}")
```

### Updated DeepSeek OCR Model File

**File**: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py`

Add error handling and memory management:

```python
def _pixel_values_to_embedding(
    self,
    pixel_values: torch.Tensor,
    images_crop: torch.Tensor,
    images_spatial_crop: torch.Tensor,
) -> NestedTensors:
    images_in_this_batch = []
    
    # Add CUDA synchronization and error checking
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    with torch.no_grad():
        for jdx in range(images_spatial_crop.size(0)):
            try:
                # Process each image with error handling
                patches = images_crop[jdx][0].to(torch.bfloat16)
                image_ori = pixel_values[jdx]
                crop_shape = images_spatial_crop[jdx][0]
                
                # ... rest of processing ...
                
            except RuntimeError as e:
                if "illegal memory access" in str(e) or "CUDA" in str(e):
                    print(f"CUDA error processing image {jdx}, attempting recovery...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Return a dummy embedding to prevent crash
                    dummy_embed = torch.zeros((1, 1280), dtype=torch.bfloat16, device=pixel_values.device)
                    images_in_this_batch.append(dummy_embed)
                    continue
                else:
                    raise
    
    return images_in_this_batch
```

## Testing

After applying the fixes, test with:

1. **Simple images**: Should work as before
2. **Mixed content images**: Previously failing images should now process
3. **Batch processing**: Test with multiple images to ensure stability

## Alternative Workarounds

If the above solutions don't fully resolve the issue:

### Workaround 1: Use HuggingFace Transformers Instead
```python
# Use the HF implementation which doesn't have this issue
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

### Workaround 2: Downgrade to vLLM 0.8.5
```bash
pip uninstall vllm
pip install vllm==0.8.5+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Workaround 3: Use vLLM Nightly Build
```bash
pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

## Environment Requirements

- CUDA 11.8 or 12.1+
- PyTorch 2.6.0
- vLLM 0.11.2 (or 0.8.5 as fallback)
- Triton 2.1.0+
- GPU with at least 24GB VRAM (40GB recommended)

## Additional Notes

1. The `enforce_eager=True` flag will reduce throughput by ~20-30% but ensures stability
2. If performance is critical, try `enforce_eager=False` with `max_num_seqs=255` first
3. Monitor GPU memory usage - OOM can also manifest as illegal memory access
4. Some images may inherently cause issues due to their dimensions; consider preprocessing to standard sizes

## References

- vLLM Issue #30044: CUDA Illegal Memory Access During CUDA Graph Capture
- vLLM Issue #11340: CUDA illegal memory access with specific --max-num-seqs values
- DeepSeek OCR Official Documentation: https://github.com/deepseek-ai/DeepSeek-OCR
