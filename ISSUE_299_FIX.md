# Fix for GitHub Issue #299: DeepSeek OCR Triton Error [CUDA] Illegal Memory Access on vLLM 0.11.2

## Problem Analysis

The error `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered` occurs in the MoE (Mixture of Experts) layer during the fused_moe_kernel execution. This is a known issue with vLLM 0.11.2 when processing certain images, particularly those with mixed handwritten and digital text.

### Root Causes:
1. **Memory Layout Issues**: The Triton kernel for MoE operations has memory alignment issues with certain input shapes
2. **CUDA Graph Compilation**: The default CUDA graph mode can cause memory access violations
3. **GPU Memory Pressure**: High GPU memory utilization (default 0.9) can lead to memory fragmentation
4. **Batch Size Issues**: Certain power-of-2 values for max_num_seqs can trigger the bug

## Solution

The fix involves multiple configuration changes and code modifications to work around the Triton/CUDA memory access issues:

### 1. Environment Variables (Critical)
Add these environment variables before importing vLLM:

```python
# Disable V1 engine which has known issues with DeepSeek models
os.environ['VLLM_USE_V1'] = '0'

# For multi-node setups (if applicable)
os.environ['VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE'] = 'shm'
```

### 2. Engine Configuration Changes
Modify the AsyncEngineArgs with these safer parameters:

```python
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    max_model_len=8192,
    
    # CRITICAL FIXES:
    enforce_eager=True,  # Disable CUDA graphs to avoid memory issues
    gpu_memory_utilization=0.75,  # Reduced from 0.9 to prevent fragmentation
    max_num_seqs=255,  # Avoid power-of-2 values (not 256)
    
    trust_remote_code=True,
    tensor_parallel_size=1,
)
```

### 3. Alternative: Piecewise CUDA Graph Mode
If you need CUDA graphs for performance, use piecewise mode instead:

```python
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    max_model_len=8192,
    enforce_eager=False,
    compilation_config='{"cudagraph_mode": "PIECEWISE"}',
    gpu_memory_utilization=0.75,
    max_num_seqs=255,
    trust_remote_code=True,
    tensor_parallel_size=1,
)
```

### 4. Config.py Adjustments
Reduce MAX_CROPS if you still encounter OOM errors:

```python
MAX_CROPS = 4  # Reduced from 6 for better memory stability
```

## Implementation Files

The following files have been created/modified:

1. **run_dpsk_ocr_image_fixed.py** - Fixed version with all workarounds
2. **run_dpsk_ocr_pdf_fixed.py** - Fixed PDF processing script
3. **run_dpsk_ocr_eval_batch_fixed.py** - Fixed batch evaluation script
4. **config_fixed.py** - Updated configuration with safer defaults

## Testing Recommendations

1. **Test with problematic images first**: Start with images that previously caused crashes
2. **Monitor GPU memory**: Use `nvidia-smi` to ensure memory usage stays below 80%
3. **Gradual rollout**: Test with single images before batch processing
4. **Fallback strategy**: Keep HuggingFace transformers version as backup

## Performance Impact

- **enforce_eager=True**: ~10-15% slower but more stable
- **gpu_memory_utilization=0.75**: Slightly reduced batch sizes but prevents crashes
- **max_num_seqs=255**: Minimal impact on throughput

## Alternative Solutions

If the issue persists:

1. **Downgrade to vLLM 0.8.5**: The README mentions this version works well
2. **Use HuggingFace Transformers**: Slower but more stable for problematic images
3. **Try SGLang**: Some users report better stability with DeepSeek models

## References

- vLLM Issue #14965: DeepSeek-R1 illegal memory access
- vLLM Issue #13824: vLLM 0.7.3 illegal memory access
- vLLM Issue #24272: Multi-node illegal memory access
- vLLM PR #13693: MoE illegal memory access bugfix

## Verification

After applying the fix:
1. Server should start without errors
2. Previously failing images should process successfully
3. No CUDA illegal memory access errors in logs
4. Stable operation over extended periods
