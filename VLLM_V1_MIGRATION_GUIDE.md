# vLLM 0.11.0 (v1 Engine) Migration Guide for DeepSeek-OCR

## Overview

This guide documents the changes made to enable DeepSeek-OCR compatibility with vLLM 0.11.0's new v1 engine architecture.

## Background

In vLLM 0.11.0, the legacy v0 engine has been internally redirected to the new v1 engine. The v1 engine introduces several breaking changes:

1. **Per-request logits processors are no longer supported directly**
2. **New `AdapterLogitsProcessor` interface** for global logits processors
3. **Changed import paths** for core components like `SamplingMetadata`
4. **Modified processor signatures** to accept tokenizer kwargs
5. **Strict initialization order** - processors must not be called before engine is ready

## Changes Made

### 1. New v1-Compatible Logits Processor Adapter

**File:** `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat_v1_adapter.py`

Created a new adapter class `NoRepeatNGramAdaptor` that:
- Extends `vllm.v1.sample.logits_processor.AdapterLogitsProcessor`
- Wraps the existing `NoRepeatNGramLogitsProcessor`
- Creates per-request processor instances from sampling parameters
- Implements required interface methods:
  - `is_argmax_invariant()`: Returns `True` since n-gram filtering affects argmax
  - `new_req_logits_processor(params)`: Creates processor from `extra_args`

### 2. Updated Core Model for v1 Compatibility

**File:** `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py`

#### a. Fallback Import for SamplingMetadata (Line ~13-16)

```python
try:
    from vllm.model_executor import SamplingMetadata
except ImportError:
    from vllm.v1.sample.metadata import SamplingMetadata
```

This ensures compatibility with both v0 and v1 engines.

#### b. Updated `_call_hf_processor` Method (Line ~156)

Added `**kwargs` parameter to accept tokenizer arguments passed by v1 engine:

```python
def _call_hf_processor(
    self,
    prompt: str,
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
    **kwargs,  # tokenizer kwargs in v1
) -> BatchFeature:
```

#### c. Updated `_cached_apply_hf_processor` Method (Line ~235-257)

Added `**kwargs` parameter and forwarded it to underlying methods:

```python
def _cached_apply_hf_processor(
    self,
    prompt: Union[str, list[int]],
    mm_data_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    **kwargs  # forward to underlying processor
) -> tuple[list[int], MultiModalKwargs, bool]:
    if mm_data_items.get_count("image", strict=False) > 2:
        return self._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            enable_hf_prompt_update=True,
            **kwargs
        )
    return super()._cached_apply_hf_processor(
        prompt=prompt,
        mm_data_items=mm_data_items,
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        **kwargs
    )
```

### 3. Updated Inference Script for v1 Engine

**File:** `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`

#### a. Enable v1 Engine (Line ~9)

```python
# Enable vLLM v1 engine for compatibility with vLLM 0.11.0+
os.environ['VLLM_USE_V1'] = '1'
```

#### b. Register v1 Logits Processor in Engine Args (Line ~150-163)

```python
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    max_model_len=8192,
    enforce_eager=False,
    trust_remote_code=True,  
    tensor_parallel_size=1,
    gpu_memory_utilization=0.75,
    # Register v1-compatible logits processor adapter
    logits_processors=["process.ngram_norepeat_v1_adapter:NoRepeatNGramAdaptor"],
)
```

#### c. Process Image AFTER Engine Initialization (Line ~165-171)

**CRITICAL CHANGE:** The processor must not be invoked before the engine is fully initialized.

```python
engine = AsyncLLMEngine.from_engine_args(engine_args)

# CRITICAL: Process image AFTER engine initialization for v1 compatibility
if image is not None and '<image>' in prompt:
    image_features = DeepseekOCRProcessor().tokenize_with_images(
        images=[image], bos=True, eos=True, cropping=CROP_MODE
    )
else:
    image_features = None
```

#### d. Pass N-gram Parameters via `extra_args` (Line ~173-183)

```python
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    skip_special_tokens=False,
    # N-gram parameters for v1 logits processor
    extra_args={
        "ngram_size": 30,
        "window_size": 90,
        "whitelist_token_ids": {128821, 128822}  # whitelist: <td>, </td>
    }
)
```

#### e. Updated Main Execution (Line ~221-233)

```python
image = load_image(INPUT_PATH).convert('RGB')

# IMPORTANT: For v1 engine, process image AFTER engine initialization
# The processor will be called inside stream_generate after engine is ready
# We pass the raw image instead of pre-processed features

prompt = PROMPT

# Pass raw image to stream_generate, which will process it after engine init
if '<image>' in PROMPT:
    result_out = asyncio.run(stream_generate(image, prompt))
else:
    result_out = asyncio.run(stream_generate(None, prompt))
```

## Installation Requirements

```bash
pip install vllm==0.11.0
pip install PyMuPDF img2pdf einops easydict addict Pillow
pip install flash_attn==2.8.1 --no-build-isolation
```

## Key Differences: v0 vs v1 Engine

| Aspect | v0 Engine | v1 Engine |
|--------|-----------|-----------|
| Logits Processors | Per-request via `SamplingParams.logits_processors` | Global via `AsyncEngineArgs.logits_processors` |
| Processor Parameters | Direct instantiation | Via `SamplingParams.extra_args` |
| Processor Interface | `LogitsProcessor` | `AdapterLogitsProcessor` |
| Initialization Order | Flexible | Strict - engine must be ready first |
| Import Path | `vllm.model_executor.SamplingMetadata` | `vllm.v1.sample.metadata.SamplingMetadata` |

## Testing

To verify the changes work correctly:

1. Ensure vLLM 0.11.0 is installed
2. Set appropriate paths in `config.py`:
   - `MODEL_PATH`: Path to DeepSeek-OCR model
   - `INPUT_PATH`: Path to test image
   - `OUTPUT_PATH`: Path for output files
3. Run the inference script:
   ```bash
   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
   python run_dpsk_ocr_image.py
   ```

## Backward Compatibility

The changes maintain backward compatibility where possible:
- The `SamplingMetadata` import uses try/except for fallback
- The `**kwargs` additions are non-breaking for existing code
- The v1 engine can be disabled by setting `os.environ['VLLM_USE_V1'] = '0'`

## Troubleshooting

### Issue: "Module 'vllm.v1.sample.logits_processor' not found"
**Solution:** Ensure vLLM 0.11.0 or later is installed.

### Issue: "KeyError: 'ngram_size' in extra_args"
**Solution:** Verify that `extra_args` is properly set in `SamplingParams` with all required keys.

### Issue: Engine initialization hangs
**Solution:** Ensure the processor is not called before engine initialization. Check that image processing happens AFTER `AsyncLLMEngine.from_engine_args()`.

### Issue: "Per-request logits processors not supported"
**Solution:** Ensure `VLLM_USE_V1='1'` is set and logits processors are registered in `AsyncEngineArgs`, not `SamplingParams`.

## References

- [vLLM v1 Engine Documentation](https://docs.vllm.ai/)
- [DeepSeek-OCR Repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- [GitHub Issue #231](https://github.com/deepseek-ai/DeepSeek-OCR/issues/231)

## Summary

These changes enable DeepSeek-OCR to work seamlessly with vLLM 0.11.0's v1 engine while maintaining code quality and backward compatibility. The key insight is that v1 requires global logits processors with per-request instantiation, and strict initialization ordering must be followed.
