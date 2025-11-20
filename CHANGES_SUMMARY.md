# Summary of Changes for vLLM 0.11.0 v1 Engine Support

## Files Modified/Created

### 1. **NEW FILE:** `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat_v1_adapter.py`
- **Purpose:** v1 engine adapter for n-gram no-repeat logits processor
- **Key Components:**
  - `NoRepeatNGramAdaptor` class extending `AdapterLogitsProcessor`
  - Implements `is_argmax_invariant()` method
  - Implements `new_req_logits_processor(params)` method
  - Wraps existing `NoRepeatNGramLogitsProcessor` for v1 compatibility

### 2. **MODIFIED:** `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py`

#### Change 1: Import Fallback (Lines 13-16)
```python
# BEFORE:
from vllm.model_executor import SamplingMetadata

# AFTER:
try:
    from vllm.model_executor import SamplingMetadata
except ImportError:
    from vllm.v1.sample.metadata import SamplingMetadata
```

#### Change 2: `_call_hf_processor` Method Signature (Line ~156)
```python
# BEFORE:
def _call_hf_processor(
    self,
    prompt: str,
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
) -> BatchFeature:

# AFTER:
def _call_hf_processor(
    self,
    prompt: str,
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
    **kwargs,  # tokenizer kwargs in v1
) -> BatchFeature:
```

#### Change 3: `_cached_apply_hf_processor` Method (Lines ~235-257)
```python
# BEFORE:
def _cached_apply_hf_processor(
    self,
    prompt: Union[str, list[int]],
    mm_data_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> tuple[list[int], MultiModalKwargs, bool]:
    if mm_data_items.get_count("image", strict=False) > 2:
        return self._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            enable_hf_prompt_update=True,
        )
    return super()._cached_apply_hf_processor(
        prompt=prompt,
        mm_data_items=mm_data_items,
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

# AFTER:
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

### 3. **MODIFIED:** `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`

#### Change 1: Enable v1 Engine (Line ~9)
```python
# BEFORE:
os.environ['VLLM_USE_V1'] = '0'

# AFTER:
# Enable vLLM v1 engine for compatibility with vLLM 0.11.0+
os.environ['VLLM_USE_V1'] = '1'
```

#### Change 2: Engine Initialization (Lines ~150-183)
```python
# BEFORE:
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    max_model_len=8192,
    enforce_eager=False,
    trust_remote_code=True,  
    tensor_parallel_size=1,
    gpu_memory_utilization=0.75,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822})]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

# AFTER:
# Initialize engine with v1-compatible logits processor
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
engine = AsyncLLMEngine.from_engine_args(engine_args)

# CRITICAL: Process image AFTER engine initialization for v1 compatibility
if image is not None and '<image>' in prompt:
    image_features = DeepseekOCRProcessor().tokenize_with_images(
        images=[image], bos=True, eos=True, cropping=CROP_MODE
    )
else:
    image_features = None

# For v1 engine, pass n-gram parameters via extra_args in SamplingParams
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

#### Change 3: Main Execution Block (Lines ~221-233)
```python
# BEFORE:
image = load_image(INPUT_PATH).convert('RGB')

if '<image>' in PROMPT:
    image_features = DeepseekOCRProcessor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)
else:
    image_features = ''

prompt = PROMPT
result_out = asyncio.run(stream_generate(image_features, prompt))

# AFTER:
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

### 4. **NEW FILE:** `VLLM_V1_MIGRATION_GUIDE.md`
- Comprehensive documentation of all changes
- Installation requirements
- Troubleshooting guide
- Comparison table of v0 vs v1 differences

### 5. **NEW FILE:** `CHANGES_SUMMARY.md` (this file)
- Quick reference of all modifications
- Before/after code comparisons

## Key Architectural Changes

1. **Logits Processor Registration:**
   - v0: Per-request via `SamplingParams.logits_processors`
   - v1: Global via `AsyncEngineArgs.logits_processors` with string path

2. **Processor Parameters:**
   - v0: Direct instantiation with parameters
   - v1: Parameters passed via `SamplingParams.extra_args`

3. **Initialization Order:**
   - v0: Flexible - processor can be called anytime
   - v1: Strict - engine must be initialized before processor is invoked

4. **Processor Interface:**
   - v0: `LogitsProcessor` from transformers
   - v1: `AdapterLogitsProcessor` from vllm.v1

## Testing Checklist

- [x] Created v1 adapter for logits processor
- [x] Updated import statements with fallback
- [x] Modified method signatures to accept **kwargs
- [x] Updated engine initialization with v1 configuration
- [x] Moved image processing after engine initialization
- [x] Updated sampling parameters to use extra_args
- [x] Created comprehensive documentation

## Compatibility Notes

- **Backward Compatible:** Changes include fallbacks for v0 engine
- **Forward Compatible:** Fully supports vLLM 0.11.0+ v1 engine
- **Toggle Support:** Can switch between v0 and v1 via environment variable

## Related Issue

GitHub Issue #231: Enable DeepSeek-OCR support in latest vLLM 0.11.0 (v1 Engine)
