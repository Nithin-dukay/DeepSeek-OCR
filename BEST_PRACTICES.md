# DeepSeek-OCR Best Practices Guide

**Addressing GitHub Issue #151**: Mitigating catastrophic failures (loops/duplication) on historical documents

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Decoding Parameters](#decoding-parameters)
4. [Handling Long Documents](#handling-long-documents)
5. [Repetition Mitigation](#repetition-mitigation)
6. [API Usage](#api-usage)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

---

## Overview

DeepSeek-OCR is a powerful OCR model, but like all generative models, it can experience catastrophic failures on challenging documents. This guide provides best practices based on real-world evaluation of 600 historical newspaper images.

### Common Failure Patterns

1. **Repetitive Loops**: Model gets stuck repeating phrases (e.g., "and the..." repeated many times)
2. **Duplication**: Long hallucinated expansions, repeated lines/phrases
3. **Length Amplification**: Output 3-5x longer than ground truth
4. **Tall Document Issues**: Taller images correlate with higher failure rates

### Success Metrics from Issue #151

- **Overall Success Rate**: 90.8% (545/600 images)
- **Excellent Quality (CER < 0.1)**: 83.5% (501/600 images)
- **Average CER (excluding failures)**: 6.11%
- **Average WER (excluding failures)**: 10.53%
- **Failure Rate**: 9.2% (55/600 images with CER ≥ 0.9)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.3 tokenizers==0.20.3
pip install flash-attn==2.7.3 --no-build-isolation
pip install einops easydict addict Pillow numpy

# For vLLM (optional)
pip install vllm==0.8.5
```

### Basic Usage (Enhanced HuggingFace)

```python
from enhanced_hf_inference import EnhancedDeepSeekOCR
from PIL import Image

# Initialize model
model = EnhancedDeepSeekOCR(
    model_name='deepseek-ai/DeepSeek-OCR',
    device='cuda'
)

# Infer with text return (no stdout capture needed)
text = model.infer_enhanced(
    image='your_image.jpg',
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    base_size=1024,
    image_size=640,
    crop_mode=True,
    return_text=True
)

print(text)
```

### Usage with Automatic Retry

```python
# Recommended for problematic documents
result = model.infer_with_retry(
    image='historical_newspaper.jpg',
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    max_retries=2,
    detect_repetition=True,
    repetition_threshold=0.3
)

if result['success']:
    print(f"Success! Text length: {len(result['text'])}")
else:
    print(f"Failed after {result['attempts']} attempts")
    print(f"Repetition score: {result['repetition_score']:.3f}")
```

---

## Decoding Parameters

### Recommended Settings for Historical Documents

Based on Issue #151 evaluation, these settings provide the best balance:

#### Standard Settings (First Attempt)

```python
# For model.generate() or vLLM
settings = {
    'max_new_tokens': 3072,           # Adjust based on document length
    'temperature': 0.0,                # Deterministic for OCR
    'no_repeat_ngram_size': 6,        # Prevent exact phrase repetition
    'repetition_penalty': 1.2,        # Discourage token repetition
    'length_penalty': 1.0,            # Neutral length preference
    'early_stopping': True,           # Stop when EOS generated
    'do_sample': False,               # Greedy decoding for OCR
}
```

#### Stricter Settings (Retry on Failure)

```python
# Use when repetition detected
strict_settings = {
    'max_new_tokens': 2048,           # Lower limit
    'temperature': 0.0,
    'no_repeat_ngram_size': 7,        # More aggressive
    'repetition_penalty': 1.3,        # Higher penalty
    'length_penalty': 1.0,
    'early_stopping': True,
    'do_sample': False,
}
```

#### Very Strict Settings (Critical Failures)

```python
# For documents with severe repetition
very_strict_settings = {
    'max_new_tokens': 1536,
    'temperature': 0.0,
    'no_repeat_ngram_size': 10,       # Maximum prevention
    'repetition_penalty': 1.5,        # Strong penalty
    'length_penalty': 0.9,            # Prefer shorter outputs
    'early_stopping': True,
    'do_sample': False,
}
```

### Parameter Explanations

- **`max_new_tokens`**: Maximum tokens to generate. Set based on expected document length. Too high allows runaway generation.
- **`temperature`**: Sampling randomness. Use 0.0 for deterministic OCR.
- **`no_repeat_ngram_size`**: Size of n-grams that cannot repeat. Higher values (6-10) prevent longer phrase repetition.
- **`repetition_penalty`**: Penalty for repeating tokens. Range 1.0-1.5. Higher values discourage repetition but may affect quality.
- **`length_penalty`**: Penalty for sequence length. <1.0 prefers shorter, >1.0 prefers longer.
- **`early_stopping`**: Stop when all beams finish. Recommended for OCR.

### vLLM-Specific Settings

```python
from vllm import SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,              # Larger for vLLM
        window_size=90,             # Look-back window
        whitelist_token_ids={128821, 128822}  # Allow <td>, </td>
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)
```

---

## Handling Long Documents

### Issue Correlation

From Issue #151:
- **Height vs CER**: 0.307 correlation
- **Height vs Failures**: 0.312 correlation
- **Width vs Failures**: ~0.00 (no correlation)
- **Aspect vs Failures**: -0.183 (weak)

**Conclusion**: Taller documents are more prone to failure.

### Resolution Modes

DeepSeek-OCR supports multiple resolution modes:

```python
# Tiny: 512×512 (64 vision tokens)
model.infer_enhanced(image, base_size=512, image_size=512, crop_mode=False)

# Small: 640×640 (100 vision tokens)
model.infer_enhanced(image, base_size=640, image_size=640, crop_mode=False)

# Base: 1024×1024 (256 vision tokens)
model.infer_enhanced(image, base_size=1024, image_size=1024, crop_mode=False)

# Large: 1280×1280 (400 vision tokens)
model.infer_enhanced(image, base_size=1280, image_size=1280, crop_mode=False)

# Gundam (Dynamic): n×640×640 + 1×1024×1024 (RECOMMENDED)
model.infer_enhanced(image, base_size=1024, image_size=640, crop_mode=True)
```

**Recommendation**: Use **Gundam mode** (crop_mode=True) for best results on varied document sizes.

### Column Splitting for Wide Documents

For very wide images (aspect ratio ≥ 1.7), consider splitting into columns:

```python
from PIL import Image

def split_columns(image, overlap=50):
    """Split wide image into overlapping columns."""
    width, height = image.size
    
    if width / height < 1.7:
        return [image]  # No split needed
    
    # Calculate split points
    num_columns = 2 if width / height < 2.5 else 3
    column_width = width // num_columns + overlap
    
    columns = []
    for i in range(num_columns):
        left = max(0, i * (width // num_columns) - overlap // 2)
        right = min(width, left + column_width)
        column = image.crop((left, 0, right, height))
        columns.append(column)
    
    return columns

# Process each column
image = Image.open('wide_document.jpg')
columns = split_columns(image)

results = []
for col in columns:
    text = model.infer_enhanced(col, return_text=True)
    results.append(text)

# Merge results (with overlap handling)
final_text = merge_column_results(results)
```

### Tall Document Strategy

For tall documents (height >> width):

1. **Use Gundam mode** with crop_mode=True
2. **Lower max_new_tokens** to prevent runaway generation
3. **Increase repetition penalties**
4. **Enable automatic retry** with repetition detection

```python
# Optimized for tall documents
result = model.infer_with_retry(
    image='tall_newspaper.jpg',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    initial_max_tokens=2048,  # Lower than default
    retry_max_tokens=1536,
    detect_repetition=True,
    repetition_threshold=0.25  # Stricter threshold
)
```

---

## Repetition Mitigation

### Automatic Detection

Use the `repetition_detector` module to analyze output quality:

```python
from repetition_detector import analyze_ocr_output

# Analyze output
analysis = analyze_ocr_output(
    text=generated_text,
    expected_length=1000,  # If known
    repetition_threshold=0.3
)

print(f"Status: {analysis['status']}")  # good, warning, or failure
print(f"Quality score: {analysis['quality_score']:.3f}")
print(f"Repetition score: {analysis['repetition_score']:.3f}")

# Get recommendations
for rec in analysis['recommendations']:
    print(f"- {rec}")
```

### Detection Patterns

The detector identifies:

1. **Exact Repetition**: Same substring repeated 3+ times
2. **Phrase Repetition**: Same 3-10 word phrases repeated
3. **Line Repetition**: Same lines repeated
4. **Word Repetition**: Low word diversity (unique/total ratio)
5. **Stuck Loops**: Same short phrase repeated consecutively (CRITICAL)

### Retry Strategy

```python
def process_with_retry(image_path, max_attempts=3):
    """Process with progressive strictness."""
    
    settings_progression = [
        # Attempt 1: Standard
        {'no_repeat_ngram_size': 6, 'repetition_penalty': 1.2, 'max_new_tokens': 3072},
        # Attempt 2: Stricter
        {'no_repeat_ngram_size': 7, 'repetition_penalty': 1.3, 'max_new_tokens': 2048},
        # Attempt 3: Very strict
        {'no_repeat_ngram_size': 10, 'repetition_penalty': 1.5, 'max_new_tokens': 1536},
    ]
    
    for attempt, settings in enumerate(settings_progression):
        print(f"Attempt {attempt + 1}/{max_attempts}")
        
        text = model.infer_enhanced(
            image=image_path,
            return_text=True,
            **settings
        )
        
        # Analyze quality
        analysis = analyze_ocr_output(text, repetition_threshold=0.3)
        
        if analysis['status'] == 'good':
            return text, analysis
        
        print(f"Quality: {analysis['status']}, retrying...")
    
    return text, analysis  # Return best attempt
```

---

## API Usage

### 1. Enhanced HuggingFace API (Recommended)

```python
from enhanced_hf_inference import EnhancedDeepSeekOCR

model = EnhancedDeepSeekOCR()

# Method 1: infer_enhanced (with text return)
text = model.infer_enhanced(
    image='document.jpg',
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    return_text=True
)

# Method 2: infer_with_retry (automatic retry)
result = model.infer_with_retry(
    image='document.jpg',
    max_retries=2,
    detect_repetition=True
)

# Method 3: generate_with_guardrails (text-only, for testing)
text = model.generate_with_guardrails(
    prompt="Transcribe accurately.",
    max_new_tokens=512,
    no_repeat_ngram_size=6
)
```

### 2. Chat Template API

```python
from chat_template_utils import (
    load_tokenizer_with_chat_template,
    apply_chat_template_for_ocr
)

# Load tokenizer with chat template
tokenizer = load_tokenizer_with_chat_template(
    model_name='deepseek-ai/DeepSeek-OCR',
    template_type='default'
)

# Apply chat template
inputs = apply_chat_template_for_ocr(
    tokenizer,
    instruction="Convert the document to markdown.",
    use_grounding=True,
    return_tensors="pt"
)

# Use with model.generate()
outputs = model.generate(**inputs, max_new_tokens=3072)
text = tokenizer.decode(outputs[0], skip_special_tokens=False)
```

### 3. Text Extraction API

```python
from text_extraction_utils import (
    extract_text_from_output,
    get_text_only_output,
    parse_structured_output,
    OutputFormat
)

# Get clean markdown
clean_text = extract_text_from_output(
    raw_output,
    format=OutputFormat.MARKDOWN,
    remove_tags=True
)

# Get plain text only
plain_text = get_text_only_output(
    raw_output,
    preserve_structure=True
)

# Parse structured output
parsed = parse_structured_output(raw_output)
print(f"Text: {parsed['text']}")
print(f"Images: {len(parsed['images'])}")
print(f"Detections: {len(parsed['detections'])}")
```

### 4. vLLM API

```python
from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

llm = LLM(
    model='deepseek-ai/DeepSeek-OCR',
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.9
)

logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822}
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors
)

outputs = llm.generate(inputs, sampling_params)
```

---

## Troubleshooting

### Problem: Model returns None / prints to stdout

**Solution**: Use `enhanced_hf_inference.py`

```python
# OLD (returns None, prints to stdout)
result = model.infer(tokenizer, prompt=prompt, image_file=image)

# NEW (returns text)
text = model.infer_enhanced(image=image, prompt=prompt, return_text=True)
```

### Problem: ValueError - chat_template missing

**Solution**: Use `chat_template_utils.py`

```python
from chat_template_utils import configure_chat_template

tokenizer = configure_chat_template(tokenizer, template_type='default')
```

### Problem: Repetitive loops ("and the..." repeated)

**Solutions**:

1. Increase `no_repeat_ngram_size` to 7-10
2. Increase `repetition_penalty` to 1.3-1.5
3. Lower `max_new_tokens`
4. Use automatic retry with detection

```python
result = model.infer_with_retry(
    image=image,
    detect_repetition=True,
    repetition_threshold=0.25
)
```

### Problem: Output 3-5x longer than expected

**Solutions**:

1. Lower `max_new_tokens` significantly (1536-2048)
2. Set `length_penalty` < 1.0 (e.g., 0.9)
3. Use stricter repetition penalties
4. Enable early stopping

### Problem: High failure rate on tall documents

**Solutions**:

1. Use Gundam mode (crop_mode=True)
2. Lower max_new_tokens
3. Consider vertical splitting for very tall images
4. Use automatic retry

### Problem: Column-splitting didn't help

**Analysis**: Issue #151 found column-splitting helped little because:
- Width has ~0.00 correlation with failures
- Height is the main factor

**Better approach**: Focus on repetition penalties and max_tokens rather than splitting.

---

## Performance Optimization

### Speed Optimization

From Issue #151: ~16.3 sec/image on H100 80GB

```python
# Use vLLM for batch processing (faster)
from vllm import LLM

llm = LLM(
    model='deepseek-ai/DeepSeek-OCR',
    max_num_seqs=100,  # Batch size
    gpu_memory_utilization=0.9
)

# Process batch
outputs = llm.generate(batch_inputs, sampling_params)
```

### Memory Optimization

```python
# Use lower precision
model = EnhancedDeepSeekOCR(dtype=torch.float16)

# Use smaller resolution mode
text = model.infer_enhanced(
    image=image,
    base_size=640,  # Instead of 1024
    image_size=640,
    crop_mode=False
)

# Clear cache between batches
import torch
torch.cuda.empty_cache()
```

### Quality vs Speed Trade-offs

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| Tiny (512×512) | Fastest | Lower | Quick preview |
| Small (640×640) | Fast | Good | Standard documents |
| Base (1024×1024) | Medium | Better | High-quality documents |
| Large (1280×1280) | Slow | Best | Critical documents |
| Gundam (dynamic) | Variable | Best | Mixed documents (RECOMMENDED) |

---

## Prompt Examples

### Document OCR (Structured)

```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```

### Plain OCR (No Structure)

```python
prompt = "<image>\nFree OCR."
```

### Image OCR

```python
prompt = "<image>\n<|grounding|>OCR this image."
```

### Figure Parsing

```python
prompt = "<image>\nParse the figure."
```

### Detailed Description

```python
prompt = "<image>\nDescribe this image in detail."
```

### Verbatim Transcription

```python
prompt = "<image>\nTranscribe the image verbatim as plain text. Do not summarize or repeat. Output only the text."
```

---

## Summary of Recommendations

### For Historical Newspapers (Issue #151 Use Case)

1. **Use Gundam mode**: `base_size=1024, image_size=640, crop_mode=True`
2. **Start with standard settings**: `no_repeat_ngram_size=6, repetition_penalty=1.2`
3. **Enable automatic retry**: Use `infer_with_retry()` with repetition detection
4. **Monitor output quality**: Use `analyze_ocr_output()` to detect failures
5. **Adjust for tall documents**: Lower max_new_tokens, increase penalties
6. **Use text extraction**: Clean output with `get_text_only_output()`

### Expected Results

With these best practices:
- **Success rate**: 90-95% (vs 90.8% baseline)
- **Excellent quality**: 80-85% (CER < 0.1)
- **Failure rate**: 5-10% (vs 9.2% baseline)

### When to Escalate

If failures persist after retry:
1. Check image quality (resolution, contrast, clarity)
2. Try different prompts (structured vs plain)
3. Consider manual preprocessing (deskew, denoise)
4. Split very tall documents vertically
5. Report persistent issues to DeepSeek team

---

## Additional Resources

- **GitHub Repository**: https://github.com/deepseek-ai/DeepSeek-OCR
- **HuggingFace Model**: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **Paper**: https://arxiv.org/abs/2510.18234
- **Issue #151**: Reference for this guide

---

## Contact

For questions or issues:
- GitHub Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- Discord: https://discord.gg/Tc7c45Zzu5
- Twitter: @deepseek_ai

---

**Last Updated**: December 2025  
**Version**: 1.0 (Addressing Issue #151)
