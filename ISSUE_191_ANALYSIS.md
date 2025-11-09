# GitHub Issue #191: Consistently Hallucinating During Free OCR

## Problem Analysis

### Root Cause
The hallucination issue occurs because the **Transformers/HuggingFace implementation is missing the `NoRepeatNGramLogitsProcessor`** that is present in the vLLM implementation. This processor is critical for preventing the model from generating repetitive n-grams, which is a common cause of hallucination in OCR tasks.

### Key Differences Between Implementations

#### vLLM Implementation (No Hallucination)
```python
logits_processors = [NoRepeatNGramLogitsProcessor(
    ngram_size=30, 
    window_size=90, 
    whitelist_token_ids={128821, 128822}  # <td>, </td>
)]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)
```

#### Transformers Implementation (Has Hallucination)
```python
# Missing logits processor!
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, ...)
```

### Why This Causes Hallucination

1. **Repetitive Patterns**: Without n-gram blocking, the model can get stuck in repetitive loops, especially with:
   - Ancient or handwritten documents
   - Low-quality images
   - Non-standard fonts or languages

2. **Temperature=0.0**: While greedy decoding (temperature=0) should be deterministic, without n-gram prevention, it can still produce repetitive outputs when the model is uncertain.

3. **Handwritten Portuguese**: Ancient handwritten documents are particularly challenging because:
   - Character ambiguity is high
   - The model may lack sufficient training data for this specific domain
   - Without repetition prevention, it falls into safe but incorrect patterns

## Solution

The solution involves:

1. **Port the `NoRepeatNGramLogitsProcessor`** from vLLM to work with Transformers
2. **Modify the inference code** to use this processor
3. **Add proper generation configuration** with appropriate parameters
4. **Provide best practices** for handling difficult documents

## Additional Recommendations

### For Ancient/Handwritten Documents:
- Use **larger resolution modes** (Base: 1024×1024 or Large: 1280×1280)
- Try **different prompts**:
  - `<image>\n<|grounding|>OCR this image.` (better for non-standard documents)
  - `<image>\n<|grounding|>Convert the document to markdown.` (better for structured documents)
- Enable **crop_mode=True** for better handling of large documents
- Adjust **ngram_size** (try 20-40) and **window_size** (try 60-120) based on document characteristics

### Image Preprocessing:
- Ensure proper image quality (contrast, brightness)
- Consider image enhancement techniques for ancient documents
- Use appropriate resolution (not too small, not unnecessarily large)
