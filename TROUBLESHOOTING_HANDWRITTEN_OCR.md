# Troubleshooting Guide: Hallucinations in Handwritten OCR

## Issue Overview

When using DeepSeek-OCR on handwritten documents, especially ancient or historical texts, you may experience "hallucinations" where the model generates text that doesn't exist in the image. This guide explains why this happens and how to fix it.

## Understanding the Problem

### What Causes Hallucinations?

1. **Incorrect Prompt Selection**: Using "Free OCR" mode without layout grounding
2. **Suboptimal Parameters**: Default sampling parameters may be too creative for precise OCR
3. **Image Quality**: Low resolution or poor preprocessing of historical documents
4. **Model Uncertainty**: Complex handwriting patterns can confuse the model

### Prompt Types and When to Use Them

DeepSeek-OCR supports multiple prompt modes:

#### 1. **Free OCR Mode** (Basic)
```python
prompt = "<|image|>Extract all text from the image."
```
- **Use for**: Simple, clean documents with minimal layout
- **Avoid for**: Complex layouts, handwritten documents, historical texts
- **Risk**: High hallucination risk on complex documents

#### 2. **Grounded OCR Mode** (Recommended for Handwritten)
```python
prompt = "<|image|><|grounding|>Extract all text with precise positioning."
```
- **Use for**: Handwritten documents, complex layouts, historical texts
- **Benefits**: Layout awareness reduces hallucinations
- **Best for**: Ancient Portuguese documents, manuscripts

#### 3. **Structured OCR Mode** (For Tables/Forms)
```python
prompt = "<|image|><|grounding|>Extract text preserving table structure."
```
- **Use for**: Forms, tables, structured documents
- **Benefits**: Maintains spatial relationships

#### 4. **Selective OCR Mode** (For Specific Regions)
```python
prompt = "<|image|>Extract text from the highlighted region: <|box_start|>(x1,y1),(x2,y2)<|box_end|>"
```
- **Use for**: Focusing on specific document areas
- **Benefits**: Reduces context confusion

## Solutions for Handwritten Documents

### Solution 1: Use Grounded OCR with Conservative Parameters

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load model
model_name = "deepseek-ai/deepseek-ocr"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load image
image = Image.open("ancient_portuguese_document.jpg")

# Use grounded prompt for better accuracy
prompt = "<|image|><|grounding|>Extract all text from this handwritten document with precise positioning."

# Conservative generation parameters
inputs = tokenizer(prompt, return_tensors="pt", images=image).to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.1,        # Lower temperature = less creative/more deterministic
    top_p=0.9,              # Nucleus sampling for quality
    do_sample=True,         # Enable sampling for better results
    repetition_penalty=1.1, # Prevent repetitive hallucinations
    num_beams=1             # Greedy decoding for consistency
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Solution 2: Optimize Image Preprocessing

```python
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_handwritten_image(image_path, output_path=None):
    """
    Optimize image for handwritten OCR
    """
    img = Image.open(image_path)
    
    # 1. Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 2. Resize if too large (max 2048px on longest side)
    max_size = 2048
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # 3. Enhance contrast for faded historical documents
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # 4. Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    
    # 5. Optional: Denoise (careful with historical documents)
    # img = img.filter(ImageFilter.MedianFilter(size=3))
    
    if output_path:
        img.save(output_path, quality=95)
    
    return img

# Use preprocessed image
preprocessed_img = preprocess_handwritten_image("ancient_document.jpg")
```

### Solution 3: Multi-Pass OCR with Validation

```python
def multi_pass_ocr(image_path, model, tokenizer):
    """
    Run OCR multiple times with different parameters and validate results
    """
    image = Image.open(image_path)
    results = []
    
    # Pass 1: Grounded OCR with low temperature
    prompt1 = "<|image|><|grounding|>Extract all text from this handwritten document."
    inputs = tokenizer(prompt1, return_tensors="pt", images=image).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True)
    result1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append(result1)
    
    # Pass 2: Grounded OCR with slightly higher temperature
    outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.3, do_sample=True)
    result2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append(result2)
    
    # Pass 3: Free OCR for comparison
    prompt2 = "<|image|>Extract all text from this image."
    inputs = tokenizer(prompt2, return_tensors="pt", images=image).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True)
    result3 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append(result3)
    
    return results

# Compare results and choose the most consistent one
results = multi_pass_ocr("document.jpg", model, tokenizer)
for i, result in enumerate(results):
    print(f"\n=== Pass {i+1} ===")
    print(result)
```

### Solution 4: Region-Based OCR for Complex Documents

```python
def region_based_ocr(image_path, regions, model, tokenizer):
    """
    Process document in regions to reduce hallucinations
    
    Args:
        regions: List of (x1, y1, x2, y2) tuples in normalized coordinates [0-1]
    """
    image = Image.open(image_path)
    width, height = image.size
    
    results = []
    for i, (x1, y1, x2, y2) in enumerate(regions):
        # Crop region
        box = (int(x1*width), int(y1*height), int(x2*width), int(y2*height))
        region_img = image.crop(box)
        
        # OCR on region
        prompt = f"<|image|><|grounding|>Extract all text from this region of a handwritten document."
        inputs = tokenizer(prompt, return_tensors="pt", images=region_img).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            'region': i+1,
            'coordinates': (x1, y1, x2, y2),
            'text': result
        })
    
    return results

# Example: Process document in 4 quadrants
regions = [
    (0.0, 0.0, 0.5, 0.5),  # Top-left
    (0.5, 0.0, 1.0, 0.5),  # Top-right
    (0.0, 0.5, 0.5, 1.0),  # Bottom-left
    (0.5, 0.5, 1.0, 1.0),  # Bottom-right
]
results = region_based_ocr("document.jpg", regions, model, tokenizer)
```

## Parameter Tuning Guide

### Temperature
- **0.0-0.2**: Most deterministic, best for precise OCR (recommended for handwritten)
- **0.3-0.5**: Balanced, good for general documents
- **0.6-1.0**: More creative, avoid for OCR tasks

### Top-p (Nucleus Sampling)
- **0.9**: Recommended for most OCR tasks
- **0.95**: Slightly more diverse outputs
- **1.0**: Full vocabulary (may increase hallucinations)

### Repetition Penalty
- **1.0**: No penalty
- **1.1-1.2**: Recommended to prevent repetitive hallucinations
- **>1.3**: May affect legitimate repeated words

### Max New Tokens
- **512**: Short documents
- **1024**: Medium documents
- **2048**: Long documents (default)
- **4096**: Very long documents (use with caution)

## Best Practices for Ancient/Historical Documents

1. **Always use grounded OCR mode** (`<|grounding|>`)
2. **Set temperature to 0.1-0.2** for maximum accuracy
3. **Preprocess images** to enhance contrast and sharpness
4. **Use high-resolution scans** (300 DPI minimum)
5. **Process in regions** if document is very large or complex
6. **Run multiple passes** and compare results
7. **Validate output** against known text samples if available

## Language-Specific Considerations

### Portuguese (Ancient/Historical)
```python
prompt = "<|image|><|grounding|>Extract all Portuguese text from this historical handwritten document, preserving original spelling and diacritics."
```

### Multi-language Documents
```python
prompt = "<|image|><|grounding|>Extract all text from this document. The text may contain multiple languages including Portuguese and Latin."
```

## Debugging Checklist

- [ ] Using grounded OCR mode (`<|grounding|>`)
- [ ] Temperature set to 0.1-0.2
- [ ] Image resolution adequate (>1000px on longest side)
- [ ] Image preprocessed (contrast, sharpness)
- [ ] Repetition penalty enabled (1.1-1.2)
- [ ] Max tokens sufficient for document length
- [ ] Tried multiple passes for validation
- [ ] Considered region-based processing

## Common Pitfalls

1. **Using Free OCR on complex documents**: Always use grounded mode
2. **High temperature values**: Keep below 0.3 for OCR
3. **Poor image quality**: Preprocess before OCR
4. **Processing entire large document at once**: Use region-based approach
5. **Not validating results**: Always review and compare multiple runs

## Example: Complete Workflow for Ancient Portuguese Documents

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageEnhance

# 1. Load model
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-ocr",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-ocr", trust_remote_code=True)

# 2. Preprocess image
img = Image.open("ancient_portuguese.jpg")
if img.mode != 'RGB':
    img = img.convert('RGB')

# Enhance for historical documents
contrast = ImageEnhance.Contrast(img)
img = contrast.enhance(1.5)
sharpness = ImageEnhance.Sharpness(img)
img = sharpness.enhance(1.3)

# 3. Run OCR with optimal parameters
prompt = "<|image|><|grounding|>Extract all Portuguese text from this historical handwritten document, preserving original spelling and diacritics."

inputs = tokenizer(prompt, return_tensors="pt", images=img).to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.1,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.15,
    num_beams=1
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Post-process and validate
print("=== OCR Result ===")
print(result)

# 5. Optional: Run second pass for validation
outputs2 = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.15,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.15
)
result2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)

print("\n=== Validation Pass ===")
print(result2)
```

## Getting Help

If you continue to experience hallucinations after following this guide:

1. Share your image characteristics (resolution, quality, complexity)
2. Share your exact prompt and parameters
3. Provide example output showing the hallucination
4. Mention the language and document type
5. Open an issue on GitHub with these details

## Additional Resources

- [DeepSeek-OCR Paper](DeepSeek_OCR_paper.pdf) - Technical details
- [README.md](README.md) - Basic usage examples
- [Examples Directory](examples/) - Practical use cases
- [Configuration Helper](DeepSeek-OCR-master/DeepSeek-OCR-hf/ocr_config_helper.py) - Auto-configuration tool
