# Solution for GitHub Issue #289: 复现deepseek-ocr的deepencoder

## Issue Summary

The user wants to extract and reproduce the DeepEncoder component from DeepSeek-OCR independently - inputting an image and outputting its encoded representation.

## Understanding DeepEncoder

### What is DeepEncoder?

DeepEncoder is the **vision encoding component** of DeepSeek-OCR that converts images into embeddings. It consists of:

1. **SAM-based encoder** (`sam_vary_sdpa.py`) - Extracts low-level spatial features
2. **CLIP-based encoder** (`clip_sdpa.py`) - Extracts high-level semantic features  
3. **MLP Projector** (`build_linear.py`) - Projects combined features to LLM embedding space

### Core Code Location

Yes, the line mentioned in the issue is indeed part of the core encoding process:

```python
vision_embeddings = self.get_multimodal_embeddings(**kwargs)
```

This is located in `deepseek_ocr.py` at line ~580 and calls the complete encoding pipeline.

## DeepEncoder Architecture

The encoding process follows these steps:

```
Input Image 
    ↓
[1] SAM Encoder (sam_model)
    ↓ (produces 1024-dim features)
[2] CLIP Encoder (vision_model) 
    ↓ (produces 1024-dim features)
[3] Feature Concatenation [CLIP + SAM]
    ↓ (2048-dim combined features)
[4] MLP Projector
    ↓ (projects to 1280-dim)
[5] Add Special Tokens (newline, view_separator)
    ↓
Final Vision Embeddings
```

### Key Methods in `deepseek_ocr.py`:

1. **`get_multimodal_embeddings()`** (line ~570)
   - Entry point for vision encoding
   - Parses and validates image inputs
   - Returns vision embeddings

2. **`_process_image_input()`** (line ~540)
   - Processes pixel values and crops
   - Calls the pixel-to-embedding conversion

3. **`_pixel_values_to_embedding()`** (line ~450)
   - **THIS IS THE CORE DEEPENCODER LOGIC**
   - Runs SAM encoder: `self.sam_model(image)`
   - Runs CLIP encoder: `self.vision_model(image, sam_features)`
   - Concatenates features: `torch.cat([clip_features, sam_features], dim=-1)`
   - Projects to embedding space: `self.projector(combined_features)`
   - Adds positional tokens (newlines, separators)

## How to Reproduce DeepEncoder Independently

### Step 1: Install Dependencies

```bash
pip install torch torchvision transformers pillow einops addict
```

### Step 2: Use the Standalone Script

I've created `reproduce_deepencoder.py` which extracts just the DeepEncoder functionality:

```bash
python reproduce_deepencoder.py --image_path your_image.jpg --output_path output.pt
```

### Step 3: Understanding the Output

The script outputs:
- **Vision embeddings tensor**: Shape `[num_tokens, 1280]`
- **Metadata**: Number of tokens, image dimensions, processing mode

### Key Parameters:

- `BASE_SIZE`: Base resolution (512/640/1024/1280)
- `IMAGE_SIZE`: Crop resolution (640/1024)  
- `CROP_MODE`: Whether to use dynamic cropping (True/False)

## Code Explanation

### Core DeepEncoder Implementation:

```python
# 1. SAM Encoder - Extract spatial features
sam_features = self.sam_model(image)  # [B, 1024, H/16, W/16]

# 2. CLIP Encoder - Extract semantic features  
clip_features = self.vision_model(image, sam_features)  # [B, seq_len, 1024]

# 3. Concatenate features
combined = torch.cat([
    clip_features[:, 1:],  # Remove CLS token
    sam_features.flatten(2).permute(0, 2, 1)  # Flatten spatial dims
], dim=-1)  # [B, seq_len, 2048]

# 4. Project to embedding space
embeddings = self.projector(combined)  # [B, seq_len, 1280]

# 5. Add special tokens for layout
# - image_newline: Marks end of each row
# - view_separator: Marks end of view (global/local)
```

### Different Processing Modes:

**Tiny Mode** (512×512, 64 tokens):
```python
BASE_SIZE = 512
IMAGE_SIZE = 512  
CROP_MODE = False
```

**Small Mode** (640×640, 100 tokens):
```python
BASE_SIZE = 640
IMAGE_SIZE = 640
CROP_MODE = False
```

**Base Mode** (1024×1024, 256 tokens):
```python
BASE_SIZE = 1024
IMAGE_SIZE = 1024
CROP_MODE = False
```

**Gundam Mode** (Dynamic, variable tokens):
```python
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True  # Uses dynamic cropping
```

## Model Weights

The DeepEncoder weights are part of the full DeepSeek-OCR model:

```python
model_path = "deepseek-ai/DeepSeek-OCR"
```

The weights are loaded from:
- `sam_model.*` - SAM encoder weights
- `vision_model.*` - CLIP encoder weights  
- `projector.*` - MLP projector weights
- `image_newline` - Newline token embedding
- `view_seperator` - View separator token embedding

## Usage Examples

### Example 1: Single Image Encoding

```python
from reproduce_deepencoder import DeepEncoderStandalone
from PIL import Image

# Initialize encoder
encoder = DeepEncoderStandalone(
    model_path="deepseek-ai/DeepSeek-OCR",
    base_size=1024,
    image_size=640,
    crop_mode=True
)

# Load and encode image
image = Image.open("document.jpg")
embeddings = encoder.encode_image(image)

print(f"Embedding shape: {embeddings.shape}")
# Output: Embedding shape: torch.Size([num_tokens, 1280])
```

### Example 2: Batch Processing

```python
images = [Image.open(f"image_{i}.jpg") for i in range(5)]
embeddings_list = encoder.encode_batch(images)

for i, emb in enumerate(embeddings_list):
    print(f"Image {i}: {emb.shape[0]} tokens")
```

### Example 3: Extract Features at Different Stages

```python
# Get intermediate features
sam_features = encoder.sam_model(image_tensor)
clip_features = encoder.vision_model(image_tensor, sam_features)
combined_features = torch.cat([clip_features[:, 1:], 
                               sam_features.flatten(2).permute(0, 2, 1)], 
                              dim=-1)
final_embeddings = encoder.projector(combined_features)
```

## Key Insights

1. **DeepEncoder is NOT a single model** - it's a pipeline of three components (SAM + CLIP + Projector)

2. **The core encoding happens in `_pixel_values_to_embedding()`** - this method orchestrates all three components

3. **Feature fusion is crucial** - concatenating SAM (spatial) and CLIP (semantic) features creates rich representations

4. **Special tokens matter** - newline and separator tokens help the LLM understand image layout

5. **Dynamic cropping** - For large images, the encoder processes both global (1024×1024) and local (640×640 tiles) views

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `IMAGE_SIZE` or disable `CROP_MODE`

### Issue: Model weights not loading
**Solution**: Ensure you have access to `deepseek-ai/DeepSeek-OCR` on HuggingFace

### Issue: Different number of tokens than expected
**Solution**: Check `CROP_MODE` setting - dynamic cropping produces variable token counts

## References

- Main model file: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py`
- SAM encoder: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/sam_vary_sdpa.py`
- CLIP encoder: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/clip_sdpa.py`
- Projector: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/build_linear.py`

## Conclusion

The DeepEncoder can be reproduced independently by:
1. Loading the three component models (SAM, CLIP, Projector)
2. Processing images through the pipeline
3. Adding special layout tokens
4. Outputting the final embeddings

The provided `reproduce_deepencoder.py` script demonstrates this complete workflow.
