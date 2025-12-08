# DeepEncoder Standalone - Solution for Issue #289

## 📋 Overview

This solution provides a **standalone implementation of DeepEncoder** from DeepSeek-OCR, allowing you to extract vision embeddings from images independently without running the full OCR pipeline.

### What is DeepEncoder?

DeepEncoder is the vision encoding component of DeepSeek-OCR that transforms images into rich embeddings suitable for language model processing. It combines:

- **SAM Encoder**: Extracts spatial/structural features (1024-dim)
- **CLIP Encoder**: Extracts semantic/content features (1024-dim)  
- **MLP Projector**: Fuses and projects features to embedding space (1280-dim)

## 🎯 Key Question from Issue #289

> **Q: Is `vision_embeddings = self.get_multimodal_embeddings(**kwargs)` the core code of DeepEncoder?**

**A: Yes!** This line calls the complete encoding pipeline. The actual core implementation is in the `_pixel_values_to_embedding()` method (line ~450 in `deepseek_ocr.py`), which:

1. Runs SAM encoder: `sam_features = self.sam_model(image)`
2. Runs CLIP encoder: `clip_features = self.vision_model(image, sam_features)`
3. Concatenates features: `combined = torch.cat([clip_features, sam_features], dim=-1)`
4. Projects to embeddings: `embeddings = self.projector(combined)`
5. Adds layout tokens (newlines, separators)

## 📁 Files Provided

```
.
├── SOLUTION_ISSUE_289.md          # Detailed technical explanation
├── reproduce_deepencoder.py       # Standalone DeepEncoder implementation
├── example_deepencoder_usage.py   # Usage examples and tutorials
└── README_DEEPENCODER.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision transformers pillow einops addict
```

### 2. Run Examples

```bash
# Run tutorial examples (uses random weights for demo)
python example_deepencoder_usage.py
```

### 3. Encode Your Own Images

```bash
# Encode a single image (requires model weights)
python reproduce_deepencoder.py \
    --image_path your_image.jpg \
    --output_path embeddings.pt \
    --base_size 1024 \
    --image_size 640 \
    --crop_mode
```

## 💡 Usage Examples

### Example 1: Basic Encoding

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

print(f"Embeddings shape: {embeddings.shape}")
# Output: Embeddings shape: torch.Size([num_tokens, 1280])
```

### Example 2: Access Intermediate Features

```python
# Get all intermediate features
result = encoder.encode_image(image, return_intermediate=True)

print(f"SAM features: {result['sam_features'].shape}")
print(f"CLIP features: {result['clip_features'].shape}")
print(f"Combined: {result['combined_features'].shape}")
print(f"Final: {result['embeddings'].shape}")
```

### Example 3: Batch Processing

```python
images = [Image.open(f"img_{i}.jpg") for i in range(5)]
embeddings_list = encoder.encode_batch(images)

for i, emb in enumerate(embeddings_list):
    print(f"Image {i}: {emb.shape[0]} tokens")
```

## 🔧 Configuration Modes

DeepSeek-OCR supports different encoding modes:

| Mode | base_size | image_size | crop_mode | Tokens | Use Case |
|------|-----------|------------|-----------|--------|----------|
| **Tiny** | 512 | 512 | False | 64 | Fast processing |
| **Small** | 640 | 640 | False | 100 | Balanced |
| **Base** | 1024 | 1024 | False | 256 | High quality |
| **Large** | 1280 | 1280 | False | 400 | Maximum detail |
| **Gundam** | 1024 | 640 | True | Variable | Dynamic cropping |

### Setting Modes

```python
# Tiny mode (fastest)
encoder = DeepEncoderStandalone(base_size=512, image_size=512, crop_mode=False)

# Base mode (recommended)
encoder = DeepEncoderStandalone(base_size=1024, image_size=1024, crop_mode=False)

# Gundam mode (for large documents)
encoder = DeepEncoderStandalone(base_size=1024, image_size=640, crop_mode=True)
```

## 🏗️ Architecture Details

### Encoding Pipeline

```
Input Image (RGB)
    ↓
┌─────────────────────────────────────┐
│  1. SAM Encoder (Spatial)           │
│     - ViT-B architecture            │
│     - Output: [B, 1024, H/16, W/16] │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. CLIP Encoder (Semantic)         │
│     - ViT-L architecture            │
│     - Input: image + SAM features   │
│     - Output: [B, seq_len, 1024]    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Feature Concatenation           │
│     - Concat: CLIP + SAM            │
│     - Output: [B, seq_len, 2048]    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. MLP Projector                   │
│     - Linear projection             │
│     - Output: [B, seq_len, 1280]    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  5. Add Layout Tokens               │
│     - image_newline (end of row)    │
│     - view_separator (end of view)  │
└─────────────────────────────────────┘
    ↓
Vision Embeddings [num_tokens, 1280]
```

### Token Calculation

For a single view (no cropping):

```
h = w = (base_size / patch_size) / downsample_ratio
num_tokens = h * (w + 1) + 1

Where:
  - patch_size = 16
  - downsample_ratio = 4
  - +1 per row for newline token
  - +1 at end for separator token
```

Examples:
- 512×512: 8×9 + 1 = **73 tokens**
- 640×640: 10×11 + 1 = **111 tokens**
- 1024×1024: 16×17 + 1 = **273 tokens**

## 📊 Understanding the Output

### Embedding Structure

```python
embeddings = encoder.encode_image(image)
# Shape: [num_tokens, 1280]

# Structure:
# [token_0]      ← First patch
# [token_1]      ← Second patch
# ...
# [token_w]      ← Last patch in row 1
# [newline]      ← End of row 1
# [token_w+1]    ← First patch in row 2
# ...
# [separator]    ← End of view
```

### Special Tokens

- **image_newline**: Marks the end of each row (like `\n` in text)
- **view_separator**: Marks the end of a view (global or local)

These tokens help the LLM understand the 2D layout of the image.

## 🔍 Code Walkthrough

### Core Method: `_pixel_values_to_embedding()`

This is the heart of DeepEncoder (from `deepseek_ocr.py`):

```python
def _pixel_values_to_embedding(self, pixel_values, images_crop, images_spatial_crop):
    """
    Core DeepEncoder logic - converts pixels to embeddings
    
    Args:
        pixel_values: Global view image [B, 3, H, W]
        images_crop: Local view crops [B, N, 3, h, w]
        images_spatial_crop: Crop configuration [B, [w_tiles, h_tiles]]
    
    Returns:
        List of vision embeddings, one per image
    """
    
    # For each image in batch
    for image, crops, crop_shape in zip(pixel_values, images_crop, images_spatial_crop):
        
        # Process local crops (if any)
        if has_crops:
            # 1. SAM encoder on crops
            local_sam = self.sam_model(crops)
            
            # 2. CLIP encoder on crops
            local_clip = self.vision_model(crops, local_sam)
            
            # 3. Concatenate and project
            local_features = torch.cat([local_clip[:, 1:], 
                                       local_sam.flatten(2).permute(0, 2, 1)], 
                                      dim=-1)
            local_embeddings = self.projector(local_features)
        
        # Process global view
        # 1. SAM encoder
        global_sam = self.sam_model(image)
        
        # 2. CLIP encoder
        global_clip = self.vision_model(image, global_sam)
        
        # 3. Concatenate and project
        global_features = torch.cat([global_clip[:, 1:],
                                    global_sam.flatten(2).permute(0, 2, 1)],
                                   dim=-1)
        global_embeddings = self.projector(global_features)
        
        # 4. Add layout tokens
        # Add newline at end of each row
        # Add separator at end
        
        # 5. Combine local and global (if applicable)
        final_embeddings = combine_views(local_embeddings, global_embeddings)
        
    return final_embeddings
```

## 🎓 Learning Resources

### Understanding Each Component

1. **SAM Encoder** (`sam_vary_sdpa.py`)
   - Based on Segment Anything Model (SAM)
   - ViT-B architecture with 12 layers
   - Captures spatial structure and boundaries
   - Output: 1024-dim features per patch

2. **CLIP Encoder** (`clip_sdpa.py`)
   - Based on CLIP vision encoder
   - ViT-L architecture with 24 layers
   - Captures semantic content
   - Takes SAM features as additional input
   - Output: 1024-dim features per token

3. **MLP Projector** (`build_linear.py`)
   - Simple linear projection
   - Maps 2048-dim → 1280-dim
   - Aligns vision features with LLM embedding space

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce resolution or disable cropping

```python
# Use smaller base_size
encoder = DeepEncoderStandalone(base_size=640, crop_mode=False)

# Or use CPU
encoder = DeepEncoderStandalone(device="cpu")
```

### Issue: Model weights not loading

**Solution**: Ensure HuggingFace access

```bash
# Login to HuggingFace
huggingface-cli login

# Or specify local path
encoder = DeepEncoderStandalone(model_path="/path/to/local/model")
```

### Issue: Different token count than expected

**Solution**: Check crop_mode setting

```python
# Calculate expected tokens
num_tokens = encoder.get_num_tokens(image)
print(f"Expected: {num_tokens} tokens")
```

## 📚 References

### Original Code Locations

- **Main model**: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py`
  - Line ~450: `_pixel_values_to_embedding()` - Core encoding logic
  - Line ~540: `_process_image_input()` - Image preprocessing
  - Line ~570: `get_multimodal_embeddings()` - Entry point

- **SAM Encoder**: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/sam_vary_sdpa.py`
  - `ImageEncoderViT` class
  - `build_sam_vit_b()` function

- **CLIP Encoder**: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/clip_sdpa.py`
  - `VitModel` class
  - `build_clip_l()` function

- **Projector**: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/build_linear.py`
  - `MlpProjector` class

### Papers and Documentation

- DeepSeek-OCR Paper: `DeepSeek_OCR_paper.pdf`
- ArXiv: https://arxiv.org/abs/2510.18234
- HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-OCR

## ✅ Summary

### Key Takeaways

1. **DeepEncoder is a pipeline**, not a single model:
   - SAM (spatial) + CLIP (semantic) + Projector

2. **Core code location**:
   - `_pixel_values_to_embedding()` in `deepseek_ocr.py`

3. **Can be used independently**:
   - Extract vision embeddings without full OCR
   - Use for custom downstream tasks

4. **Flexible configuration**:
   - Multiple resolution modes
   - Dynamic cropping for large images
   - Adjustable for speed/quality tradeoff

### Next Steps

1. ✅ Read `SOLUTION_ISSUE_289.md` for detailed explanation
2. ✅ Run `example_deepencoder_usage.py` to see examples
3. ✅ Use `reproduce_deepencoder.py` for your own images
4. ✅ Integrate into your own projects

## 🤝 Contributing

If you find issues or have improvements, please:
1. Check existing issues on GitHub
2. Create a new issue with details
3. Submit a pull request with fixes

## 📄 License

This solution follows the same license as DeepSeek-OCR.

---

**Created for GitHub Issue #289**  
*Helping users understand and reproduce DeepEncoder functionality*
