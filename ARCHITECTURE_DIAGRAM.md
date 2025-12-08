# DeepEncoder Architecture Diagram

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DeepSeek-OCR System                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │                    DeepEncoder (Vision)                    │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │   SAM    │→ │   CLIP   │→ │  Concat  │→ │Projector │  │    │
│  │  │ Encoder  │  │ Encoder  │  │          │  │          │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │    │
│  └───────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│                    Vision Embeddings                                │
│                       [N, 1280]                                     │
│                              ↓                                      │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │              Language Model (DeepSeek-V2/V3)              │    │
│  │                                                            │    │
│  │  Processes vision embeddings + text tokens                │    │
│  └───────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│                        OCR Output Text                              │
└─────────────────────────────────────────────────────────────────────┘
```

## DeepEncoder Detailed Architecture

```
Input Image: [B, 3, H, W]
│
├─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STEP 1: SAM Encoder (Spatial Feature Extraction)                  │
│  ════════════════════════════════════════════════                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  ImageEncoderViT (ViT-B)                                │      │
│  │  ┌────────────────────────────────────────────────┐     │      │
│  │  │  Patch Embedding (16×16 patches)               │     │      │
│  │  │  Input: [B, 3, H, W]                           │     │      │
│  │  │  Output: [B, H/16, W/16, 768]                  │     │      │
│  │  └────────────────────────────────────────────────┘     │      │
│  │                      ↓                                   │      │
│  │  ┌────────────────────────────────────────────────┐     │      │
│  │  │  12 Transformer Blocks                         │     │      │
│  │  │  - Multi-head attention (12 heads)             │     │      │
│  │  │  - MLP with GELU                               │     │      │
│  │  │  - Relative position encoding                  │     │      │
│  │  └────────────────────────────────────────────────┘     │      │
│  │                      ↓                                   │      │
│  │  ┌────────────────────────────────────────────────┐     │      │
│  │  │  Neck (Conv layers)                            │     │      │
│  │  │  768 → 256 → 512 → 1024                        │     │      │
│  │  │  Output: [B, 1024, H/64, W/64]                 │     │      │
│  │  └────────────────────────────────────────────────┘     │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    SAM Features: [B, 1024, H/64, W/64]
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STEP 2: CLIP Encoder (Semantic Feature Extraction)                │
│  ═════════════════════════════════════════════════                 │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  VitModel (ViT-L)                                       │      │
│  │  ┌────────────────────────────────────────────────┐     │      │
│  │  │  Vision Embeddings                             │     │      │
│  │  │  - Patch embedding from SAM features           │     │      │
│  │  │  - Add CLS token                               │     │      │
│  │  │  - Add positional embeddings                   │     │      │
│  │  │  Output: [B, seq_len+1, 1024]                  │     │      │
│  │  └────────────────────────────────────────────────┘     │      │
│  │                      ↓                                   │      │
│  │  ┌────────────────────────────────────────────────┐     │      │
│  │  │  24 Transformer Blocks                         │     │      │
│  │  │  - Multi-head attention (16 heads)             │     │      │
│  │  │  - MLP with QuickGELU                          │     │      │
│  │  │  - Layer normalization                         │     │      │
│  │  └────────────────────────────────────────────────┘     │      │
│  │                      ↓                                   │      │
│  │  ┌────────────────────────────────────────────────┐     │      │
│  │  │  Pre-LayerNorm                                 │     │      │
│  │  │  Output: [B, seq_len+1, 1024]                  │     │      │
│  │  └────────────────────────────────────────────────┘     │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    CLIP Features: [B, seq_len+1, 1024]
                              ↓
                    Remove CLS token: [B, seq_len, 1024]
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STEP 3: Feature Concatenation                                     │
│  ════════════════════════════                                      │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Flatten SAM features:                                  │      │
│  │  [B, 1024, H/64, W/64] → [B, seq_len, 1024]            │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Concatenate along feature dimension:                   │      │
│  │                                                          │      │
│  │  CLIP [B, seq_len, 1024] ┐                              │      │
│  │                          ├→ [B, seq_len, 2048]          │      │
│  │  SAM  [B, seq_len, 1024] ┘                              │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    Combined Features: [B, seq_len, 2048]
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STEP 4: MLP Projector                                             │
│  ═══════════════════                                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Linear Projection                                      │      │
│  │  Input:  [B, seq_len, 2048]                             │      │
│  │  Weight: [2048, 1280]                                   │      │
│  │  Output: [B, seq_len, 1280]                             │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    Projected Features: [B, seq_len, 1280]
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STEP 5: Add Layout Tokens                                         │
│  ════════════════════════                                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Reshape to 2D grid: [B, h, w, 1280]                    │      │
│  │  where h = w = sqrt(seq_len)                            │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Add newline token at end of each row:                  │      │
│  │                                                          │      │
│  │  [patch_0] [patch_1] ... [patch_w-1] [newline]          │      │
│  │  [patch_w] [patch_w+1] ... [patch_2w-1] [newline]       │      │
│  │  ...                                                     │      │
│  │  [patch_hw-w] ... [patch_hw-1] [newline]                │      │
│  │                                                          │      │
│  │  Output: [B, h*(w+1), 1280]                             │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Flatten: [B, h*(w+1), 1280]                            │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Add view separator at end:                             │      │
│  │  [...all tokens...] [separator]                         │      │
│  │                                                          │      │
│  │  Output: [h*(w+1)+1, 1280]                              │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    Final Vision Embeddings
                    [num_tokens, 1280]
```

## Token Layout Visualization

### Single View (No Cropping)

For a 1024×1024 image with base_size=1024:

```
Grid: 16×16 patches (after downsampling)

┌─────────────────────────────────────────────────────────┐
│ [0]  [1]  [2]  ... [14] [15] [NL]  ← Row 0 + newline   │
│ [16] [17] [18] ... [30] [31] [NL]  ← Row 1 + newline   │
│ [32] [33] [34] ... [46] [47] [NL]  ← Row 2 + newline   │
│  .    .    .         .    .   .                         │
│  .    .    .         .    .   .                         │
│  .    .    .         .    .   .                         │
│ [240][241][242]...[254][255][NL]  ← Row 15 + newline   │
│ [SEP]                              ← View separator     │
└─────────────────────────────────────────────────────────┘

Total tokens: 16 rows × 17 tokens/row + 1 separator = 273 tokens
```

### Multi-View (With Cropping)

For a large image with crop_mode=True:

```
┌─────────────────────────────────────────────────────────┐
│                    LOCAL VIEW (Crops)                   │
│  ┌──────────┬──────────┬──────────┐                     │
│  │ Crop 0,0 │ Crop 1,0 │ Crop 2,0 │  ← 3 crops wide    │
│  ├──────────┼──────────┼──────────┤                     │
│  │ Crop 0,1 │ Crop 1,1 │ Crop 2,1 │  ← 2 crops tall    │
│  └──────────┴──────────┴──────────┘                     │
│                                                          │
│  Each crop: 10×10 patches = 111 tokens (with newlines)  │
│  Total local: 3×2 crops × 111 = 666 tokens              │
└─────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────┐
│                   GLOBAL VIEW (Full)                    │
│  ┌────────────────────────────────────────────┐         │
│  │         Full image at 1024×1024            │         │
│  │         16×16 patches = 273 tokens         │         │
│  └────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
                              +
                        [SEPARATOR]
                              =
                Total: 666 + 273 + 1 = 940 tokens
```

## Feature Dimensions at Each Stage

```
Stage                          Shape                    Description
─────────────────────────────────────────────────────────────────────
Input Image                    [B, 3, H, W]            RGB image

SAM Patch Embedding            [B, H/16, W/16, 768]    16×16 patches
SAM Transformer Blocks         [B, H/16, W/16, 768]    12 layers
SAM Neck Conv1                 [B, H/16, W/16, 256]    First conv
SAM Neck Conv2                 [B, H/32, W/32, 512]    Downsample
SAM Neck Conv3                 [B, H/64, W/64, 1024]   Final output

CLIP Patch Embedding           [B, seq_len+1, 1024]    With CLS token
CLIP Transformer Blocks        [B, seq_len+1, 1024]    24 layers
CLIP Output (no CLS)           [B, seq_len, 1024]      Remove CLS

SAM Flattened                  [B, seq_len, 1024]      Spatial→Sequential
Concatenated                   [B, seq_len, 2048]      CLIP + SAM

Projected                      [B, seq_len, 1280]      Linear projection
With Newlines                  [B, h*(w+1), 1280]      Add row markers
With Separator                 [h*(w+1)+1, 1280]       Final output
```

## Code Mapping

```python
# File: deepseek_ocr.py

class DeepseekOCRForCausalLM:
    
    def __init__(self):
        # Initialize components
        self.sam_model = build_sam_vit_b()        # SAM encoder
        self.vision_model = build_clip_l()        # CLIP encoder
        self.projector = MlpProjector(...)        # Projector
        self.image_newline = nn.Parameter(...)    # Newline token
        self.view_seperator = nn.Parameter(...)   # Separator token
    
    def get_multimodal_embeddings(self, **kwargs):
        """Entry point - called from forward()"""
        image_input = self._parse_and_validate_image_input(**kwargs)
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
    
    def _process_image_input(self, image_input):
        """Preprocess and route to encoding"""
        pixel_values, images_crop, images_spatial_crop = image_input
        vision_features = self._pixel_values_to_embedding(
            pixel_values, images_crop, images_spatial_crop
        )
        return vision_features
    
    def _pixel_values_to_embedding(self, pixel_values, images_crop, images_spatial_crop):
        """
        ⭐ CORE DEEPENCODER LOGIC ⭐
        This is the main encoding pipeline
        """
        images_in_batch = []
        
        for image, crops, crop_shape in zip(...):
            # Step 1: SAM encoder
            sam_features = self.sam_model(image)
            
            # Step 2: CLIP encoder
            clip_features = self.vision_model(image, sam_features)
            
            # Step 3: Concatenate
            combined = torch.cat([
                clip_features[:, 1:],  # Remove CLS
                sam_features.flatten(2).permute(0, 2, 1)
            ], dim=-1)
            
            # Step 4: Project
            embeddings = self.projector(combined)
            
            # Step 5: Add layout tokens
            embeddings = self._add_layout_tokens(embeddings)
            
            images_in_batch.append(embeddings)
        
        return images_in_batch
```

## Performance Characteristics

```
Mode      Resolution  Tokens  Memory    Speed      Use Case
────────────────────────────────────────────────────────────
Tiny      512×512     73      ~2GB      Fastest    Quick preview
Small     640×640     111     ~3GB      Fast       Standard docs
Base      1024×1024   273     ~6GB      Medium     High quality
Large     1280×1280   427     ~10GB     Slow       Max detail
Gundam    Dynamic     Varies  ~8-15GB   Variable   Large docs
```

## Key Insights

1. **Two-Stage Encoding**: SAM captures structure, CLIP captures semantics
2. **Feature Fusion**: Concatenation preserves both types of information
3. **Layout Tokens**: Help LLM understand 2D spatial relationships
4. **Dynamic Cropping**: Handles large images by processing tiles
5. **Efficient Design**: Linear projector keeps computation manageable

---

This diagram shows the complete DeepEncoder architecture from input image to final embeddings.
