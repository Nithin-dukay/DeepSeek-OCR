# DeepSeek-OCR Repository Analysis

**Analysis Date:** December 11, 2025  
**Repository:** DeepSeek-OCR  
**License:** MIT License (Copyright 2025 DeepSeek)

---

## Executive Summary

DeepSeek-OCR is a state-of-the-art Optical Character Recognition (OCR) model developed by DeepSeek AI that investigates the role of vision encoders from an LLM-centric viewpoint. The model focuses on **"Contexts Optical Compression"** - exploring the boundaries of visual-text compression. It was officially released on October 20, 2025, and is now supported in upstream vLLM as of October 23, 2025.

---

## 1. Project Overview

### Purpose
DeepSeek-OCR is designed to:
- Perform high-quality OCR on documents, images, and PDFs
- Convert documents to markdown format with layout preservation
- Extract text with grounding information (bounding boxes)
- Handle various document types including figures, charts, and complex layouts
- Support multiple resolution modes for different use cases

### Key Features
- **Multiple Resolution Modes**: Tiny (512×512), Small (640×640), Base (1024×1024), Large (1280×1280), and Dynamic "Gundam" mode
- **Vision Token Compression**: Reduces vision tokens significantly (64-400 tokens depending on mode)
- **Layout Preservation**: Maintains document structure with grounding tags
- **Batch Processing**: Efficient concurrent processing for PDFs and image batches
- **Streaming Output**: Real-time text generation for interactive use
- **Multi-modal Support**: Handles images, PDFs, and various document formats

---

## 2. Architecture & Technical Stack

### Core Technologies
- **Framework**: PyTorch 2.6.0 with CUDA 11.8
- **Inference Engine**: vLLM 0.8.5 (with upstream support in v0.11.1+)
- **Transformers**: HuggingFace Transformers 4.46.3
- **Vision Encoders**: 
  - SAM (Segment Anything Model) - ViT-B variant
  - CLIP - Large variant
- **Language Model**: DeepSeek V2/V3 architecture with MLA (Multi-head Latent Attention)

### Model Architecture Components

#### Vision Processing Pipeline
1. **Dual Vision Encoders**:
   - `sam_model`: SAM ViT-B for spatial feature extraction
   - `vision_model`: CLIP-L for semantic understanding
   - Features are concatenated: `[CLIP_features, SAM_features]`

2. **MLP Projector**:
   - Projects concatenated vision features (2048-dim) to language model embedding space (1280-dim)
   - Linear projection type

3. **Dynamic Resolution Handling**:
   - Global view: Base resolution (e.g., 1024×1024)
   - Local views: Cropped patches (e.g., 640×640 tiles)
   - Adaptive tiling based on aspect ratio (2-9 tiles)

4. **Special Tokens**:
   - `<image>`: Image placeholder token
   - `<|grounding|>`: Enables layout/bounding box output
   - `<|ref|>...<|/ref|>`: Reference tags for object detection
   - `<|det|>...<|/det|>`: Detection coordinate tags
   - `image_newline`: Token for row separation in vision features
   - `view_separator`: Token separating global and local views

#### Language Model
- Based on DeepSeek V2/V3 architecture
- Supports topk_method "noaux_tc" for DeepSeek V3
- Multi-head Latent Attention (MLA) for efficient processing
- Max context length: 8192 tokens

---

## 3. Repository Structure

```
/vercel/sandbox/
├── DeepSeek_OCR_paper.pdf          # Research paper
├── LICENSE                          # MIT License
├── README.md                        # Documentation
├── requirements.txt                 # Python dependencies
├── assets/                          # Images and badges
│   ├── badge.svg, logo.svg
│   ├── fig1.png                     # Architecture diagram
│   └── show1-4.jpg                  # Example outputs
└── DeepSeek-OCR-master/
    ├── DeepSeek-OCR-hf/            # HuggingFace Transformers implementation
    │   └── run_dpsk_ocr.py         # Simple inference script
    └── DeepSeek-OCR-vllm/          # vLLM implementation (production)
        ├── config.py                # Configuration settings
        ├── deepseek_ocr.py         # Main model implementation
        ├── run_dpsk_ocr_image.py   # Single image inference (streaming)
        ├── run_dpsk_ocr_pdf.py     # PDF batch processing
        ├── run_dpsk_ocr_eval_batch.py  # Benchmark evaluation
        ├── process/
        │   ├── image_process.py    # Image preprocessing & dynamic tiling
        │   └── ngram_norepeat.py   # N-gram repetition prevention
        └── deepencoder/
            ├── build_linear.py     # MLP projector
            ├── clip_sdpa.py        # CLIP encoder with SDPA
            └── sam_vary_sdpa.py    # SAM encoder with SDPA
```

---

## 4. Implementation Details

### 4.1 Configuration (`config.py`)

Key configurable parameters:
```python
BASE_SIZE = 1024              # Global view resolution
IMAGE_SIZE = 640              # Local view tile size
CROP_MODE = True              # Enable dynamic tiling
MIN_CROPS = 2                 # Minimum number of tiles
MAX_CROPS = 6                 # Maximum tiles (9 max, 6 for memory)
MAX_CONCURRENCY = 100         # Batch processing concurrency
NUM_WORKERS = 64              # Image preprocessing workers
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
```

Supported prompts:
- `"<image>\n<|grounding|>Convert the document to markdown."` - Document OCR with layout
- `"<image>\nFree OCR."` - Plain text extraction without layout
- `"<image>\n<|grounding|>OCR this image."` - General image OCR
- `"<image>\nParse the figure."` - Figure/chart parsing
- `"<image>\nDescribe this image in detail."` - Image captioning
- `"<image>\nLocate <|ref|>xxxx<|/ref|> in the image."` - Object localization

### 4.2 Vision Processing (`deepseek_ocr.py`)

**Dynamic Tiling Algorithm**:
```python
def count_tiles(width, height, image_size=640):
    aspect_ratio = width / height
    # Generate candidate tile configurations (1x1 to 3x3, etc.)
    target_ratios = [(i,j) for n in range(MIN_CROPS, MAX_CROPS+1) 
                     for i,j in range(1,n+1) if i*j <= MAX_CROPS]
    # Select closest aspect ratio match
    best_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios)
    return best_ratio  # e.g., (2, 3) for 2 columns × 3 rows
```

**Feature Extraction Pipeline**:
```python
# For each image:
# 1. Extract global features (base resolution)
global_features_1 = sam_model(image_base)           # SAM features
global_features_2 = vision_model(image_base, ...)   # CLIP features
global_features = concat([CLIP[:, 1:], SAM_flat])   # Combine
global_features = projector(global_features)        # Project to LLM space

# 2. Extract local features (if cropped)
local_features_1 = sam_model(image_patches)
local_features_2 = vision_model(image_patches, ...)
local_features = concat([CLIP[:, 1:], SAM_flat])
local_features = projector(local_features)

# 3. Add special tokens and reshape
global_features = add_newlines(global_features)     # Add row separators
local_features = add_newlines(local_features)
final_features = concat([local_features, global_features, view_separator])
```

**Token Count Calculation**:
- Global view: `h × (w + 1)` tokens where `h = w = ceil((base_size/16)/4)`
  - Example: 1024×1024 → 16×17 = 272 tokens (16×16 grid + newlines)
- Local views: `(h_tiles × h2) × (w_tiles × w2 + 1)` tokens
  - Example: 2×3 tiles of 640×640 → 30×21 = 630 tokens
- Total: global + local + 1 separator token

### 4.3 Inference Scripts

#### Single Image Streaming (`run_dpsk_ocr_image.py`)
- Uses `AsyncLLMEngine` for streaming output
- Real-time token generation with `async for` loop
- Post-processing: extracts bounding boxes, crops images, generates visualizations
- Supports special outputs: geometry diagrams, structural formulas

#### PDF Batch Processing (`run_dpsk_ocr_pdf.py`)
- Converts PDF to images at 144 DPI
- Parallel preprocessing with `ThreadPoolExecutor` (64 workers)
- Batch inference with vLLM (up to 100 concurrent sequences)
- Throughput: ~2500 tokens/s on A100-40G
- Outputs: markdown files, cropped images, annotated PDF with bounding boxes

#### HuggingFace Transformers (`run_dpsk_ocr.py`)
- Simple single-image inference
- Uses `model.infer()` method with flash attention 2
- Suitable for quick testing and prototyping

### 4.4 Advanced Features

#### N-gram Repetition Prevention (`ngram_norepeat.py`)
```python
class NoRepeatNGramLogitsProcessor:
    def __init__(self, ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822}):
        # Prevents repetition of n-grams within a sliding window
        # Whitelist: <td>, </td> tokens (allowed to repeat in tables)
```

#### Image Preprocessing (`image_process.py`)
- EXIF orientation correction
- Dynamic aspect ratio-based tiling
- Normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
- Efficient batching with `BatchFeature`

#### Grounding Output Parsing
- Regex pattern: `<|ref|>(.*?)</|ref|><|det|>(.*?)</|det|>`
- Coordinate format: `[[x1, y1, x2, y2], ...]` (normalized 0-999)
- Label types: image, title, text, table, figure, etc.
- Post-processing: draws bounding boxes, crops regions, replaces with markdown image links

---

## 5. Dependencies

### Core Requirements (`requirements.txt`)
```
transformers==4.46.3
tokenizers==0.20.3
PyMuPDF                 # PDF processing
img2pdf                 # PDF generation
einops                  # Tensor operations
easydict, addict        # Configuration management
Pillow                  # Image processing
numpy
```

### Additional Requirements (from README)
```
torch==2.6.0
torchvision==0.21.0
vllm==0.8.5+cu118       # Inference engine
flash-attn==2.7.3       # Flash attention for efficiency
```

### System Requirements
- CUDA 11.8+
- Python 3.12.9 (recommended)
- GPU: A100-40G recommended for production (can run on smaller GPUs with reduced concurrency)

---

## 6. Usage Modes

### Resolution Modes
| Mode   | Base Size | Image Size | Crop Mode | Vision Tokens | Use Case |
|--------|-----------|------------|-----------|---------------|----------|
| Tiny   | 512       | 512        | False     | 64            | Low-res documents |
| Small  | 640       | 640        | False     | 100           | Standard documents |
| Base   | 1024      | 1024       | False     | 256           | High-quality single page |
| Large  | 1280      | 1280       | False     | 400           | Very high resolution |
| Gundam | 1024      | 640        | True      | 256 + n×100   | Dynamic multi-page |

**Gundam Mode** (Dynamic Resolution):
- Automatically tiles large images into 640×640 patches
- Adds a 1024×1024 global view for context
- Optimal for documents with varying sizes and aspect ratios

---

## 7. Performance Characteristics

### Throughput
- **PDF Processing**: ~2500 tokens/s on A100-40G with concurrency=100
- **Streaming**: Real-time output for interactive applications
- **Batch Processing**: Efficient parallel preprocessing with 64 workers

### Memory Optimization
- `MAX_CROPS=6` recommended for limited GPU memory (max is 9)
- `gpu_memory_utilization=0.75-0.9` configurable
- Prefix caching disabled for multi-modal inputs
- `disable_mm_preprocessor_cache=True` for batch processing

### Quality Features
- Flash Attention 2 for efficient long-context processing
- N-gram repetition prevention for clean output
- EXIF-aware image loading
- High-DPI PDF rendering (144 DPI default)

---

## 8. Integration Points

### vLLM Integration (Upstream)
As of October 23, 2025, DeepSeek-OCR is officially supported in vLLM:
```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)
```

### HuggingFace Hub
- Model: `deepseek-ai/DeepSeek-OCR`
- Supports `trust_remote_code=True` for custom model code
- Compatible with `AutoModel` and `AutoTokenizer`

### Custom Processor Registration
```python
from vllm.model_executor.models.registry import ModelRegistry
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
```

---

## 9. Output Formats

### Markdown with Grounding
```markdown
# Document Title <|ref|>title<|/ref|><|det|>[[50, 100, 950, 200]]<|/det|>

This is text content...

![](images/0.jpg)  # Extracted figure

| Column 1 | Column 2 |
|----------|----------|
| Data     | Data     |
```

### Processed Outputs
1. **`result_ori.mmd`**: Raw output with grounding tags
2. **`result.mmd`**: Clean markdown with image links
3. **`result_with_boxes.jpg`**: Annotated image with bounding boxes
4. **`images/`**: Cropped regions (figures, images)
5. **`_layouts.pdf`**: PDF with visual annotations

### Special Formats
- **Geometry**: Matplotlib diagrams for geometric problems
- **Chemistry**: SMILES format for structural formulas
- **Tables**: Markdown table format with proper alignment

---

## 10. Benchmarks & Evaluation

### Supported Benchmarks
- **Fox**: Document understanding benchmark
- **OmniDocBench**: Comprehensive document OCR evaluation
- Batch evaluation script: `run_dpsk_ocr_eval_batch.py`

### Evaluation Features
- Skip repeated outputs with `SKIP_REPEAT=True`
- Configurable concurrency for throughput testing
- Automatic result aggregation and formatting

---

## 11. Acknowledgements

The project builds upon and acknowledges:
- **Vary**: Vision-language model architecture
- **GOT-OCR2.0**: OCR methodology
- **MinerU**: Document understanding
- **PaddleOCR**: OCR techniques
- **OneChart**: Chart parsing
- **Slow Perception**: Vision processing concepts

---

## 12. Research Paper

**Title**: DeepSeek-OCR: Contexts Optical Compression  
**Authors**: Wei, Haoran; Sun, Yaofeng; Li, Yukun  
**Publication**: arXiv preprint arXiv:2510.18234 (2025)  
**Available**: `/vercel/sandbox/DeepSeek_OCR_paper.pdf`

---

## 13. Key Innovations

1. **Dual Vision Encoder Architecture**: Combines SAM (spatial) and CLIP (semantic) for comprehensive understanding
2. **Optical Compression**: Reduces vision tokens by 10-100× compared to traditional approaches
3. **Dynamic Resolution**: Adaptive tiling based on content and aspect ratio
4. **Grounding Integration**: Seamless layout preservation with coordinate output
5. **Production-Ready**: vLLM integration for high-throughput deployment
6. **Multi-Modal Flexibility**: Handles images, PDFs, and various document types uniformly

---

## 14. Limitations & Considerations

1. **GPU Memory**: Large images with many tiles can exceed memory limits
   - Solution: Reduce `MAX_CROPS` or use smaller resolution modes
2. **Repetition Handling**: N-gram processor may occasionally suppress valid repetitions
   - Solution: Adjust `ngram_size` and `window_size` parameters
3. **CUDA Version**: Requires CUDA 11.8+ for optimal performance
4. **Interactive Commands**: Not supported in shell execution
5. **Context Length**: Limited to 8192 tokens (may truncate very long documents)

---

## 15. Future Directions

Based on the codebase structure:
- Support for more vision encoder backends
- Extended context length for longer documents
- Additional output formats (HTML, LaTeX)
- Fine-tuning scripts for domain-specific OCR
- Multi-language support enhancements
- Real-time video OCR capabilities

---

## Conclusion

DeepSeek-OCR represents a significant advancement in document understanding and OCR technology. Its dual-encoder architecture, dynamic resolution handling, and production-ready vLLM integration make it suitable for both research and industrial applications. The model excels at preserving document layouts while achieving high compression ratios, making it ideal for large-scale document processing pipelines.

**Strengths**:
- State-of-the-art OCR quality with layout preservation
- Efficient vision token compression
- Production-ready with vLLM support
- Flexible resolution modes for different use cases
- Comprehensive grounding and bounding box support

**Best Use Cases**:
- Document digitization and archival
- PDF to markdown conversion
- Academic paper processing
- Form and table extraction
- Multi-modal document understanding
- Large-scale document analysis pipelines

---

**Analysis completed on December 11, 2025**
