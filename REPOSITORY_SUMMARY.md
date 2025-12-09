# DeepSeek-OCR Repository Summary

**Generated on:** December 9, 2025  
**Repository:** DeepSeek-OCR  
**License:** MIT License (Copyright 2025 DeepSeek)

---

## 📋 Overview

DeepSeek-OCR is an advanced Optical Character Recognition (OCR) model developed by DeepSeek AI that investigates the role of vision encoders from an LLM-centric viewpoint. The model focuses on **Contexts Optical Compression** and provides state-of-the-art OCR capabilities for documents, images, and PDFs.

### Key Features
- **Multi-resolution Support**: Native resolutions from 512×512 to 1280×1280
- **Dynamic Resolution**: "Gundam" mode with flexible cropping
- **High Performance**: ~2500 tokens/s on A100-40G GPU
- **Versatile Applications**: Document OCR, image description, figure parsing, text localization
- **Multiple Inference Options**: Both vLLM and Transformers implementations

---

## 🗂️ Repository Structure

```
/vercel/sandbox/
├── DeepSeek_OCR_paper.pdf          # Research paper (arXiv:2510.18234)
├── LICENSE                          # MIT License
├── README.md                        # Main documentation
├── requirements.txt                 # Python dependencies
├── assets/                          # Visual assets and examples
│   ├── badge.svg
│   ├── fig1.png
│   ├── logo.svg
│   └── show1-4.jpg                 # Example visualizations
└── DeepSeek-OCR-master/
    ├── DeepSeek-OCR-hf/            # Hugging Face Transformers implementation
    │   └── run_dpsk_ocr.py
    └── DeepSeek-OCR-vllm/          # vLLM implementation
        ├── config.py                # Configuration settings
        ├── deepseek_ocr.py         # Main model implementation
        ├── run_dpsk_ocr_eval_batch.py
        ├── run_dpsk_ocr_image.py   # Image inference script
        ├── run_dpsk_ocr_pdf.py     # PDF inference script
        ├── deepencoder/            # Vision encoder modules
        └── process/                # Image processing utilities
```

---

## 🚀 Key Components

### 1. **Model Implementations**

#### vLLM Implementation (`DeepSeek-OCR-vllm/`)
- **Primary Model**: `deepseek_ocr.py` - Full vLLM-compatible implementation
- **Configuration**: `config.py` - Centralized settings for inference
- **Inference Scripts**:
  - `run_dpsk_ocr_image.py` - Streaming output for images
  - `run_dpsk_ocr_pdf.py` - Concurrent PDF processing (~2500 tokens/s)
  - `run_dpsk_ocr_eval_batch.py` - Batch evaluation for benchmarks

#### Transformers Implementation (`DeepSeek-OCR-hf/`)
- **Script**: `run_dpsk_ocr.py` - Hugging Face Transformers-based inference
- Supports flash attention 2.0 for optimized performance

### 2. **Vision Architecture**
The model combines multiple vision encoders:
- **SAM (Segment Anything Model)**: `build_sam_vit_b()` - Base vision features
- **CLIP**: `build_clip_l()` - Large CLIP model for semantic understanding
- **MLP Projector**: Linear projection layer (2048 → 1280 dimensions)
- **Special Tokens**: Image newline and view separator embeddings

### 3. **Resolution Modes**

| Mode   | Base Size | Image Size | Crop Mode | Vision Tokens |
|--------|-----------|------------|-----------|---------------|
| Tiny   | 512       | 512        | False     | 64            |
| Small  | 640       | 640        | False     | 100           |
| Base   | 1024      | 1024       | False     | 256           |
| Large  | 1280      | 1280       | False     | 400           |
| Gundam | 1024      | 640        | True      | Variable      |

**Gundam Mode**: Dynamic resolution with `n×640×640 + 1×1024×1024` tiles for large images.

---

## 🔧 Technical Details

### Dependencies
```
transformers==4.46.3
tokenizers==0.20.3
PyMuPDF                 # PDF processing
img2pdf                 # Image to PDF conversion
einops                  # Tensor operations
easydict, addict        # Configuration management
Pillow, numpy           # Image processing
vllm==0.8.5+cu118       # vLLM inference engine
torch==2.6.0            # PyTorch framework
flash-attn==2.7.3       # Flash attention optimization
```

### System Requirements
- **CUDA**: 11.8+
- **PyTorch**: 2.6.0
- **Python**: 3.12.9 (recommended)
- **GPU**: A100-40G or equivalent (for optimal performance)

### Configuration Parameters (`config.py`)
```python
BASE_SIZE = 1024              # Base resolution
IMAGE_SIZE = 640              # Tile resolution
CROP_MODE = True              # Enable dynamic cropping
MIN_CROPS = 2                 # Minimum crop tiles
MAX_CROPS = 6                 # Maximum crop tiles (max: 9)
MAX_CONCURRENCY = 100         # Concurrent processing limit
NUM_WORKERS = 64              # Image preprocessing workers
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
```

---

## 💡 Usage Examples

### 1. **Transformers Inference**
```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2',
    trust_remote_code=True
).eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown."
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file='your_image.jpg',
    base_size=1024, 
    image_size=640, 
    crop_mode=True
)
```

### 2. **vLLM Inference**
```python
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0
)

image = Image.open("path/to/image.png").convert("RGB")
prompt = "<image>\nFree OCR."

sampling_param = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822}  # <td>, </td>
    )
)

outputs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": image}}], 
                       sampling_param)
```

### 3. **Prompt Templates**
```python
# Document OCR with layout
"<image>\n<|grounding|>Convert the document to markdown."

# General OCR
"<image>\n<|grounding|>OCR this image."

# Without layout preservation
"<image>\nFree OCR."

# Figure parsing
"<image>\nParse the figure."

# Image description
"<image>\nDescribe this image in detail."

# Text localization
"<image>\nLocate <|ref|>specific text<|/ref|> in the image."
```

---

## 🎯 Use Cases

1. **Document Digitization**: Convert scanned documents to markdown/text
2. **PDF Processing**: Batch OCR for large PDF collections
3. **Figure Extraction**: Parse charts, diagrams, and figures from papers
4. **Layout Preservation**: Maintain document structure during OCR
5. **Multilingual OCR**: Support for various languages including Chinese
6. **Benchmark Evaluation**: Batch processing for OCR benchmarks (Fox, OmniDocBench)

---

## 📊 Performance Characteristics

- **Throughput**: ~2500 tokens/s on A100-40G (PDF processing with concurrency)
- **Vision Token Efficiency**: 
  - Tiny mode: 64 tokens (512×512)
  - Gundam mode: Variable based on image complexity
- **Memory Optimization**: 
  - Prefix caching support
  - Flash attention 2.0 integration
  - Configurable concurrency limits

---

## 🔗 Model Architecture

### Vision Processing Pipeline
1. **Input**: Image → Dynamic preprocessing (resize/crop)
2. **Dual Encoding**:
   - Global view: Full image at base resolution (1024×1024)
   - Local views: Cropped tiles at image resolution (640×640)
3. **Feature Extraction**:
   - SAM encoder → spatial features
   - CLIP encoder → semantic features
   - Concatenation → 2048-dim features
4. **Projection**: MLP projector → 1280-dim embeddings
5. **Token Assembly**: 
   - Add newline tokens between rows
   - Add view separator between global/local views
6. **LLM Processing**: DeepSeek language model (V2/V3)

### Special Token Handling
- `<image>`: Image placeholder token
- `<|grounding|>`: Layout-aware OCR mode
- `<|ref|>...<|/ref|>`: Text localization markers
- Image newline: Row separator in 2D tile layout
- View separator: Delimiter between global and local views

---

## 📚 Research & References

### Paper
- **Title**: DeepSeek-OCR: Contexts Optical Compression
- **Authors**: Wei, Haoran; Sun, Yaofeng; Li, Yukun
- **arXiv**: [2510.18234](https://arxiv.org/abs/2510.18234)
- **Year**: 2025

### Acknowledgments
The project builds upon:
- **Vary**: Vision-language model architecture
- **GOT-OCR2.0**: OCR methodology
- **MinerU**: Document understanding
- **PaddleOCR**: OCR toolkit
- **OneChart**: Chart understanding
- **Slow Perception**: Vision processing techniques

### Benchmarks
- **Fox**: OCR evaluation benchmark
- **OmniDocBench**: Document understanding benchmark

---

## 🌐 Community & Resources

- **Homepage**: [deepseek.com](https://www.deepseek.com/)
- **Hugging Face**: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **GitHub**: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- **Discord**: [DeepSeek AI Community](https://discord.gg/Tc7c45Zzu5)
- **Twitter**: [@deepseek_ai](https://twitter.com/deepseek_ai)

---

## 📅 Release Timeline

- **October 23, 2025**: Official vLLM upstream support
- **October 20, 2025**: Initial DeepSeek-OCR release

---

## 🎓 Citation

```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```

---

## 🔐 License

This project is licensed under the **MIT License** (Copyright 2025 DeepSeek).

Permission is granted to use, modify, and distribute the software with proper attribution. See the LICENSE file for full details.

---

## 📝 Notes

- The model requires GPU with CUDA support for optimal performance
- Flash attention 2.0 is recommended for memory efficiency
- vLLM version 0.8.5+ provides the best compatibility
- For limited GPU memory, reduce `MAX_CROPS` and `MAX_CONCURRENCY` parameters
- The model supports both streaming and batch inference modes

---

**End of Summary**
