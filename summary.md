# DeepSeek-OCR Summary

## Overview
DeepSeek-OCR is an advanced optical character recognition (OCR) model that investigates the role of vision encoders from an LLM-centric viewpoint. The project focuses on visual-text compression and context optical compression, providing state-of-the-art OCR capabilities for documents, images, and PDFs.

## Key Features
- **Context Optical Compression**: Advanced compression techniques for visual-text processing
- **Multiple Resolution Modes**: Support for native and dynamic resolutions ranging from 512×512 to 1280×1280
- **High Performance**: Achieves ~2500 tokens/s concurrency on A100-40G GPU
- **Flexible Inference**: Supports both vLLM and Transformers inference methods
- **Batch Processing**: Capable of handling batch evaluations for benchmarks

## Model Capabilities

### Supported Resolution Modes
**Native Resolution:**
- Tiny: 512×512 (64 vision tokens)
- Small: 640×640 (100 vision tokens)
- Base: 1024×1024 (256 vision tokens)
- Large: 1280×1280 (400 vision tokens)

**Dynamic Resolution:**
- Gundam: n×640×640 + 1×1024×1024

### Use Cases
- Document conversion to markdown
- Image OCR with or without layouts
- Figure parsing in documents
- Detailed image description
- Text localization in images

## Technical Stack

### Environment Requirements
- CUDA 11.8
- PyTorch 2.6.0
- Python 3.12.9
- vLLM 0.8.5 or upstream vLLM (0.11.1+)

### Core Dependencies
- transformers (4.46.3)
- tokenizers (0.20.3)
- PyMuPDF (PDF processing)
- img2pdf (image conversion)
- einops (tensor operations)
- Flash Attention 2.7.3

## Inference Methods

### 1. vLLM Inference
- Streaming output for images
- High-performance PDF processing with concurrency
- Batch evaluation for benchmarks
- NGram logits processor support

### 2. Transformers Inference
- Direct model loading with AutoModel
- Flash Attention 2 support
- Flexible image processing with crop mode
- Results saving and compression testing

## Project Structure
```
DeepSeek-OCR/
├── DeepSeek-OCR-master/
│   ├── DeepSeek-OCR-vllm/    # vLLM inference implementation
│   └── DeepSeek-OCR-hf/       # Hugging Face transformers implementation
├── assets/                     # Images and branding
├── requirements.txt            # Python dependencies
├── LICENSE                     # Project license
└── README.md                   # Detailed documentation
```

## Getting Started

### Installation
1. Clone the repository
2. Create conda environment with Python 3.12.9
3. Install PyTorch 2.6.0 with CUDA 11.8
4. Install vLLM 0.8.5
5. Install requirements and Flash Attention

### Basic Usage

**Document to Markdown:**
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```

**General OCR:**
```python
prompt = "<image>\n<|grounding|>OCR this image."
```

**Free OCR (without layouts):**
```python
prompt = "<image>\nFree OCR."
```

## Resources
- **Model Download**: [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Paper**: Available in repository as DeepSeek_OCR_paper.pdf
- **ArXiv**: [2510.18234](https://arxiv.org/abs/2510.18234)
- **Official Website**: [deepseek.com](https://www.deepseek.com/)

## Recent Updates
- **2025/10/23**: Official support in upstream vLLM
- **2025/10/20**: Initial release of DeepSeek-OCR

## Acknowledgments
The project builds upon and acknowledges contributions from:
- Vary
- GOT-OCR2.0
- MinerU
- PaddleOCR
- OneChart
- Slow Perception
- Fox (benchmark)
- OminiDocBench (benchmark)

## Citation
```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```

## Community
- **Discord**: [DeepSeek AI Community](https://discord.gg/Tc7c45Zzu5)
- **Twitter**: [@deepseek_ai](https://twitter.com/deepseek_ai)
