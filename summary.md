# DeepSeek-OCR Project Summary

## Overview
DeepSeek-OCR is an advanced optical character recognition (OCR) model that investigates the role of vision encoders from an LLM-centric viewpoint. The project focuses on **Contexts Optical Compression**, exploring the boundaries of visual-text compression for document understanding and text extraction.

## Key Features
- **Multiple Resolution Modes**: Supports native and dynamic resolution processing
  - Native: Tiny (512×512), Small (640×640), Base (1024×1024), Large (1280×1280)
  - Dynamic: Gundam mode (n×640×640 + 1×1024×1024)
- **Flexible Inference**: Supports both vLLM and Transformers backends
- **High Performance**: Achieves ~2500 tokens/s concurrency on A100-40G GPU
- **Versatile OCR Tasks**: Document conversion, image OCR, figure parsing, and text localization

## Project Structure
```
/vercel/sandbox/
├── DeepSeek_OCR_paper.pdf          # Research paper
├── README.md                        # Main documentation
├── requirements.txt                 # Python dependencies
├── LICENSE                          # License file
├── assets/                          # Visual assets and examples
│   ├── badge.svg, logo.svg
│   ├── fig1.png
│   └── show1-4.jpg                 # Visualization examples
└── DeepSeek-OCR-master/
    ├── DeepSeek-OCR-hf/            # Hugging Face Transformers implementation
    │   └── run_dpsk_ocr.py
    └── DeepSeek-OCR-vllm/          # vLLM implementation
        ├── config.py                # Configuration settings
        ├── deepseek_ocr.py
        ├── run_dpsk_ocr_eval_batch.py
        ├── run_dpsk_ocr_image.py
        └── run_dpsk_ocr_pdf.py
```

## Technical Stack
- **Framework**: PyTorch 2.6.0 with CUDA 11.8
- **Inference Engines**: vLLM 0.8.5 and Transformers 4.46.3
- **Model**: deepseek-ai/DeepSeek-OCR (Hugging Face)
- **Attention**: Flash Attention 2.7.3
- **Python Version**: 3.12.9

## Dependencies
Core Python packages:
- `transformers==4.46.3` - Model loading and inference
- `tokenizers==0.20.3` - Text tokenization
- `torch==2.6.0` - Deep learning framework
- `vllm-0.8.5+cu118` - High-performance inference
- `flash-attn==2.7.3` - Optimized attention mechanism
- `PyMuPDF`, `img2pdf` - PDF processing
- `Pillow`, `numpy` - Image processing
- `einops`, `easydict`, `addict` - Utilities

## Installation
1. **Clone Repository**:
   ```bash
   git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
   ```

2. **Create Environment**:
   ```bash
   conda create -n deepseek-ocr python=3.12.9 -y
   conda activate deepseek-ocr
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
   pip install -r requirements.txt
   pip install flash-attn==2.7.3 --no-build-isolation
   ```

## Usage

### vLLM Inference (Recommended for Production)
Configure `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` with your paths and settings.

**Image Processing** (streaming output):
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

**PDF Processing** (high concurrency):
```bash
python run_dpsk_ocr_pdf.py
```

**Batch Evaluation**:
```bash
python run_dpsk_ocr_eval_batch.py
```

### Transformers Inference
```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, 
                                   _attn_implementation='flash_attention_2',
                                   trust_remote_code=True,
                                   use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\\n<|grounding|>Convert the document to markdown."
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

res = model.infer(tokenizer, 
                  prompt=prompt,
                  image_file=image_file,
                  output_path=output_path,
                  base_size=1024,
                  image_size=640,
                  crop_mode=True,
                  save_results=True,
                  test_compress=True)
```

## Prompt Examples
- **Document to Markdown**: `<image>\\n<|grounding|>Convert the document to markdown.`
- **General OCR**: `<image>\\n<|grounding|>OCR this image.`
- **Free OCR (no layout)**: `<image>\\nFree OCR.`
- **Figure Parsing**: `<image>\\nParse the figure.`
- **Image Description**: `<image>\\nDescribe this image in detail.`
- **Text Localization**: `<image>\\nLocate <|ref|>text<|/ref|> in the image.`

## Configuration Modes
Adjust `base_size`, `image_size`, and `crop_mode` in config.py:
- **Tiny**: 512, 512, False (64 vision tokens)
- **Small**: 640, 640, False (100 vision tokens)
- **Base**: 1024, 1024, False (256 vision tokens)
- **Large**: 1280, 1280, False (400 vision tokens)
- **Gundam**: 1024, 640, True (dynamic resolution)

## Performance Characteristics
- **Concurrency**: ~2500 tokens/s on A100-40G GPU
- **Max Concurrency**: 100 (configurable based on GPU memory)
- **Worker Threads**: 64 for image preprocessing
- **Max Crops**: 6 (up to 9 for high-memory GPUs)

## Recent Updates
- **2025/10/23**: Official vLLM upstream support added
- **2025/10/20**: Initial release of DeepSeek-OCR model

## Resources
- **Model**: [Hugging Face - deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Paper**: [arXiv:2510.18234](https://arxiv.org/abs/2510.18234)
- **GitHub**: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- **Discord**: [DeepSeek AI Community](https://discord.gg/Tc7c45Zzu5)
- **Twitter**: [@deepseek_ai](https://twitter.com/deepseek_ai)

## Acknowledgements
Built upon research and tools from:
- Vary, GOT-OCR2.0, MinerU, PaddleOCR
- OneChart, Slow Perception
- Benchmarks: Fox, OminiDocBench

## Citation
```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```

## License
See LICENSE file for details.
