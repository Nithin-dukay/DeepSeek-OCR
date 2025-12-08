# DeepSeek-OCR Mixed Dataset Training Guide

## Overview

This guide explains how to fine-tune DeepSeek-OCR models with mixed text and multimodal data using the ms-swift framework. This addresses GitHub Issue #292.

## Problem Statement

When fine-tuning DeepSeek-OCR models using ms-swift, training with datasets containing both text-only and multimodal (text + image) samples would fail because the model expected image inputs for all samples.

## Solution

The DeepSeek-OCR model has been updated to gracefully handle both text-only and multimodal inputs in the same training batch. The key changes include:

1. **Improved input validation** - The model now properly detects and handles missing or empty image inputs
2. **Conditional vision processing** - Vision encoders are only invoked when valid images are present
3. **Seamless fallback** - Text-only samples are processed through the language model without vision features

## Dataset Format

### Text-Only Sample

```json
{
  "messages": [
    {"role": "user", "content": "What is optical character recognition?"},
    {"role": "assistant", "content": "OCR is a technology that converts different types of documents..."}
  ]
}
```

### Multimodal Sample (Image + Text)

```json
{
  "messages": [
    {"role": "user", "content": "<image>\n<|grounding|>Convert the document to markdown."},
    {"role": "assistant", "content": "# Document Title\n\nThis is the extracted text..."}
  ],
  "images": ["path/to/document.jpg"]
}
```

### Mixed Dataset Example (JSONL format)

```jsonl
{"messages": [{"role": "user", "content": "Explain OCR technology."}, {"role": "assistant", "content": "OCR stands for..."}]}
{"messages": [{"role": "user", "content": "<image>\nFree OCR."}, {"role": "assistant", "content": "The text in the image says..."}], "images": ["image1.jpg"]}
{"messages": [{"role": "user", "content": "What are the benefits of OCR?"}, {"role": "assistant", "content": "OCR provides..."}]}
{"messages": [{"role": "user", "content": "<image>\n<|grounding|>Parse the figure."}, {"role": "assistant", "content": "This figure shows..."}], "images": ["chart.png"]}
```

## Training with ms-swift

### Installation

```bash
# Install ms-swift
pip install ms-swift -U

# Or install from source
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

### Basic Training Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
  --model deepseek-ai/DeepSeek-OCR \
  --dataset your_mixed_dataset.jsonl \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --gradient_accumulation_steps 8 \
  --max_length 8192 \
  --output_dir output/deepseek-ocr-mixed
```

### Advanced Training Configuration

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
swift sft \
  --model deepseek-ai/DeepSeek-OCR \
  --dataset your_mixed_dataset.jsonl \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 3 \
  --learning_rate 2e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner false \
  --gradient_accumulation_steps 16 \
  --max_length 8192 \
  --warmup_ratio 0.05 \
  --save_strategy epoch \
  --save_total_limit 3 \
  --logging_steps 10 \
  --output_dir output/deepseek-ocr-mixed-advanced
```

## Key Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--model` | Model identifier | `deepseek-ai/DeepSeek-OCR` |
| `--dataset` | Path to your mixed dataset | `your_dataset.jsonl` |
| `--train_type` | Training method | `lora` (efficient) or `full` |
| `--torch_dtype` | Data type for training | `bfloat16` |
| `--freeze_vit` | Freeze vision encoders | `true` (recommended) |
| `--freeze_aligner` | Freeze projector | `true` or `false` |
| `--max_length` | Maximum sequence length | `8192` |
| `--per_device_train_batch_size` | Batch size per GPU | `2-4` (adjust based on VRAM) |

## Supported Prompts

### Document OCR
```
<image>\n<|grounding|>Convert the document to markdown.
```

### General OCR
```
<image>\n<|grounding|>OCR this image.
```

### Free OCR (without layouts)
```
<image>\nFree OCR.
```

### Figure Parsing
```
<image>\nParse the figure.
```

### General Description
```
<image>\nDescribe this image in detail.
```

### Text Localization
```
<image>\nLocate <|ref|>specific text<|/ref|> in the image.
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use smaller `max_length`
- Enable `freeze_vit` and `freeze_aligner`

### Issue: Slow Training Speed

**Solution:**
- Increase batch size if memory allows
- Use multiple GPUs with `NPROC_PER_NODE`
- Enable gradient checkpointing
- Consider using `train_type=lora` instead of full fine-tuning

### Issue: Model Not Learning from Text-Only Samples

**Solution:**
- Ensure text-only samples don't include `<image>` token
- Balance the ratio of text-only to multimodal samples
- Adjust learning rate for better convergence

### Issue: Vision Features Not Being Used

**Solution:**
- Don't freeze both `vit` and `aligner` simultaneously
- Ensure multimodal samples include the `<image>` token
- Verify image paths are correct in the dataset

## Performance Tips

1. **Data Mixing Ratio**: Use 80-90% multimodal data and 10-20% text-only data for best results
2. **Batch Composition**: ms-swift automatically handles mixed batches
3. **Vision Encoder Freezing**: Freeze vision encoders to save memory and speed up training
4. **LoRA Configuration**: Use rank 8-16 for efficient fine-tuning
5. **Gradient Accumulation**: Use larger accumulation steps for effective larger batch sizes

## Validation

After training, test your model with both text-only and multimodal inputs:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_path = "output/deepseek-ocr-mixed/checkpoint-best"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)

# Test text-only
text_prompt = "What is OCR?"
text_response = model.generate(tokenizer, prompt=text_prompt)

# Test multimodal
image_prompt = "<image>\nFree OCR."
image_response = model.infer(tokenizer, prompt=image_prompt, image_file="test.jpg")
```

## References

- [ms-swift GitHub Repository](https://github.com/modelscope/ms-swift)
- [DeepSeek-OCR Model Card](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [ms-swift Documentation](https://swift.readthedocs.io/)

## Support

For issues or questions:
- DeepSeek-OCR Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- ms-swift Issues: https://github.com/modelscope/ms-swift/issues
