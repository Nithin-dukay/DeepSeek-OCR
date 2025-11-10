"""
DeepSeek-OCR Inference Script (Transformers)

This script demonstrates how to use DeepSeek-OCR with Transformers
while avoiding common warnings.
"""

from transformers import AutoTokenizer
import torch
import os
import warnings

# Import the custom model wrapper
from modeling_deepseek_ocr import DeepSeekOCRForCausalLM

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Model configuration
model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
# Use the custom wrapper to load the model without warnings
model = DeepSeekOCRForCausalLM.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',  # Use flash attention if available
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    device_map='auto'  # Automatically handle device placement
)

# Set model to evaluation mode
model = model.eval()

# If device_map is not used, manually move to CUDA
if not hasattr(model, 'hf_device_map'):
    model = model.cuda().to(torch.bfloat16)

print("Model loaded successfully!")

# Inference configuration
# prompt = "<image>\\nFree OCR. "
prompt = "<image>\\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

# Resolution modes:
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam (Dynamic): base_size = 1024, image_size = 640, crop_mode = True

print("Running inference...")
# Run inference with warning suppression built into the wrapper
res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)

print("Inference completed!")
print(f"Results: {res}")
