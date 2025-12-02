"""
DeepSeek-OCR CPU-Only Inference Script

This script demonstrates how to run DeepSeek-OCR on CPU without flash-attention.
Suitable for systems without CUDA support or with incompatible GPUs.

Note: CPU inference will be significantly slower than GPU inference.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os

# Disable CUDA to force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Model configuration
model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model for CPU inference...")
print("Note: This may take a few minutes and will use significant RAM.")

# Load model WITHOUT flash_attention_2 (use default eager attention)
# Do NOT specify _attn_implementation parameter to use the default eager mode
model = AutoModel.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    use_safetensors=True,
    torch_dtype=torch.float32  # Use float32 for CPU (bfloat16 is not well supported on CPU)
)

# Set model to evaluation mode and keep on CPU
model = model.eval()

print(f"Model loaded successfully on device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Configure your inference
# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'test.png'  # Change this to your image path
output_path = 'output'   # Change this to your output directory

# Inference parameters for different model sizes:
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

print("\nStarting inference...")
print("WARNING: CPU inference is slow. This may take several minutes depending on image size.")

# For CPU, it's recommended to use smaller image sizes to reduce processing time
# Start with Tiny or Small configuration for testing
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=512,      # Using Tiny size for faster CPU processing
    image_size=512,     # Using Tiny size for faster CPU processing
    crop_mode=False,    # Disable cropping for simpler processing
    save_results=True, 
    test_compress=True
)

print("\nInference completed!")
print(f"Results saved to: {output_path}")
