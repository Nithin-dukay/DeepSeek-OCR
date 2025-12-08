"""
DeepSeek-OCR Example for Windows 11 with CPU Only
For testing without GPU or when GPU has insufficient memory
WARNING: This will be very slow!
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
import time

# Model configuration
model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model for CPU inference...")
print("WARNING: This may take several minutes and inference will be slow!")

model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='eager',  # Eager mode for CPU
    trust_remote_code=True, 
    use_safetensors=True,
    torch_dtype=torch.float32,    # CPU works better with float32
    low_cpu_mem_usage=True
)

print("Model loaded on CPU")
model = model.eval()  # No .cuda() call for CPU mode

# OCR configuration
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'test.png'  # Change to your image path
output_path = 'output'   # Change to your output directory

# Create output directory
os.makedirs(output_path, exist_ok=True)

print(f"\nProcessing image: {image_file}")
print("Using Tiny mode (512x512) for CPU...")
print("This may take several minutes...")

start_time = time.time()

# Run inference with smallest settings for CPU
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=512,       # Tiny mode
    image_size=512,      # Tiny mode
    crop_mode=False,     # Disable cropping
    save_results=True, 
    test_compress=False  # Disable compression for speed
)

elapsed_time = time.time() - start_time

print(f"\nOCR completed in {elapsed_time:.2f} seconds!")
print(f"Results saved to: {output_path}")
