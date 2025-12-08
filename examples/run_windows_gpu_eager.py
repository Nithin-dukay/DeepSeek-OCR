"""
DeepSeek-OCR Example for Windows 11 with GPU (Eager Mode)
Most compatible mode, works on all GPUs
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Model configuration
model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model with eager attention...")
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='eager',  # Use eager attention (most compatible)
    trust_remote_code=True, 
    use_safetensors=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("Moving model to GPU...")
model = model.eval().cuda()

# Check GPU info
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# OCR configuration
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'test.png'  # Change to your image path
output_path = 'output'   # Change to your output directory

# Create output directory
os.makedirs(output_path, exist_ok=True)

print(f"\nProcessing image: {image_file}")
print("Using Tiny mode (512x512) for maximum compatibility...")

# Run inference with smallest settings for compatibility
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=512,       # Tiny mode
    image_size=512,      # Tiny mode
    crop_mode=False,     # Disable cropping
    save_results=True, 
    test_compress=True
)

print("\nOCR completed!")
print(f"Results saved to: {output_path}")
