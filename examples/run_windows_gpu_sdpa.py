"""
DeepSeek-OCR Example for Windows 11 with GPU (SDPA Mode)
Recommended for RTX 1060 and other GPUs without flash-attention support
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

print("Loading model with SDPA attention...")
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='sdpa',  # Use Scaled Dot Product Attention (PyTorch native)
    trust_remote_code=True, 
    use_safetensors=True,
    torch_dtype=torch.float16,  # Use float16 for RTX 1060 compatibility
    low_cpu_mem_usage=True       # Reduce CPU memory during loading
)

print("Moving model to GPU...")
model = model.eval().cuda()

# Check GPU memory
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# OCR configuration
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'test.png'  # Change to your image path
output_path = 'output'   # Change to your output directory

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

print(f"\nProcessing image: {image_file}")
print("Using Small mode (640x640) for RTX 1060...")

# Run inference with settings optimized for RTX 1060 (6GB VRAM)
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=640,       # Small mode
    image_size=640,      # Small mode
    crop_mode=False,     # Disable cropping to save memory
    save_results=True, 
    test_compress=True
)

print("\nOCR completed!")
print(f"Results saved to: {output_path}")

# Print GPU memory usage after inference
if torch.cuda.is_available():
    print(f"\nFinal VRAM usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
