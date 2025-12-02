from transformers import AutoModel, AutoTokenizer
import torch
import os

# CPU-only setup - no CUDA environment variables needed
model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model for CPU inference...")
# Use eager attention for CPU compatibility
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',  # Use eager instead of flash_attention_2
    trust_remote_code=True,
    use_safetensors=True
)

# CPU setup
device = torch.device("cpu")
model = model.eval().to(device).to(torch.float32)  # Use float32 for CPU

print("Model loaded successfully on CPU")

# Example usage
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'test.png'  # Make sure this file exists or change to your image path
output_path = 'output'   # Create this directory or change to your output path

# For CPU, use smaller sizes for better performance
# Tiny: base_size = 512, image_size = 512, crop_mode = False
print("Starting inference...")
try:
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_path,
        base_size=512,      # Smaller size for CPU
        image_size=512,     # Smaller size for CPU
        crop_mode=False,    # Disable crop mode for simpler processing
        save_results=True,
        test_compress=False  # Disable compression test for CPU
    )
    print("Inference completed successfully!")
    print(f"Results saved to: {output_path}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the image file exists and the output directory is created.")
except Exception as e:
    print(f"Inference failed: {e}")
    print("This might be due to missing dependencies or incompatible image format.")