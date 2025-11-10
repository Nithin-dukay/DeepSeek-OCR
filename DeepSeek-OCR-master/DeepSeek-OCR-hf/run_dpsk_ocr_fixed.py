"""
DeepSeek-OCR inference script with transformers compatibility fix.

This script includes a workaround for the LlamaFlashAttention2 import error
that occurs with transformers>=4.47.0

For more details, see: ISSUE_7_FIX.md
"""
import os
import sys

# Add parent directory to path to import the compatibility fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Apply compatibility patch if needed
try:
    from fix_transformers_compatibility import apply_transformers_compatibility_patch
    apply_transformers_compatibility_patch()
except ImportError:
    print("Warning: Could not import compatibility patch. Make sure you're using transformers==4.46.3")

# Now import the required libraries
from transformers import AutoModel, AutoTokenizer
import torch

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Model configuration
model_name = 'deepseek-ai/DeepSeek-OCR'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
try:
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    print("✓ Model loaded with FlashAttention2")
except Exception as e:
    print(f"Warning: Could not load with FlashAttention2: {e}")
    print("Falling back to eager attention...")
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='eager',
        trust_remote_code=True,
        use_safetensors=True
    )
    print("✓ Model loaded with eager attention")

model = model.eval().cuda().to(torch.bfloat16)

# Inference configuration
# prompt = "<image>\\nFree OCR. "
prompt = "<image>\\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

# Model size configurations:
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

print("Running inference...")
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

print("✓ Inference completed successfully!")
