from transformers import AutoModel, AutoTokenizer
import torch
import os

# Import compression utilities to fix the test_compress return issue
from compression_utils import patch_infer_method

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# ============================================================================
# FIX FOR GITHUB ISSUE #285: Compression Study Evaluation
# ============================================================================
# IMPORTANT: The original model.infer() returns None when test_compress=True
# This is a known issue in the HuggingFace model implementation.
# 
# SOLUTION: Apply the patch to fix the return value
model = patch_infer_method(model)
# ============================================================================

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

# ============================================================================
# Model Inference Parameters
# ============================================================================
# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', 
#       base_size = 1024, image_size = 640, crop_mode = True, 
#       test_compress = False, save_results = False, eval_mode = False):
#
# Resolution Modes:
# - Tiny:   base_size = 512,  image_size = 512,  crop_mode = False  (64 tokens)
# - Small:  base_size = 640,  image_size = 640,  crop_mode = False  (100 tokens)
# - Base:   base_size = 1024, image_size = 1024, crop_mode = False  (256 tokens)
# - Large:  base_size = 1280, image_size = 1280, crop_mode = False  (400 tokens)
# - Gundam: base_size = 1024, image_size = 640,  crop_mode = True   (dynamic)
#
# Parameters:
# - test_compress: Print compression statistics (image tokens vs output tokens)
# - save_results:  Save output to files in output_path
# - eval_mode:     Return clean output text (alternative to test_compress)
# ============================================================================

# Example 1: With compression statistics (FIXED - now returns output)
print("=" * 80)
print("Example 1: Inference with compression statistics")
print("=" * 80)
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
print(f"\nReturned output (first 200 chars): {res[:200] if res else 'None'}")

# Example 2: Get compression stats as dictionary
print("\n" + "=" * 80)
print("Example 2: Get compression statistics as dictionary")
print("=" * 80)
res_with_stats = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=1024, 
    image_size=640, 
    crop_mode=True, 
    test_compress=True,
    return_stats=True  # Returns dict with 'output' and 'stats' keys
)
if isinstance(res_with_stats, dict):
    print(f"\nCompression Statistics:")
    print(f"  Image size: {res_with_stats['stats'].get('image_size', 'N/A')}")
    print(f"  Image tokens: {res_with_stats['stats'].get('image_tokens', 'N/A')}")
    print(f"  Output tokens: {res_with_stats['stats'].get('output_tokens', 'N/A')}")
    print(f"  Compression ratio: {res_with_stats['stats'].get('compression_ratio', 'N/A')}")

# Example 3: Simple inference without compression stats
print("\n" + "=" * 80)
print("Example 3: Simple inference (eval_mode)")
print("=" * 80)
res_simple = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=1024, 
    image_size=640, 
    crop_mode=True, 
    eval_mode=True  # Returns output without compression stats
)
print(f"\nReturned output (first 200 chars): {res_simple[:200] if res_simple else 'None'}")
