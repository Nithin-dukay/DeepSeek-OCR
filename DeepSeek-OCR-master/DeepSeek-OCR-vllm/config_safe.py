"""
Safe configuration for DeepSeek OCR to avoid Triton/CUDA errors (Issue #299)

This configuration uses more conservative settings to prevent:
- Illegal memory access in Triton kernels
- CUDA out-of-memory errors
- MoE kernel overflow issues

Key changes from default config.py:
1. Reduced MAX_CROPS from 6 to 4 (less memory pressure)
2. Reduced MAX_CONCURRENCY from 100 to 50 (safer batch processing)
3. Added comments explaining each parameter
"""

# TODO: change modes
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

# For problematic images (mixed handwritten/digital text), consider:
# - Using smaller sizes (640x640 instead of 1024x1024)
# - Disabling CROP_MODE (set to False)
# - Reducing MAX_CROPS to 2-3

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS = 2

# IMPORTANT: Reduced from 6 to 4 to prevent MoE kernel overflow
# If you still encounter errors, try reducing to 2 or 3
# Max value is 9, but higher values increase risk of Triton errors
MAX_CROPS = 4  # Reduced for stability (was 6)

# IMPORTANT: Reduced from 100 to 50 for safer batch processing
# Lower values reduce memory pressure on GPU and MoE kernels
# If processing many images, consider reducing further to 20-30
MAX_CONCURRENCY = 50  # Reduced for stability (was 100)

# Number of CPU workers for image preprocessing
# This doesn't affect GPU memory, safe to keep high
NUM_WORKERS = 64

# Print number of vision tokens (useful for debugging)
PRINT_NUM_VIS_TOKENS = False

# Skip repeated processing
SKIP_REPEAT = True

# Model path - change to your local path if needed
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = '' 
OUTPUT_PATH = ''

# Default prompt for document OCR
PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'

# Alternative prompts:
# PROMPT = '<image>\\nFree OCR.'
# TODO commonly used prompts
# document: <image>\\n<|grounding|>Convert the document to markdown.
# other image: <image>\\n<|grounding|>OCR this image.
# without layouts: <image>\\nFree OCR.
# figures in document: <image>\\nParse the figure.
# general: <image>\\nDescribe this image in detail.
# rec: <image>\\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......

from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Additional safety parameters (can be used in scripts)
SAFE_MODE_PARAMS = {
    'enforce_eager': True,  # Disable CUDA graphs (workaround for Triton bug)
    'gpu_memory_utilization': 0.75,  # Reduced from default 0.9
    'max_model_len': 8192,  # Can be reduced to 4096 if needed
    'block_size': 256,  # Memory alignment
    'enable_prefix_caching': False,  # Disable to reduce memory pressure
}

# Troubleshooting guide
TROUBLESHOOTING = """
If you encounter "illegal memory access" or Triton errors:

1. Use run_dpsk_ocr_image_fixed.py instead of run_dpsk_ocr_image.py
2. Reduce MAX_CROPS to 2 or 3
3. Reduce MAX_CONCURRENCY to 20-30
4. Try smaller image sizes (BASE_SIZE=640, IMAGE_SIZE=640)
5. Disable CROP_MODE (set to False)
6. Update to vLLM nightly: pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
7. Fall back to HuggingFace Transformers: cd ../DeepSeek-OCR-hf && python run_dpsk_ocr.py

For more details, see ISSUE_299_FIX.md
"""
