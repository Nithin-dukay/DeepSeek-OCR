# TODO: change modes
# You can either set MODE to one of the predefined modes, or manually set BASE_SIZE, IMAGE_SIZE, and CROP_MODE
# Available modes: Tiny, Small, Base, Large, Gundam
# 
# Tiny: base_size = 512, image_size = 512, crop_mode = False (64 vision tokens)
# Small: base_size = 640, image_size = 640, crop_mode = False (100 vision tokens)
# Base: base_size = 1024, image_size = 1024, crop_mode = False (256 vision tokens)
# Large: base_size = 1280, image_size = 1280, crop_mode = False (400 vision tokens)
# Gundam: base_size = 1024, image_size = 640, crop_mode = True (dynamic resolution)

# Option 1: Use predefined mode (recommended)
MODE = 'Gundam'  # Change this to: Tiny, Small, Base, Large, or Gundam

# Option 2: Manual configuration (overrides MODE if set)
# Uncomment and set these to override MODE setting:
# BASE_SIZE = 1024
# IMAGE_SIZE = 640
# CROP_MODE = True

# Auto-configure based on MODE if manual settings are not provided
try:
    from modes import get_mode_config
    if 'BASE_SIZE' not in locals() or 'IMAGE_SIZE' not in locals() or 'CROP_MODE' not in locals():
        BASE_SIZE, IMAGE_SIZE, CROP_MODE = get_mode_config(MODE)
except ImportError:
    # Fallback to default Gundam mode if modes.py is not available
    if 'BASE_SIZE' not in locals():
        BASE_SIZE = 1024
    if 'IMAGE_SIZE' not in locals():
        IMAGE_SIZE = 640
    if 'CROP_MODE' not in locals():
        CROP_MODE = True
MIN_CROPS= 2
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers 
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR' # change to your model path

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = '' 
OUTPUT_PATH = ''

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'
# TODO commonly used prompts
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
