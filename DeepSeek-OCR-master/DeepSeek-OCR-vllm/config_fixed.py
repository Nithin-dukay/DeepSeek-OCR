# TODO: change modes
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MIN_CROPS= 2

# CRITICAL FIX: Reduced MAX_CROPS to prevent memory issues
# Original value was 6, reduced to 4 for better stability
# If you still encounter OOM errors, try reducing to 3
MAX_CROPS= 4  # max:9; Reduced from 6 to prevent CUDA memory issues

# CRITICAL FIX: Reduced MAX_CONCURRENCY for stability
# Original value was 100, reduced to 50 to prevent memory fragmentation
MAX_CONCURRENCY = 50  # Reduced from 100 for better memory management

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

PROMPT = '<image>\\n<|grounding|>Convert the document to markdown.'
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
