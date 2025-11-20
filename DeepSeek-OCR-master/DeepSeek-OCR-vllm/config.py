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
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers 
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR' # change to your model path

# ============================================================================
# REPETITION PREVENTION SETTINGS
# ============================================================================
# These settings help prevent infinite loops and excessive repetition
# (e.g., ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .")
# See: https://github.com/deepseek-ai/DeepSeek-OCR/issues/250

# N-gram size for repetition detection
# - Larger values catch longer repetitive patterns
# - Recommended: 20-40 for document OCR
# - Image inference: 30, PDF inference: 20, Batch eval: 40
NGRAM_SIZE = 30

# Window size for n-gram search
# - How many tokens back to search for repeated n-grams
# - Larger values use more memory but catch distant repetitions
# - Recommended: 50-100
WINDOW_SIZE = 90

# Whitelist token IDs (tokens allowed to repeat)
# - Useful for structural elements like table tags
# - 128821: <td> (table cell start)
# - 128822: </td> (table cell end)
# - Add more token IDs as needed for your use case
WHITELIST_TOKEN_IDS = {128821, 128822}

# Minimum n-gram size for short pattern detection
# - Helps catch patterns like ". . . ."
# - Recommended: 2-3
MIN_NGRAM_SIZE = 2

# Maximum consecutive identical tokens allowed
# - Prevents immediate repetition like "the the the the"
# - Recommended: 3-5
MAX_CONSECUTIVE_REPEATS = 3

# Enable repetition detection and early stopping
# - Set to False to disable all repetition prevention
ENABLE_REPETITION_PREVENTION = True

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
