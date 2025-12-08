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

# Repetition prevention settings (Fix for Issue #257)
# These settings help prevent the model from generating endless repetitive output
# especially for structured content like menus, tables, and lists
NGRAM_SIZE = 40  # Increased from 30 to catch longer repetitive patterns
WINDOW_SIZE = 90  # Window to search for repeated n-grams
MAX_CONSECUTIVE_REPEATS = 3  # Maximum allowed consecutive similar patterns
REPETITION_PENALTY_SCALE = 2.0  # Penalty multiplier for repeated patterns
PATTERN_DETECTION_SIZE = 20  # Size of pattern to detect for repetition

# Early stopping settings for repetition detection
ENABLE_EARLY_STOPPING = True  # Enable early stopping when repetition is detected
MAX_CONSECUTIVE_SIMILAR_LINES = 5  # Stop if more than this many similar lines in a row
SIMILARITY_THRESHOLD = 0.7  # Threshold for considering lines similar (0-1)
MAX_NUMBER_SEQUENCE_LENGTH = 20  # Stop if numbers increment for more than this many steps

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
