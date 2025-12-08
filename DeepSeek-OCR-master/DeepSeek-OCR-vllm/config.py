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

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = '' 
OUTPUT_PATH = ''

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'

# ============================================================================
# PROMPT FORMATTING GUIDELINES
# ============================================================================
# The model is sensitive to prompt formatting. For best results:
#
# 1. KEEP PROMPTS SIMPLE AND DIRECT
#    - Use concise, single-sentence instructions
#    - Avoid complex clauses or multiple instructions
#    - Avoid negative instructions (e.g., "don't add extra space")
#
# 2. USE RECOMMENDED PROMPT PATTERNS
#    The following prompts are tested and optimized:
#
#    Document OCR (with layout):
#      '<image>\n<|grounding|>Convert the document to markdown.'
#
#    Image OCR (with layout):
#      '<image>\n<|grounding|>OCR this image.'
#
#    Free OCR (without layout):
#      '<image>\nFree OCR.'
#
#    Figure/Chart parsing:
#      '<image>\nParse the figure.'
#
#    General description:
#      '<image>\nDescribe this image in detail.'
#
#    Recognition/Localization:
#      '<image>\nLocate <|ref|>text_to_find<|/ref|> in the image.'
#
# 3. SPECIAL TOKENS
#    - <image>: Required at the start of every prompt
#    - <|grounding|>: Use for layout-aware OCR tasks
#    - <|ref|>...<|/ref|>: Use for text localization tasks
#
# 4. AVOID PROBLEMATIC PATTERNS
#    - Multiple comma-separated clauses
#    - Negative instructions ("don't", "without", "no extra")
#    - Overly specific formatting requirements
#    - Instructions longer than 2-3 sentences
#
# 5. PROMPT VALIDATION
#    Enable automatic prompt validation by setting VALIDATE_PROMPTS = True
#    This will check and sanitize prompts before inference.
# ============================================================================

# Enable automatic prompt validation and sanitization
VALIDATE_PROMPTS = True

# N-gram repetition prevention settings
# These settings help prevent the model from generating repetitive output
NGRAM_SIZE = 30          # Size of n-grams to check (smaller = stricter)
NGRAM_WINDOW = 90        # Window size for checking repetitions
NGRAM_ADAPTIVE = True    # Enable adaptive parameter adjustment
NGRAM_REPETITION_THRESHOLD = 3  # Max times a pattern can repeat

# Whitelist token IDs that are allowed to repeat
# 128821 and 128822 are special tokens for the model
NGRAM_WHITELIST = {128821, 128822}


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
