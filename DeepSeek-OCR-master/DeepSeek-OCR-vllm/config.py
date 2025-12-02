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

# Prompt validation settings
ENABLE_PROMPT_VALIDATION = True  # Set to False to disable validation warnings
STRICT_PROMPT_MODE = False  # Set to True to reject invalid prompts

# Default prompt - using recommended format
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# RECOMMENDED PROMPTS - These formats are tested and work reliably:
# ============================================================================
# Document to Markdown: <image>\n<|grounding|>Convert the document to markdown.
# General OCR:          <image>\n<|grounding|>OCR this image.
# Free OCR (no layout): <image>\nFree OCR.
# Figure/Chart parsing: <image>\nParse the figure.
# General description:  <image>\nDescribe this image in detail.
# Text location:        <image>\nLocate <|ref|>TEXT<|/ref|> in the image.
# ============================================================================
#
# ⚠️  WARNING: Modifying prompts can cause repetitive or incorrect outputs!
#
# SAFE modifications (append to base prompt):
#   - Output format: "... in JSON format", "... as plain text"
#   - Focus area: "... focus on tables", "... extract text only"
#
# UNSAFE modifications (avoid these):
#   - Negative instructions: "don't add spaces", "without extra lines"
#   - Meta-instructions: "make sure to...", "always...", "never..."
#   - Multiple sentences with complex constraints
#
# If you need custom behavior, start with a recommended prompt and make
# minimal, positive additions. See PROMPT_GUIDELINES.md for details.


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
