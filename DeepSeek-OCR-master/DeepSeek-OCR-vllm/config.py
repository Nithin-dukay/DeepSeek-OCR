# ============================================================================
# MODE CONFIGURATION
# ============================================================================
# Select one of the following modes: 'tiny', 'small', 'base', 'large', 'gundam'
# 
# Mode descriptions:
# - Tiny:   512×512 resolution (64 vision tokens)   - Fastest, lowest memory
# - Small:  640×640 resolution (100 vision tokens)  - Fast, low memory
# - Base:   1024×1024 resolution (256 vision tokens) - Balanced
# - Large:  1280×1280 resolution (400 vision tokens) - High quality, more memory
# - Gundam: Dynamic resolution with tiles (n×640×640 + 1×1024×1024) - Best quality, adaptive
#
# You can either:
# 1. Set MODE to automatically configure BASE_SIZE, IMAGE_SIZE, and CROP_MODE
# 2. Manually set BASE_SIZE, IMAGE_SIZE, and CROP_MODE (advanced users)
# ============================================================================

MODE = 'gundam'  # Options: 'tiny', 'small', 'base', 'large', 'gundam'

# Mode configurations mapping
MODE_CONFIGS = {
    'tiny': {
        'base_size': 512,
        'image_size': 512,
        'crop_mode': False,
        'description': '512×512 resolution (64 vision tokens) - Fastest, lowest memory'
    },
    'small': {
        'base_size': 640,
        'image_size': 640,
        'crop_mode': False,
        'description': '640×640 resolution (100 vision tokens) - Fast, low memory'
    },
    'base': {
        'base_size': 1024,
        'image_size': 1024,
        'crop_mode': False,
        'description': '1024×1024 resolution (256 vision tokens) - Balanced'
    },
    'large': {
        'base_size': 1280,
        'image_size': 1280,
        'crop_mode': False,
        'description': '1280×1280 resolution (400 vision tokens) - High quality, more memory'
    },
    'gundam': {
        'base_size': 1024,
        'image_size': 640,
        'crop_mode': True,
        'description': 'Dynamic resolution with tiles (n×640×640 + 1×1024×1024) - Best quality, adaptive'
    }
}

# Validate and apply mode configuration
if MODE.lower() not in MODE_CONFIGS:
    raise ValueError(f"Invalid MODE '{MODE}'. Must be one of: {list(MODE_CONFIGS.keys())}")

_mode_config = MODE_CONFIGS[MODE.lower()]
BASE_SIZE = _mode_config['base_size']
IMAGE_SIZE = _mode_config['image_size']
CROP_MODE = _mode_config['crop_mode']

# Advanced users: You can manually override these values after mode selection
# BASE_SIZE = 1024
# IMAGE_SIZE = 640
# CROP_MODE = True
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
