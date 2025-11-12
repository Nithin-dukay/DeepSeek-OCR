import os
import sys

# Version check for transformers - do this BEFORE importing torch to catch issues early
try:
    import transformers
    from packaging import version
    
    transformers_version = version.parse(transformers.__version__)
    required_version = version.parse("4.46.3")
    max_version = version.parse("4.47.0")
    
    if transformers_version != required_version:
        print("=" * 70)
        print("⚠️  WARNING: Incompatible transformers version detected!")
        print("=" * 70)
        print(f"Installed version: {transformers.__version__}")
        print(f"Required version:  4.46.3")
        print()
        
        if transformers_version >= max_version:
            print("❌ ERROR: transformers 4.47+ has breaking changes!")
            print("   The LlamaFlashAttention2 import is not available in this version.")
            print()
            print("To fix this issue:")
            print("  pip install transformers==4.46.3 --force-reinstall")
            print()
            print("For Colab users:")
            print("  !pip install transformers==4.46.3 --force-reinstall")
            print("  Then: Runtime -> Restart runtime")
            print()
            print("See: https://github.com/deepseek-ai/DeepSeek-OCR/issues/7")
            print("=" * 70)
            sys.exit(1)
        else:
            print("⚠️  You may encounter compatibility issues.")
            print("   Recommended: pip install transformers==4.46.3")
            print("=" * 70)
            print()
except ImportError:
    print("Warning: Could not verify transformers version (packaging not installed)")
    print("If you encounter import errors, ensure transformers==4.46.3 is installed")
    print()

from transformers import AutoModel, AutoTokenizer
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


model_name = 'deepseek-ai/DeepSeek-OCR'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)



# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'



# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
