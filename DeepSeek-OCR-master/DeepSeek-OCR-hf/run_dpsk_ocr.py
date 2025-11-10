from transformers import AutoModel, AutoTokenizer
import torch
import os
from ngram_norepeat_hf import NoRepeatNGramLogitsProcessor, get_recommended_params


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


model_name = 'deepseek-ai/DeepSeek-OCR'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)



# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

# Configure hallucination prevention
# Choose document type: "handwritten", "printed", "table", or "general"
document_type = "handwritten"  # Change this based on your document type
params = get_recommended_params(document_type)

# Create logits processor to prevent hallucinations
logits_processor = NoRepeatNGramLogitsProcessor(
    ngram_size=params["ngram_size"],
    window_size=params["window_size"],
    whitelist_token_ids=params["whitelist_token_ids"]
)



# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=1024, 
    image_size=640, 
    crop_mode=True, 
    save_results=True, 
    test_compress=True,
    logits_processor=logits_processor  # Add this to prevent hallucinations
)
