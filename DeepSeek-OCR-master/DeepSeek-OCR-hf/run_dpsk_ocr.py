from transformers import AutoModel, AutoTokenizer
import torch
import os

# Import repetition prevention
try:
    from ngram_norepeat import NoRepeatNGramLogitsProcessor
    REPETITION_PREVENTION_AVAILABLE = True
except ImportError:
    print("Warning: NoRepeatNGramLogitsProcessor not available. Repetition prevention disabled.")
    REPETITION_PREVENTION_AVAILABLE = False


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


model_name = 'deepseek-ai/DeepSeek-OCR'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)



# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

# Configure repetition prevention
# These parameters help prevent infinite loops like ". . . . . . . . . ."
# - ngram_size: Size of patterns to track (30 is good for general text)
# - window_size: How far back to look for repetitions (90 tokens)
# - whitelist_token_ids: Tokens allowed to repeat (e.g., table tags <td>, </td>)
# Token IDs: 128821 = <td>, 128822 = </td>
if REPETITION_PREVENTION_AVAILABLE:
    logits_processor = NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # Allow table tags to repeat
        min_ngram_size=2,
        max_consecutive_repeats=3
    )
    print("Repetition prevention enabled (ngram_size=30, window_size=90)")
else:
    logits_processor = None


# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

# Note: The model.infer() method is defined in the remote model code (trust_remote_code=True)
# If the model supports logits_processor parameter, pass it. Otherwise, this serves as documentation.
# Check the model's documentation or source code for the exact parameter name.
try:
    # Try passing logits_processor if supported
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
        logits_processor=logits_processor if REPETITION_PREVENTION_AVAILABLE else None
    )
except TypeError as e:
    # If logits_processor parameter not supported, fall back to default
    print(f"Note: logits_processor parameter not supported by model.infer(). Using default generation.")
    print(f"To enable repetition prevention, the model's infer() method needs to be updated.")
    res = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=image_file, 
        output_path=output_path, 
        base_size=1024, 
        image_size=640, 
        crop_mode=True, 
        save_results=True, 
        test_compress=True
    )
