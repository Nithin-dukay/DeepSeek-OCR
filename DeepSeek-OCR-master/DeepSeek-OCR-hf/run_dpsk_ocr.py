from transformers import AutoModel, AutoTokenizer, GenerationConfig
import torch
import os


# Set CUDA device and check availability
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. The model will run on CPU, which may be slow.")
else:
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")


model_name = 'deepseek-ai/DeepSeek-OCR'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model with proper device mapping and attention implementation
try:
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    print("Model loaded successfully with Flash Attention 2.0")
except Exception as e:
    print(f"Failed to load with Flash Attention 2.0: {e}")
    print("Falling back to standard attention...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )

model = model.eval()

# Fix generation config to suppress warnings
model.config.pad_token_id = model.config.eos_token_id
model.generation_config = GenerationConfig(
    pad_token_id=model.config.eos_token_id,
    do_sample=False,
    temperature=1.0,  # Set to 1.0 to avoid warning when do_sample=False
)



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
