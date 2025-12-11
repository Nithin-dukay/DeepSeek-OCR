from transformers import AutoModel, AutoTokenizer
import torch
import os


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

image = Image.open(image_file).convert('RGB')

processor = DeepseekOCRProcessor(base_size=1024, image_size=640)

image_features = processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=True)

input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']

inputs = {'input_ids': input_ids, **image_features}

streamer = CustomStreamer(tokenizer)

generated = model.generate(**inputs, streamer=streamer, max_new_tokens=8192, do_sample=False, temperature=0.0)

res = streamer.generated_text

print('\n')

if save_results:
    print('='*15 + 'save results:' + '='*15)

    image_draw = image.copy()

    outputs = res

    with open(f'{output_path}/result_ori.mmd', 'w', encoding = 'utf-8') as afile:
        afile.write(outputs)

    matches_ref, matches_images, mathes_other = re_match(outputs)
    result = process_image_with_refs(image_draw, matches_ref)

    for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
        outputs = outputs.replace(a_match_image, f'![](images/{idx}.jpg)\n')

    for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
        outputs = outputs.replace(a_match_other, '').replace('\\\\coloneqq', ':=').replace('\\\\eqqcolon', '=:')

    with open(f'{output_path}/result.mmd', 'w', encoding = 'utf-8') as afile:
        afile.write(outputs)

    result.save(f'{output_path}/result_with_boxes.jpg')
