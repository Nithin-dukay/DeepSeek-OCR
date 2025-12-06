import os
import re
from tqdm import tqdm
import torch
import argparse
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, MAX_CONCURRENCY, CROP_MODE, NUM_WORKERS
from concurrent.futures import ThreadPoolExecutor
import glob
from PIL import Image
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from modes import get_mode_config, get_available_modes
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True, 
    max_model_len=8192,
    swap_space=0,
    max_num_seqs = MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=40, window_size=90, whitelist_token_ids= {128821, 128822})] #window for fast；whitelist_token_ids: <td>,</td>

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m' 

def clean_formula(text):

    formula_pattern = r'\\\[(.*?)\\\]'
    
    def process_formula(match):
        formula = match.group(1)

        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        
        formula = formula.strip()
        
        return r'\[' + formula + r'\]'

    cleaned_text = re.sub(formula_pattern, process_formula, text)
    
    return cleaned_text

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    # mathes_image = []
    mathes_other = []
    for a_match in matches:
        mathes_other.append(a_match[0])
    return matches, mathes_other

def process_single_image(image, crop_mode, image_size, base_size):
    """single image"""
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(
            images=[image], bos=True, eos=True, cropping=crop_mode,
            image_size=image_size, base_size=base_size
        )},
    }
    return cache_item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSeek-OCR Batch Evaluation with vLLM')
    parser.add_argument('--mode', type=str, default=None, 
                        choices=get_available_modes(),
                        help=f'OCR mode to use. Available: {", ".join(get_available_modes())}. If not specified, uses config.py settings.')
    parser.add_argument('--input', type=str, default=None,
                        help='Input directory path with images (overrides config.py INPUT_PATH)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory path (overrides config.py OUTPUT_PATH)')
    
    args = parser.parse_args()
    
    # Get mode configuration
    if args.mode:
        base_size, image_size, crop_mode = get_mode_config(args.mode)
        print(f"Using mode: {args.mode} (base_size={base_size}, image_size={image_size}, crop_mode={crop_mode})")
    else:
        # Use config.py defaults
        from config import BASE_SIZE, IMAGE_SIZE
        base_size, image_size, crop_mode = BASE_SIZE, IMAGE_SIZE, CROP_MODE
        print(f"Using config.py settings (base_size={base_size}, image_size={image_size}, crop_mode={crop_mode})")
    
    # Get input/output paths
    input_path = args.input if args.input else INPUT_PATH
    output_path = args.output if args.output else OUTPUT_PATH

    # INPUT_PATH = OmniDocBench images path

    os.makedirs(output_path, exist_ok=True)

    # print('image processing until processing prompts.....')

    print(f'{Colors.RED}glob images.....{Colors.RESET}')

    images_path = glob.glob(f'{input_path}/*')

    images = []

    for image_path in images_path:
        image = Image.open(image_path).convert('RGB')
        images.append(image)

    prompt = PROMPT

    # batch_inputs = []


    # for image in tqdm(images):

    #     prompt_in = prompt
    #     cache_list = [
    #         {
    #             "prompt": prompt_in,
    #             "multi_modal_data": {"image": Image.open(image).convert('RGB')},
    #         }
    #     ]
    #     batch_inputs.extend(cache_list)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
        batch_inputs = list(tqdm(
            executor.map(lambda img: process_single_image(img, crop_mode, image_size, base_size), images),
            total=len(images),
            desc="Pre-processed images"
        ))


    

    outputs_list = llm.generate(
        batch_inputs,
        sampling_params=sampling_params
    )


    os.makedirs(output_path, exist_ok=True)

    for output, image in zip(outputs_list, images_path):

        content = output.outputs[0].text
        mmd_det_path = output_path + image.split('/')[-1].replace('.jpg', '_det.md')

        with open(mmd_det_path, 'w', encoding='utf-8') as afile:
            afile.write(content)

        content = clean_formula(content)
        matches_ref, mathes_other = re_match(content)
        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
            content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
        
        mmd_path = output_path + image.split('/')[-1].replace('.jpg', '.md')

        with open(mmd_path, 'w', encoding='utf-8') as afile:
            afile.write(content)
