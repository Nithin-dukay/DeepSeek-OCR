"""
Example script demonstrating dynamic mode selection with vLLM.

This script shows how to use different modes (Tiny, Small, Base, Large, Gundam)
when processing images with DeepSeek-OCR deployed via vLLM.

Modes:
- Tiny: base_size=512, image_size=512, crop_mode=False (64 vision tokens)
- Small: base_size=640, image_size=640, crop_mode=False (100 vision tokens)
- Base: base_size=1024, image_size=1024, crop_mode=False (256 vision tokens)
- Large: base_size=1280, image_size=1280, crop_mode=False (400 vision tokens)
- Gundam: base_size=1024, image_size=640, crop_mode=True (dynamic, n×100 + 256 tokens)
"""

import asyncio
import re
import os

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
import time
from deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, OUTPUT_PATH

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Mode configurations
MODES = {
    'tiny': {'base_size': 512, 'image_size': 512, 'crop_mode': False},
    'small': {'base_size': 640, 'image_size': 640, 'crop_mode': False},
    'base': {'base_size': 1024, 'image_size': 1024, 'crop_mode': False},
    'large': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False},
    'gundam': {'base_size': 1024, 'image_size': 640, 'crop_mode': True},
}

def load_image(image_path):
    """Load and correct image orientation."""
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    """Extract reference patterns from text."""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """Extract coordinates and labels from reference text."""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs):
    """Draw bounding boxes on image based on references."""
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20, )
                
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{OUTPUT_PATH}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts):
    """Process image with reference texts."""
    result_image = draw_bounding_boxes(image, ref_texts)
    return result_image


async def stream_generate(image=None, prompt='', mode='base'):
    """
    Generate OCR output with streaming.
    
    Args:
        image: Preprocessed image features
        prompt: Text prompt
        mode: Mode name (tiny, small, base, large, gundam)
    """
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,  
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822})]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )
    
    request_id = f"request-{int(time.time())}"
    printed_length = 0

    if image and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    elif prompt:
        request = {
            "prompt": prompt
        }
    else:
        assert False, f'prompt is none!!!'
        
    print(f"\n{'='*50}")
    print(f"Processing with mode: {mode.upper()}")
    print(f"{'='*50}\n")
    
    async for request_output in engine.generate(
        request, sampling_params, request_id
    ):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text
    print('\n') 

    return final_output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSeek-OCR with dynamic mode selection')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--mode', type=str, default='base', 
                       choices=['tiny', 'small', 'base', 'large', 'gundam'],
                       help='OCR mode: tiny (512), small (640), base (1024), large (1280), gundam (dynamic)')
    parser.add_argument('--prompt', type=str, default='<image>\\n<|grounding|>Convert the document to markdown.',
                       help='Prompt template')
    parser.add_argument('--save-results', action='store_true', help='Save results to files')
    
    args = parser.parse_args()
    
    # Get mode configuration
    mode_config = MODES[args.mode]
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(f'{args.output}/images', exist_ok=True)

    # Load image
    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    if image is None:
        print("Failed to load image!")
        exit(1)
    
    image = image.convert('RGB')
    
    # Process image with selected mode
    print(f"\nProcessing with mode: {args.mode}")
    print(f"  - base_size: {mode_config['base_size']}")
    print(f"  - image_size: {mode_config['image_size']}")
    print(f"  - crop_mode: {mode_config['crop_mode']}")
    
    if '<image>' in args.prompt:
        image_features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image], 
            bos=True, 
            eos=True, 
            base_size=mode_config['base_size'],
            image_size=mode_config['image_size'],
            crop_mode=mode_config['crop_mode']
        )
    else:
        image_features = ''

    # Run inference
    result_out = asyncio.run(stream_generate(image_features, args.prompt, args.mode))

    # Save results
    if args.save_results and '<image>' in args.prompt:
        print('='*15 + ' Saving results ' + '='*15)

        image_draw = image.copy()
        outputs = result_out

        # Save original output
        with open(f'{args.output}/result_ori_{args.mode}.mmd', 'w', encoding='utf-8') as afile:
            afile.write(outputs)

        # Process references
        matches_ref, matches_images, mathes_other = re_match(outputs)
        result = process_image_with_refs(image_draw, matches_ref)

        # Replace image references
        for idx, a_match_image in enumerate(tqdm(matches_images, desc="Processing images")):
            outputs = outputs.replace(a_match_image, f'![](images/' + str(idx) + '.jpg)\\n')

        # Clean up other references
        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="Processing other refs")):
            outputs = outputs.replace(a_match_other, '').replace('\\\\coloneqq', ':=').replace('\\\\eqqcolon', '=:')

        # Save processed output
        with open(f'{args.output}/result_{args.mode}.mmd', 'w', encoding='utf-8') as afile:
            afile.write(outputs)

        # Save image with bounding boxes
        result.save(f'{args.output}/result_with_boxes_{args.mode}.jpg')
        
        print(f"\nResults saved to {args.output}/")
        print(f"  - result_ori_{args.mode}.mmd (original output)")
        print(f"  - result_{args.mode}.mmd (processed output)")
        print(f"  - result_with_boxes_{args.mode}.jpg (annotated image)")
