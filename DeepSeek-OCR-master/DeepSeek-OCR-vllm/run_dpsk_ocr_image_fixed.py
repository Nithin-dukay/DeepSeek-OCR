"""
Fixed version of run_dpsk_ocr_image.py to address GitHub Issue #299
Triton Error [CUDA]: illegal memory access on vLLM 0.11.2

Key fixes:
1. Enable enforce_eager to disable CUDA graphs (workaround for Triton kernel bug)
2. Reduce GPU memory utilization to 0.75
3. Add CUDA_LAUNCH_BLOCKING for better error reporting
4. Reduce max_model_len to avoid memory issues
5. Add retry logic with fallback parameters
"""

import asyncio
import re
import os
import sys

import torch

# Critical environment variables for fixing Triton/CUDA issues
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

# Use V0 engine (more stable for MoE models)
os.environ['VLLM_USE_V1'] = '0'

# Enable CUDA blocking for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set visible devices
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
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

def load_image(image_path):
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
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)

def draw_bounding_boxes(image, refs):
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
    result_image = draw_bounding_boxes(image, ref_texts)
    return result_image

async def stream_generate(image=None, prompt='', retry_count=0, max_retries=2):
    """
    Generate text from image with retry logic for handling Triton/CUDA errors.
    
    Args:
        image: Preprocessed image features
        prompt: Text prompt
        retry_count: Current retry attempt
        max_retries: Maximum number of retries
    """
    
    # Adjust parameters based on retry count
    if retry_count == 0:
        # First attempt: Use safer defaults with enforce_eager
        print("Attempt 1: Using enforce_eager=True (safer mode)")
        enforce_eager = True
        gpu_memory_util = 0.75
        max_model_len = 8192
        block_size = 256
    elif retry_count == 1:
        # Second attempt: Further reduce memory and model length
        print("Attempt 2: Reducing max_model_len and GPU memory")
        enforce_eager = True
        gpu_memory_util = 0.65
        max_model_len = 4096
        block_size = 128
    else:
        # Final attempt: Minimal settings
        print("Attempt 3: Using minimal settings")
        enforce_eager = True
        gpu_memory_util = 0.5
        max_model_len = 2048
        block_size = 64

    try:
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=block_size,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,  # Critical: Disable CUDA graphs to avoid Triton kernel bug
            trust_remote_code=True,  
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_util,
            # Disable prefix caching to reduce memory pressure
            enable_prefix_caching=False,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        logits_processors = [NoRepeatNGramLogitsProcessor(
            ngram_size=30, 
            window_size=90, 
            whitelist_token_ids={128821, 128822}
        )]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_model_len,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )
        
        request_id = f"request-{int(time.time())}-{retry_count}"

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
            raise ValueError('prompt is none!!!')
            
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
        
    except RuntimeError as e:
        error_msg = str(e)
        if ("illegal memory access" in error_msg or 
            "CUBLAS_STATUS_EXECUTION_FAILED" in error_msg or
            "Triton Error" in error_msg):
            
            print(f"\n{'='*60}")
            print(f"ERROR: Triton/CUDA error encountered on attempt {retry_count + 1}")
            print(f"Error: {error_msg[:200]}...")
            print(f"{'='*60}\n")
            
            if retry_count < max_retries:
                print(f"Retrying with adjusted parameters (attempt {retry_count + 2}/{max_retries + 1})...\n")
                # Clear CUDA cache before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                await asyncio.sleep(2)  # Brief pause before retry
                return await stream_generate(image, prompt, retry_count + 1, max_retries)
            else:
                print("\n" + "="*60)
                print("FAILED: All retry attempts exhausted")
                print("="*60)
                print("\nSuggestions:")
                print("1. Try processing with HuggingFace Transformers backend instead:")
                print("   cd DeepSeek-OCR-master/DeepSeek-OCR-hf && python run_dpsk_ocr.py")
                print("2. Reduce image resolution in config.py:")
                print("   BASE_SIZE = 640, IMAGE_SIZE = 640, CROP_MODE = False")
                print("3. Update to vLLM nightly build:")
                print("   pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly")
                print("="*60 + "\n")
                raise
        else:
            # Different error, re-raise
            raise

if __name__ == "__main__":
    print("="*60)
    print("DeepSeek OCR - Fixed Version for Issue #299")
    print("Triton/CUDA Illegal Memory Access Fix")
    print("="*60 + "\n")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)

    if not INPUT_PATH or not os.path.exists(INPUT_PATH):
        print(f"ERROR: INPUT_PATH not set or file doesn't exist: {INPUT_PATH}")
        print("Please set INPUT_PATH in config.py")
        sys.exit(1)

    print(f"Loading image: {INPUT_PATH}")
    image = load_image(INPUT_PATH)
    
    if image is None:
        print(f"ERROR: Failed to load image from {INPUT_PATH}")
        sys.exit(1)
        
    image = image.convert('RGB')
    print(f"Image loaded: {image.size[0]}x{image.size[1]}")

    if '<image>' in PROMPT:
        print("Preprocessing image...")
        image_features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image], 
            bos=True, 
            eos=True, 
            cropping=CROP_MODE
        )
        print("Image preprocessing complete")
    else:
        image_features = ''

    prompt = PROMPT
    print(f"\nPrompt: {prompt}")
    print("\nGenerating output...\n")

    try:
        result_out = asyncio.run(stream_generate(image_features, prompt))
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)

    save_results = 1

    if save_results and '<image>' in prompt:
        print('\n' + '='*15 + ' Saving results ' + '='*15)

        image_draw = image.copy()
        outputs = result_out

        with open(f'{OUTPUT_PATH}/result_ori.mmd', 'w', encoding='utf-8') as afile:
            afile.write(outputs)
        print(f"Saved: {OUTPUT_PATH}/result_ori.mmd")

        matches_ref, matches_images, mathes_other = re_match(outputs)
        result = process_image_with_refs(image_draw, matches_ref)

        for idx, a_match_image in enumerate(tqdm(matches_images, desc="Processing images")):
            outputs = outputs.replace(a_match_image, f'![](images/' + str(idx) + '.jpg)\n')

        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="Processing other refs")):
            outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

        with open(f'{OUTPUT_PATH}/result.mmd', 'w', encoding='utf-8') as afile:
            afile.write(outputs)
        print(f"Saved: {OUTPUT_PATH}/result.mmd")

        if 'line_type' in outputs:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            lines = eval(outputs)['Line']['line']
            line_type = eval(outputs)['Line']['line_type']
            endpoints = eval(outputs)['Line']['line_endpoint']

            fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)

            for idx, line in enumerate(lines):
                try:
                    p0 = eval(line.split(' -- ')[0])
                    p1 = eval(line.split(' -- ')[-1])

                    if line_type[idx] == '--':
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                    else:
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')

                    ax.scatter(p0[0], p0[1], s=5, color='k')
                    ax.scatter(p1[0], p1[1], s=5, color='k')
                except:
                    pass

            for endpoint in endpoints:
                label = endpoint.split(': ')[0]
                (x, y) = eval(endpoint.split(': ')[1])
                ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                            fontsize=5, fontweight='light')
            
            try:
                if 'Circle' in eval(outputs).keys():
                    circle_centers = eval(outputs)['Circle']['circle_center']
                    radius = eval(outputs)['Circle']['radius']

                    for center, r in zip(circle_centers, radius):
                        center = eval(center.split(': ')[1])
                        circle = Circle(center, radius=r, fill=False, edgecolor='black', linewidth=0.8)
                        ax.add_patch(circle)
            except:
                pass

            plt.savefig(f'{OUTPUT_PATH}/geo.jpg')
            plt.close()
            print(f"Saved: {OUTPUT_PATH}/geo.jpg")

        result.save(f'{OUTPUT_PATH}/result_with_boxes.jpg')
        print(f"Saved: {OUTPUT_PATH}/result_with_boxes.jpg")
        
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
