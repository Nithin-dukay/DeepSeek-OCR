import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
 

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


from config import (MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, 
                    MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE, PDF_BATCH_SIZE, 
                    INFERENCE_BATCH_SIZE, ENABLE_MEMORY_MONITORING, MEMORY_CLEANUP_FREQUENCY)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True, 
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    disable_mm_preprocessor_cache=True
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids= {128821, 128822})] #window for fast；whitelist_token_ids: <td>,</td>

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m' 

def get_memory_usage():
    """Get current memory usage in GB"""
    if ENABLE_MEMORY_MONITORING:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / (1024 ** 3)
        
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            return mem_gb, gpu_mem_allocated, gpu_mem_reserved
        return mem_gb, 0, 0
    return 0, 0, 0

def cleanup_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def pdf_to_images_chunked(pdf_path, dpi=144, image_format="PNG", batch_size=PDF_BATCH_SIZE):
    """
    Convert PDF to images in chunks to avoid memory overflow
    Yields batches of images instead of loading all at once
    """
    pdf_document = fitz.open(pdf_path)
    total_pages = pdf_document.page_count
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    print(f'{Colors.YELLOW}Total pages: {total_pages}, processing in batches of {batch_size}{Colors.RESET}')
    
    for start_idx in range(0, total_pages, batch_size):
        end_idx = min(start_idx + batch_size, total_pages)
        images_batch = []
        
        if ENABLE_MEMORY_MONITORING:
            mem_gb, gpu_alloc, gpu_res = get_memory_usage()
            print(f'{Colors.BLUE}Processing pages {start_idx+1}-{end_idx} | RAM: {mem_gb:.2f}GB | GPU Alloc: {gpu_alloc:.2f}GB | GPU Reserved: {gpu_res:.2f}GB{Colors.RESET}')
        
        for page_num in range(start_idx, end_idx):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            Image.MAX_IMAGE_PIXELS = None

            if image_format.upper() == "PNG":
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
            else:
                img_data = pixmap.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
            
            images_batch.append(img)
            
            # Explicitly delete pixmap to free memory
            del pixmap
            del img_data
        
        yield images_batch, start_idx, end_idx
        
        # Clean up after each batch
        del images_batch
        cleanup_memory()
    
    pdf_document.close()

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    Legacy function - now uses chunked processing internally
    Returns all images (for backward compatibility with small PDFs)
    """
    all_images = []
    for images_batch, _, _ in pdf_to_images_chunked(pdf_path, dpi, image_format):
        all_images.extend(images_batch)
    return all_images

def pil_to_pdf_img2pdf(pil_images, output_path):

    if not pil_images:
        return
    
    image_bytes_list = []
    
    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)
    
    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")



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


def draw_bounding_boxes(image, refs, jdx):

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    #     except IOError:
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
                            cropped.save(f"{OUTPUT_PATH}/images/{jdx}_{img_idx}.jpg")
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


def process_image_with_refs(image, ref_texts, jdx):
    result_image = draw_bounding_boxes(image, ref_texts, jdx)
    return result_image


def process_single_image(image):
    """single image"""
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images = [image], bos=True, eos=True, cropping=CROP_MODE)},
    }
    return cache_item

def process_images_in_batches(images, batch_size=INFERENCE_BATCH_SIZE):
    """
    Process images in smaller batches for inference
    Yields batches of processed inputs
    """
    total_images = len(images)
    print(f'{Colors.YELLOW}Processing {total_images} images in inference batches of {batch_size}{Colors.RESET}')
    
    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        batch_images = images[start_idx:end_idx]
        
        # Process batch with thread pool
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(tqdm(
                executor.map(process_single_image, batch_images),
                total=len(batch_images),
                desc=f"Pre-processing batch {start_idx//batch_size + 1}"
            ))
        
        yield batch_inputs, start_idx, end_idx
        
        # Clean up
        del batch_inputs
        del batch_images


if __name__ == "__main__":

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)
    
    print(f'{Colors.RED}PDF loading and processing with memory optimization.....{Colors.RESET}')
    
    if ENABLE_MEMORY_MONITORING:
        mem_gb, gpu_alloc, gpu_res = get_memory_usage()
        print(f'{Colors.BLUE}Initial Memory - RAM: {mem_gb:.2f}GB | GPU Alloc: {gpu_alloc:.2f}GB | GPU Reserved: {gpu_res:.2f}GB{Colors.RESET}')

    prompt = PROMPT
    
    output_path = OUTPUT_PATH
    os.makedirs(output_path, exist_ok=True)
    
    mmd_det_path = output_path + '/' + INPUT_PATH.split('/')[-1].replace('.pdf', '_det.mmd')
    mmd_path = output_path + '/' + INPUT_PATH.split('/')[-1].replace('pdf', 'mmd')
    pdf_out_path = output_path + '/' + INPUT_PATH.split('/')[-1].replace('.pdf', '_layouts.pdf')
    
    contents_det = ''
    contents = ''
    draw_images = []
    jdx = 0
    batch_counter = 0
    
    # Process PDF in chunks
    for images_batch, pdf_start_idx, pdf_end_idx in pdf_to_images_chunked(INPUT_PATH):
        print(f'{Colors.GREEN}Processing PDF pages {pdf_start_idx+1}-{pdf_end_idx}{Colors.RESET}')
        
        # Process images in inference batches
        for batch_inputs, inf_start_idx, inf_end_idx in process_images_in_batches(images_batch, INFERENCE_BATCH_SIZE):
            
            if ENABLE_MEMORY_MONITORING:
                mem_gb, gpu_alloc, gpu_res = get_memory_usage()
                print(f'{Colors.BLUE}Before inference - RAM: {mem_gb:.2f}GB | GPU Alloc: {gpu_alloc:.2f}GB | GPU Reserved: {gpu_res:.2f}GB{Colors.RESET}')
            
            # Run inference on batch
            outputs_list = llm.generate(
                batch_inputs,
                sampling_params=sampling_params
            )
            
            # Process outputs immediately
            batch_images = images_batch[inf_start_idx:inf_end_idx]
            for output, img in zip(outputs_list, batch_images):
                content = output.outputs[0].text

                if '<｜end▁of▁sentence｜>' in content:  # repeat no eos
                    content = content.replace('<｜end▁of▁sentence｜>', '')
                else:
                    if SKIP_REPEAT:
                        jdx += 1
                        continue

                page_num = f'\n<--- Page Split --->'
                contents_det += content + f'\n{page_num}\n'

                image_draw = img.copy()
                matches_ref, matches_images, mathes_other = re_match(content)
                result_image = process_image_with_refs(image_draw, matches_ref, jdx)
                draw_images.append(result_image)

                for idx, a_match_image in enumerate(matches_images):
                    content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

                for idx, a_match_other in enumerate(mathes_other):
                    content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

                contents += content + f'\n{page_num}\n'
                jdx += 1
                
                # Clean up image references
                del image_draw
            
            # Clean up batch
            del outputs_list
            del batch_inputs
            del batch_images
            
            batch_counter += 1
            
            # Periodic memory cleanup
            if batch_counter % MEMORY_CLEANUP_FREQUENCY == 0:
                print(f'{Colors.YELLOW}Running periodic memory cleanup...{Colors.RESET}')
                cleanup_memory()
                
                if ENABLE_MEMORY_MONITORING:
                    mem_gb, gpu_alloc, gpu_res = get_memory_usage()
                    print(f'{Colors.BLUE}After cleanup - RAM: {mem_gb:.2f}GB | GPU Alloc: {gpu_alloc:.2f}GB | GPU Reserved: {gpu_res:.2f}GB{Colors.RESET}')
        
        # Clean up PDF batch
        del images_batch
        cleanup_memory()
    
    print(f'{Colors.GREEN}All pages processed. Writing output files...{Colors.RESET}')
    
    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)

    pil_to_pdf_img2pdf(draw_images, pdf_out_path)
    
    print(f'{Colors.GREEN}Processing complete!{Colors.RESET}')
    
    if ENABLE_MEMORY_MONITORING:
        mem_gb, gpu_alloc, gpu_res = get_memory_usage()
        print(f'{Colors.BLUE}Final Memory - RAM: {mem_gb:.2f}GB | GPU Alloc: {gpu_alloc:.2f}GB | GPU Reserved: {gpu_res:.2f}GB{Colors.RESET}')

