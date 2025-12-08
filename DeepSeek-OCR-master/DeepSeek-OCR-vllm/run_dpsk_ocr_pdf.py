import os
import fitz
import img2pdf
import io
import re
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import gc
 

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


from config import (MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, 
                    MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE, CHUNK_SIZE, 
                    CHUNKED_PROCESSING_THRESHOLD, ENABLE_MEMORY_MONITORING)

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

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images - loads all pages at once (use for small PDFs)
    """
    images = []
    
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
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
        
        images.append(img)
    
    pdf_document.close()
    return images


def pdf_to_images_chunked(pdf_path, start_page, end_page, dpi=144, image_format="PNG"):
    """
    Convert specific pages of PDF to images (memory-efficient for large PDFs)
    
    Args:
        pdf_path: Path to PDF file
        start_page: Starting page index (0-based)
        end_page: Ending page index (exclusive)
        dpi: Resolution for rendering
        image_format: Output image format
    
    Returns:
        List of PIL Image objects for the specified page range
    """
    images = []
    
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(start_page, min(end_page, pdf_document.page_count)):
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
        
        images.append(img)
        
        # Clean up pixmap to free memory
        pixmap = None
    
    pdf_document.close()
    return images


def get_pdf_page_count(pdf_path):
    """Get total number of pages in PDF without loading all pages"""
    pdf_document = fitz.open(pdf_path)
    page_count = pdf_document.page_count
    pdf_document.close()
    return page_count

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


def clear_memory():
    """Explicitly clear Python and CUDA memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_memory_usage(stage=""):
    """Log current memory usage for monitoring"""
    if not ENABLE_MEMORY_MONITORING:
        return
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024  # GB
            print(f'{Colors.YELLOW}[{stage}] RAM: {ram_usage:.2f}GB | GPU Allocated: {gpu_allocated:.2f}GB | GPU Reserved: {gpu_reserved:.2f}GB{Colors.RESET}')
        else:
            print(f'{Colors.YELLOW}[{stage}] RAM: {ram_usage:.2f}GB{Colors.RESET}')
    except ImportError:
        # psutil not available, skip monitoring
        pass
    except Exception as e:
        print(f'{Colors.YELLOW}[{stage}] Memory monitoring error: {e}{Colors.RESET}')


if __name__ == "__main__":

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)
    
    print(f'{Colors.RED}PDF loading .....{Colors.RESET}')

    # Get total page count without loading all pages
    total_pages = get_pdf_page_count(INPUT_PATH)
    print(f'{Colors.GREEN}Total pages: {total_pages}{Colors.RESET}')
    
    # Log initial memory state
    log_memory_usage("Initial")
    
    # Determine if we should use chunked processing
    use_chunked_processing = total_pages > CHUNKED_PROCESSING_THRESHOLD
    
    if use_chunked_processing:
        print(f'{Colors.YELLOW}Using chunked processing (CHUNK_SIZE={CHUNK_SIZE}) for large PDF{Colors.RESET}')
    else:
        print(f'{Colors.YELLOW}Using standard processing for small PDF{Colors.RESET}')

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
    
    if use_chunked_processing:
        # Process PDF in chunks to prevent memory overflow
        num_chunks = (total_pages + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        for chunk_idx in range(num_chunks):
            start_page = chunk_idx * CHUNK_SIZE
            end_page = min((chunk_idx + 1) * CHUNK_SIZE, total_pages)
            
            print(f'{Colors.BLUE}Processing chunk {chunk_idx + 1}/{num_chunks} (pages {start_page + 1}-{end_page}){Colors.RESET}')
            
            log_memory_usage(f"Before chunk {chunk_idx + 1}")
            
            # Load only the current chunk of pages
            images = pdf_to_images_chunked(INPUT_PATH, start_page, end_page)
            
            # Preprocess images with thread pool
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
                batch_inputs = list(tqdm(
                    executor.map(process_single_image, images),
                    total=len(images),
                    desc=f"Pre-processing chunk {chunk_idx + 1}/{num_chunks}"
                ))
            
            # Generate outputs for this chunk
            outputs_list = llm.generate(
                batch_inputs,
                sampling_params=sampling_params
            )
            
            # Process outputs
            for output, img in zip(outputs_list, images):
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
            
            # Explicitly clean up memory after each chunk
            del images
            del batch_inputs
            del outputs_list
            clear_memory()
            
            log_memory_usage(f"After chunk {chunk_idx + 1}")
            print(f'{Colors.GREEN}Completed chunk {chunk_idx + 1}/{num_chunks}, processed {jdx} pages total{Colors.RESET}')
    
    else:
        # Standard processing for smaller PDFs (< 500 pages)
        images = pdf_to_images_high_quality(INPUT_PATH)

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
            batch_inputs = list(tqdm(
                executor.map(process_single_image, images),
                total=len(images),
                desc="Pre-processed images"
            ))

        outputs_list = llm.generate(
            batch_inputs,
            sampling_params=sampling_params
        )

        for output, img in zip(outputs_list, images):
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
        
        # Clean up memory
        del images
        del batch_inputs
        del outputs_list
        clear_memory()

    # Write final outputs
    print(f'{Colors.BLUE}Writing output files...{Colors.RESET}')
    
    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)

    pil_to_pdf_img2pdf(draw_images, pdf_out_path)
    
    print(f'{Colors.GREEN}Processing complete! Processed {jdx} pages.{Colors.RESET}')
    print(f'{Colors.GREEN}Output files saved to: {output_path}{Colors.RESET}')

