import os
import fitz
import img2pdf
import io
import re
import json
import gc
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
 

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


from config import (MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, SKIP_REPEAT, 
                    MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE, PDF_BATCH_SIZE,
                    ENABLE_MEMORY_MONITORING, MEMORY_CLEANUP_FREQUENCY,
                    CHECKPOINT_ENABLED, CHECKPOINT_DIR, GPU_MEMORY_THRESHOLD_GB)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from process.memory_utils import (MemoryMonitor, clear_memory, clear_memory_aggressive,
                                  MemoryContext, print_memory_summary)

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
    pdf2images - Load all pages (for backward compatibility)
    WARNING: This loads all pages into memory. Use pdf_to_images_chunked for large PDFs.
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
    Convert a range of PDF pages to images (memory-efficient).
    
    Args:
        pdf_path: Path to PDF file
        start_page: Starting page number (0-indexed)
        end_page: Ending page number (exclusive)
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
        
        # Clear pixmap memory
        pixmap = None
    
    pdf_document.close()
    
    return images


def get_pdf_page_count(pdf_path):
    """Get total number of pages in PDF."""
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


def save_checkpoint(checkpoint_path, processed_pages, contents_det, contents, draw_images, jdx):
    """Save processing checkpoint."""
    checkpoint_data = {
        'processed_pages': processed_pages,
        'contents_det': contents_det,
        'contents': contents,
        'jdx': jdx,
        'num_draw_images': len(draw_images)
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save checkpoint metadata
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
    
    # Save draw images separately
    draw_images_dir = checkpoint_path.replace('.json', '_images')
    os.makedirs(draw_images_dir, exist_ok=True)
    for idx, img in enumerate(draw_images):
        img.save(os.path.join(draw_images_dir, f'page_{idx}.png'))
    
    print(f"✓ Checkpoint saved: {processed_pages} pages processed")


def load_checkpoint(checkpoint_path):
    """Load processing checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None
    
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)
    
    # Load draw images
    draw_images_dir = checkpoint_path.replace('.json', '_images')
    draw_images = []
    if os.path.exists(draw_images_dir):
        for idx in range(checkpoint_data['num_draw_images']):
            img_path = os.path.join(draw_images_dir, f'page_{idx}.png')
            if os.path.exists(img_path):
                draw_images.append(Image.open(img_path))
    
    checkpoint_data['draw_images'] = draw_images
    print(f"✓ Checkpoint loaded: resuming from page {checkpoint_data['processed_pages']}")
    
    return checkpoint_data


def cleanup_checkpoint(checkpoint_path):
    """Remove checkpoint files after successful completion."""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    draw_images_dir = checkpoint_path.replace('.json', '_images')
    if os.path.exists(draw_images_dir):
        import shutil
        shutil.rmtree(draw_images_dir)
    
    print("✓ Checkpoint cleaned up")


if __name__ == "__main__":

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(enable_monitoring=ENABLE_MEMORY_MONITORING)
    memory_monitor.log_memory_usage("Initial")
    
    print(f'{Colors.RED}PDF loading .....{Colors.RESET}')
    
    # Get total page count
    total_pages = get_pdf_page_count(INPUT_PATH)
    print(f'{Colors.GREEN}Total pages: {total_pages}{Colors.RESET}')
    
    # Determine if we should use chunked processing
    use_chunked_processing = total_pages > PDF_BATCH_SIZE
    
    if use_chunked_processing:
        print(f'{Colors.YELLOW}Large PDF detected. Using chunked processing with batch size: {PDF_BATCH_SIZE}{Colors.RESET}')
    
    prompt = PROMPT
    
    # Setup checkpoint
    checkpoint_path = None
    if CHECKPOINT_ENABLED:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        pdf_name = os.path.basename(INPUT_PATH).replace('.pdf', '')
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{pdf_name}_checkpoint.json')
    
    # Try to load checkpoint
    checkpoint_data = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_data = load_checkpoint(checkpoint_path)
    
    # Initialize or restore from checkpoint
    if checkpoint_data:
        contents_det = checkpoint_data['contents_det']
        contents = checkpoint_data['contents']
        draw_images = checkpoint_data['draw_images']
        jdx = checkpoint_data['jdx']
        start_page = checkpoint_data['processed_pages']
        print(f'{Colors.GREEN}Resuming from page {start_page}{Colors.RESET}')
    else:
        contents_det = ''
        contents = ''
        draw_images = []
        jdx = 0
        start_page = 0
    
    output_path = OUTPUT_PATH
    os.makedirs(output_path, exist_ok=True)
    
    mmd_det_path = output_path + '/' + INPUT_PATH.split('/')[-1].replace('.pdf', '_det.mmd')
    mmd_path = output_path + '/' + INPUT_PATH.split('/')[-1].replace('pdf', 'mmd')
    pdf_out_path = output_path + '/' + INPUT_PATH.split('/')[-1].replace('.pdf', '_layouts.pdf')
    
    try:
        # Process in chunks
        for chunk_start in range(start_page, total_pages, PDF_BATCH_SIZE):
            chunk_end = min(chunk_start + PDF_BATCH_SIZE, total_pages)
            
            print(f'\n{Colors.BLUE}Processing pages {chunk_start} to {chunk_end-1} ({chunk_end-chunk_start} pages){Colors.RESET}')
            
            with MemoryContext(f"Chunk {chunk_start}-{chunk_end}", memory_monitor):
                # Load images for this chunk
                if use_chunked_processing:
                    images = pdf_to_images_chunked(INPUT_PATH, chunk_start, chunk_end)
                else:
                    # For small PDFs, load all at once (backward compatibility)
                    images = pdf_to_images_high_quality(INPUT_PATH)
                
                memory_monitor.log_memory_usage(f"After loading {len(images)} images")
                
                # Preprocess images with memory cleanup
                batch_inputs = []
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    batch_inputs = list(tqdm(
                        executor.map(process_single_image, images),
                        total=len(images),
                        desc=f"Pre-processing pages {chunk_start}-{chunk_end-1}"
                    ))
                
                memory_monitor.log_memory_usage("After preprocessing")
                
                # Check memory before inference
                if not memory_monitor.check_memory_threshold(GPU_MEMORY_THRESHOLD_GB):
                    print(f'{Colors.YELLOW}Performing aggressive memory cleanup...{Colors.RESET}')
                    clear_memory_aggressive()
                    memory_monitor.log_memory_usage("After aggressive cleanup")
                
                # Run inference
                print(f'{Colors.GREEN}Running inference...{Colors.RESET}')
                outputs_list = llm.generate(
                    batch_inputs,
                    sampling_params=sampling_params
                )
                
                memory_monitor.log_memory_usage("After inference")
                
                # Process outputs
                for output, img in zip(outputs_list, images):
                    content = output.outputs[0].text

                    if '<｜end▁of▁sentence｜>' in content:
                        content = content.replace('<｜end▁of▁sentence｜>', '')
                    else:
                        if SKIP_REPEAT:
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
                
                # Clear references to images and outputs
                del images
                del batch_inputs
                del outputs_list
                
                # Aggressive memory cleanup after each chunk
                clear_memory_aggressive()
                
                # Save checkpoint after each chunk
                if checkpoint_path and CHECKPOINT_ENABLED:
                    save_checkpoint(checkpoint_path, chunk_end, contents_det, contents, draw_images, jdx)
                
                memory_monitor.log_memory_usage(f"After chunk {chunk_start}-{chunk_end} cleanup")
        
        # Write final outputs
        print(f'\n{Colors.GREEN}Writing final outputs...{Colors.RESET}')
        
        with open(mmd_det_path, 'w', encoding='utf-8') as afile:
            afile.write(contents_det)

        with open(mmd_path, 'w', encoding='utf-8') as afile:
            afile.write(contents)

        pil_to_pdf_img2pdf(draw_images, pdf_out_path)
        
        # Cleanup checkpoint on success
        if checkpoint_path and CHECKPOINT_ENABLED:
            cleanup_checkpoint(checkpoint_path)
        
        print(f'\n{Colors.GREEN}✓ Processing completed successfully!{Colors.RESET}')
        print_memory_summary(memory_monitor)
        
    except Exception as e:
        print(f'\n{Colors.RED}✗ Error during processing: {e}{Colors.RESET}')
        if checkpoint_path and CHECKPOINT_ENABLED:
            print(f'{Colors.YELLOW}Checkpoint saved. You can resume processing by running the script again.{Colors.RESET}')
        raise

