"""
Batched PDF processing script for DeepSeek OCR
Designed to handle large PDFs (2800+ pages) without memory issues
Processes PDF in configurable batches with aggressive memory cleanup
"""

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
                    MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE, BATCH_SIZE, 
                    ENABLE_MEMORY_CLEANUP, VERBOSE_MEMORY_STATS)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from process.memory_utils import cleanup_memory, MemoryMonitor, print_memory_stats

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


def pdf_to_images_batched(pdf_path, start_page, end_page, dpi=144, image_format="PNG"):
    """
    Convert a range of PDF pages to images (memory-efficient version).
    
    Args:
        pdf_path: Path to the PDF file
        start_page: Starting page index (0-based)
        end_page: Ending page index (exclusive)
        dpi: Resolution for image conversion
        image_format: Output image format
    
    Returns:
        List of PIL Images for the specified page range
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
        
        # Clean up pixmap
        del pixmap
    
    pdf_document.close()
    return images


def get_pdf_page_count(pdf_path):
    """Get total number of pages in PDF."""
    pdf_document = fitz.open(pdf_path)
    page_count = pdf_document.page_count
    pdf_document.close()
    return page_count


def pil_to_pdf_img2pdf(pil_images, output_path):
    """Convert PIL images to PDF using img2pdf."""
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
        print(f"Error creating PDF: {e}")


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


def draw_bounding_boxes(image, refs, jdx, output_path):
    """Draw bounding boxes on image and save cropped regions."""
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
                            cropped.save(f"{output_path}/images/{jdx}_{img_idx}.jpg")
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


def process_single_image(image):
    """Process a single image for OCR."""
    prompt_in = PROMPT
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=CROP_MODE
            )
        },
    }
    return cache_item


def process_batch(llm, images, sampling_params, batch_start_idx, output_path):
    """
    Process a batch of images through the OCR model.
    
    Args:
        llm: The LLM model instance
        images: List of PIL Images to process
        sampling_params: Sampling parameters for generation
        batch_start_idx: Starting index for this batch (for page numbering)
        output_path: Output directory path
    
    Returns:
        Tuple of (contents_det, contents, draw_images)
    """
    with MemoryMonitor(f"Batch {batch_start_idx}-{batch_start_idx + len(images)}", 
                       verbose=VERBOSE_MEMORY_STATS):
        # Preprocess images
        print(f"{Colors.YELLOW}Preprocessing batch of {len(images)} images...{Colors.RESET}")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(tqdm(
                executor.map(process_single_image, images),
                total=len(images),
                desc="Pre-processing images"
            ))
        
        # Run inference
        print(f"{Colors.YELLOW}Running OCR inference...{Colors.RESET}")
        outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)
        
        # Clean up batch inputs to free memory
        del batch_inputs
        if ENABLE_MEMORY_CLEANUP:
            cleanup_memory(verbose=VERBOSE_MEMORY_STATS)
        
        # Process outputs
        contents_det = ''
        contents = ''
        draw_images = []
        
        for idx, (output, img) in enumerate(zip(outputs_list, images)):
            jdx = batch_start_idx + idx
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
            result_image = draw_bounding_boxes(image_draw, matches_ref, jdx, output_path)
            draw_images.append(result_image)

            for img_idx, a_match_image in enumerate(matches_images):
                content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(img_idx) + '.jpg)\n')

            for idx2, a_match_other in enumerate(mathes_other):
                content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

            contents += content + f'\n{page_num}\n'
        
        # Clean up outputs
        del outputs_list
        if ENABLE_MEMORY_CLEANUP:
            cleanup_memory(verbose=VERBOSE_MEMORY_STATS)
        
        return contents_det, contents, draw_images


def main():
    """Main function for batched PDF processing."""
    print(f"{Colors.GREEN}=== DeepSeek OCR - Batched PDF Processing ==={Colors.RESET}")
    print(f"Batch size: {BATCH_SIZE} pages")
    print(f"Memory cleanup: {'Enabled' if ENABLE_MEMORY_CLEANUP else 'Disabled'}")
    
    # Create output directories
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)
    
    # Initialize model
    print(f'{Colors.RED}Initializing model...{Colors.RESET}')
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

    logits_processors = [NoRepeatNGramLogitsProcessor(
        ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
    )]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )
    
    # Get total page count
    print(f'{Colors.RED}Loading PDF...{Colors.RESET}')
    total_pages = get_pdf_page_count(INPUT_PATH)
    print(f'{Colors.GREEN}Total pages: {total_pages}{Colors.RESET}')
    
    # Prepare output files
    mmd_det_path = OUTPUT_PATH + '/' + INPUT_PATH.split('/')[-1].replace('.pdf', '_det.mmd')
    mmd_path = OUTPUT_PATH + '/' + INPUT_PATH.split('/')[-1].replace('pdf', 'mmd')
    pdf_out_path = OUTPUT_PATH + '/' + INPUT_PATH.split('/')[-1].replace('.pdf', '_layouts.pdf')
    
    # Open output files in append mode
    with open(mmd_det_path, 'w', encoding='utf-8') as f:
        f.write('')  # Clear file
    with open(mmd_path, 'w', encoding='utf-8') as f:
        f.write('')  # Clear file
    
    all_draw_images = []
    
    # Process PDF in batches
    num_batches = (total_pages + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_page = batch_idx * BATCH_SIZE
        end_page = min((batch_idx + 1) * BATCH_SIZE, total_pages)
        
        print(f'\n{Colors.BLUE}{"="*60}{Colors.RESET}')
        print(f'{Colors.BLUE}Processing batch {batch_idx + 1}/{num_batches}: '
              f'Pages {start_page + 1}-{end_page}{Colors.RESET}')
        print(f'{Colors.BLUE}{"="*60}{Colors.RESET}\n')
        
        if VERBOSE_MEMORY_STATS:
            print_memory_stats("Before batch: ")
        
        # Load images for this batch
        print(f'{Colors.YELLOW}Loading pages {start_page + 1}-{end_page}...{Colors.RESET}')
        images = pdf_to_images_batched(INPUT_PATH, start_page, end_page)
        
        # Process batch
        contents_det, contents, draw_images = process_batch(
            llm, images, sampling_params, start_page, OUTPUT_PATH
        )
        
        # Append results to files
        with open(mmd_det_path, 'a', encoding='utf-8') as f:
            f.write(contents_det)
        with open(mmd_path, 'a', encoding='utf-8') as f:
            f.write(contents)
        
        # Store draw images
        all_draw_images.extend(draw_images)
        
        # Clean up batch data
        del images
        del draw_images
        del contents_det
        del contents
        
        if ENABLE_MEMORY_CLEANUP:
            cleanup_memory(verbose=VERBOSE_MEMORY_STATS)
        
        print(f'{Colors.GREEN}Batch {batch_idx + 1}/{num_batches} completed{Colors.RESET}')
        
        if VERBOSE_MEMORY_STATS:
            print_memory_stats("After batch: ")
    
    # Create output PDF with all drawn images
    print(f'\n{Colors.YELLOW}Creating output PDF with layouts...{Colors.RESET}')
    pil_to_pdf_img2pdf(all_draw_images, pdf_out_path)
    
    # Final cleanup
    del all_draw_images
    cleanup_memory(verbose=True)
    
    print(f'\n{Colors.GREEN}{"="*60}{Colors.RESET}')
    print(f'{Colors.GREEN}Processing completed successfully!{Colors.RESET}')
    print(f'{Colors.GREEN}Output files:{Colors.RESET}')
    print(f'  - {mmd_det_path}')
    print(f'  - {mmd_path}')
    print(f'  - {pdf_out_path}')
    print(f'{Colors.GREEN}{"="*60}{Colors.RESET}\n')


if __name__ == "__main__":
    main()
