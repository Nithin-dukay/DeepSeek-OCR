"""
Test script for GitHub Issue #297: Partial content recognition problem

This script demonstrates the fix for the issue where OCR only recognizes 
partial content (e.g., "xxx单位") but misses the date on the right side.

Usage:
    python test_issue_297.py --image <path_to_image> [--disable-ngram] [--ngram-size N] [--window-size N]

Example:
    # Test with default settings
    python test_issue_297.py --image test_issue_297.png
    
    # Test with disabled n-gram filtering (recommended for sparse documents)
    python test_issue_297.py --image test_issue_297.png --disable-ngram
    
    # Test with adjusted n-gram parameters
    python test_issue_297.py --image test_issue_297.png --ngram-size 20 --window-size 150
"""

import asyncio
import argparse
import os
import sys
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
from PIL import Image, ImageOps
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

# Import default config values
from config import MODEL_PATH, CROP_MODE

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def load_image(image_path):
    """Load and correct image orientation"""
    try:
        image = Image.open(image_path)
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        print(f"Error loading image: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


async def test_ocr(image_path, disable_ngram=False, ngram_size=30, window_size=90, prompt=None):
    """
    Test OCR with configurable n-gram filtering
    
    Args:
        image_path: Path to the image file
        disable_ngram: If True, disable n-gram filtering completely
        ngram_size: Size of n-gram for repetition detection
        window_size: Window size for searching repeated n-grams
        prompt: Custom prompt (default: Free OCR)
    """
    
    print("="*80)
    print("DeepSeek-OCR Test for Issue #297")
    print("="*80)
    print(f"Image: {image_path}")
    print(f"N-gram filtering: {'DISABLED' if disable_ngram else 'ENABLED'}")
    if not disable_ngram:
        print(f"N-gram size: {ngram_size}")
        print(f"Window size: {window_size}")
    print("="*80)
    
    # Load image
    image = load_image(image_path)
    if image is None:
        print("Failed to load image!")
        return None
    
    image = image.convert('RGB')
    print(f"Image size: {image.size}")
    
    # Setup engine
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
    
    print("Initializing engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Configure logits processors
    if disable_ngram:
        logits_processors = []
        print("✓ N-gram filtering disabled")
    else:
        logits_processors = [NoRepeatNGramLogitsProcessor(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids={128821, 128822}  # <td>, </td>
        )]
        print(f"✓ N-gram filter configured: size={ngram_size}, window={window_size}")
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )
    
    # Prepare prompt
    if prompt is None:
        prompt = "<image>\nFree OCR."
    
    print(f"Prompt: {prompt}")
    print("="*80)
    
    # Process image
    if '<image>' in prompt:
        image_features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image], 
            bos=True, 
            eos=True, 
            cropping=CROP_MODE
        )
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features}
        }
    else:
        request = {"prompt": prompt}
    
    # Generate
    request_id = f"test-{int(time.time())}"
    print("Generating OCR output...")
    print("-"*80)
    
    printed_length = 0
    final_output = ""
    
    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text
    
    print("\n" + "="*80)
    print("OCR Complete!")
    print("="*80)
    
    return final_output


def main():
    parser = argparse.ArgumentParser(
        description="Test DeepSeek-OCR with configurable n-gram filtering (Issue #297)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings
  python test_issue_297.py --image test_issue_297.png
  
  # Test with disabled n-gram filtering (recommended for sparse documents)
  python test_issue_297.py --image test_issue_297.png --disable-ngram
  
  # Test with adjusted n-gram parameters
  python test_issue_297.py --image test_issue_297.png --ngram-size 20 --window-size 150
  
  # Test with custom prompt
  python test_issue_297.py --image test_issue_297.png --prompt "<image>\\n<|grounding|>Convert the document to markdown."
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file to test')
    parser.add_argument('--disable-ngram', action='store_true',
                        help='Disable n-gram filtering completely (recommended for sparse documents)')
    parser.add_argument('--ngram-size', type=int, default=30,
                        help='N-gram size for repetition detection (default: 30)')
    parser.add_argument('--window-size', type=int, default=90,
                        help='Window size for searching repeated n-grams (default: 90)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt (default: "<image>\\nFree OCR.")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save results (optional)')
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Run test
    result = asyncio.run(test_ocr(
        image_path=args.image,
        disable_ngram=args.disable_ngram,
        ngram_size=args.ngram_size,
        window_size=args.window_size,
        prompt=args.prompt
    ))
    
    # Save output if requested
    if args.output and result:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n✓ Results saved to: {args.output}")
    
    # Analysis
    if result:
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)
        print(f"Total characters: {len(result)}")
        print(f"Total lines: {result.count(chr(10)) + 1}")
        
        # Check for common date patterns
        import re
        date_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日',  # Chinese date format
            r'\d{4}-\d{2}-\d{2}',           # ISO date format
            r'\d{2}/\d{2}/\d{4}',           # US date format
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, result)
            dates_found.extend(matches)
        
        if dates_found:
            print(f"✓ Dates found: {', '.join(dates_found)}")
        else:
            print("⚠ No dates detected in output")
        
        print("="*80)


if __name__ == "__main__":
    main()
