"""
Fixed DeepSeek-OCR Inference Script with Anti-Hallucination Features

This script addresses GitHub Issue #191 by incorporating the NoRepeatNGramLogitsProcessor
that prevents hallucination during OCR, especially for challenging documents like
ancient handwritten text.

Usage:
    python run_dpsk_ocr_fixed.py --image your_image.jpg --output ./output --prompt "Free OCR"
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
import argparse
from pathlib import Path
import sys

# Import the anti-hallucination processor
from transformers_logits_processor import create_anti_hallucination_processors


def setup_model(model_name: str = 'deepseek-ai/DeepSeek-OCR', device: str = '0'):
    """
    Load and setup the DeepSeek-OCR model with proper configuration.
    
    Args:
        model_name: HuggingFace model name or local path
        device: CUDA device ID
    
    Returns:
        tuple: (model, tokenizer)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        _attn_implementation='flash_attention_2', 
        trust_remote_code=True, 
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    print("Model loaded successfully!")
    
    return model, tokenizer


def infer_with_anti_hallucination(
    model,
    tokenizer,
    image_file: str,
    prompt: str = "<image>\nFree OCR. ",
    output_path: str = "./output",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    save_results: bool = True,
    test_compress: bool = True,
    ngram_size: int = 30,
    window_size: int = 90,
    mode: str = "standard"
):
    """
    Run OCR inference with anti-hallucination features.
    
    Args:
        model: DeepSeek-OCR model
        tokenizer: Model tokenizer
        image_file: Path to input image
        prompt: OCR prompt (see PROMPT_TEMPLATES for options)
        output_path: Directory to save results
        base_size: Base resolution (512/640/1024/1280)
        image_size: Image size for cropping (640 for Gundam mode)
        crop_mode: Enable dynamic resolution (Gundam mode)
        save_results: Save output to file
        test_compress: Enable compression testing
        ngram_size: N-gram size for repetition blocking (20-40 recommended)
        window_size: Sliding window size (60-120 recommended)
        mode: "standard" or "adaptive" processor mode
    
    Returns:
        str: OCR result text
    """
    # Create anti-hallucination processors
    logits_processors = create_anti_hallucination_processors(
        mode=mode,
        ngram_size=ngram_size,
        window_size=window_size,
        whitelist_token_ids={128821, 128822}  # <td>, </td> for tables
    )
    
    print(f"\nProcessing image: {image_file}")
    print(f"Prompt: {prompt}")
    print(f"Resolution mode: {'Gundam (dynamic)' if crop_mode else f'{base_size}x{base_size} (native)'}")
    print(f"Anti-hallucination: {mode} mode (ngram={ngram_size}, window={window_size})")
    
    # Run inference with the model's infer method
    # Note: We need to patch the model's generate method to use our processors
    original_generate = model.generate
    
    def patched_generate(*args, **kwargs):
        # Inject our logits processors
        if 'logits_processor' in kwargs:
            kwargs['logits_processor'].extend(logits_processors)
        else:
            kwargs['logits_processor'] = logits_processors
        
        # Ensure proper generation config
        kwargs.setdefault('max_new_tokens', 8192)
        kwargs.setdefault('temperature', 0.0)
        kwargs.setdefault('do_sample', False)
        
        return original_generate(*args, **kwargs)
    
    # Temporarily patch the generate method
    model.generate = patched_generate
    
    try:
        result = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=image_file, 
            output_path=output_path, 
            base_size=base_size, 
            image_size=image_size, 
            crop_mode=crop_mode, 
            save_results=save_results, 
            test_compress=test_compress
        )
    finally:
        # Restore original generate method
        model.generate = original_generate
    
    print(f"\n✓ OCR completed successfully!")
    if save_results:
        print(f"✓ Results saved to: {output_path}")
    
    return result


# Prompt templates for different use cases
PROMPT_TEMPLATES = {
    "free_ocr": "<image>\nFree OCR. ",
    "markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
    "ocr_image": "<image>\n<|grounding|>OCR this image. ",
    "parse_figure": "<image>\nParse the figure. ",
    "describe": "<image>\nDescribe this image in detail. ",
}

# Resolution presets
RESOLUTION_PRESETS = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR with Anti-Hallucination (Fix for Issue #191)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Free OCR
  python run_dpsk_ocr_fixed.py --image document.jpg --output ./results
  
  # For ancient/handwritten documents (recommended)
  python run_dpsk_ocr_fixed.py --image ancient.jpg --preset large --ngram 40 --window 120
  
  # Convert document to markdown
  python run_dpsk_ocr_fixed.py --image doc.jpg --prompt markdown --preset gundam
  
  # Adaptive mode for long documents
  python run_dpsk_ocr_fixed.py --image long_doc.jpg --mode adaptive --preset gundam

Prompt Templates:
  free_ocr    : Free OCR without layout preservation
  markdown    : Convert document to markdown (best for structured docs)
  ocr_image   : OCR with layout (best for non-standard documents)
  parse_figure: Parse figures and charts
  describe    : General image description

Resolution Presets:
  tiny   : 512x512   (64 vision tokens)  - Fast, low memory
  small  : 640x640   (100 vision tokens) - Balanced
  base   : 1024x1024 (256 vision tokens) - Good quality
  large  : 1280x1280 (400 vision tokens) - Best quality
  gundam : Dynamic resolution (n×640×640 + 1×1024×1024) - Best for large docs
        """
    )
    
    # Required arguments
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image file')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-OCR',
                        help='Model name or path (default: deepseek-ai/DeepSeek-OCR)')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device ID (default: 0)')
    
    # Prompt options
    parser.add_argument('--prompt', type=str, default='free_ocr',
                        help='Prompt template name or custom prompt (default: free_ocr)')
    
    # Resolution options
    parser.add_argument('--preset', type=str, default='gundam',
                        choices=['tiny', 'small', 'base', 'large', 'gundam'],
                        help='Resolution preset (default: gundam)')
    parser.add_argument('--base-size', type=int, default=None,
                        help='Override base size (512/640/1024/1280)')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Override image size for cropping')
    parser.add_argument('--no-crop', action='store_true',
                        help='Disable crop mode (use native resolution)')
    
    # Anti-hallucination options
    parser.add_argument('--ngram', type=int, default=30,
                        help='N-gram size for repetition blocking (default: 30, range: 20-40)')
    parser.add_argument('--window', type=int, default=90,
                        help='Sliding window size (default: 90, range: 60-120)')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'adaptive'],
                        help='Processor mode: standard or adaptive (default: standard)')
    
    # Other options
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to file')
    parser.add_argument('--no-compress', action='store_true',
                        help='Disable compression testing')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get resolution settings
    if args.base_size or args.image_size or args.no_crop:
        # Manual override
        preset = RESOLUTION_PRESETS[args.preset]
        base_size = args.base_size or preset['base_size']
        image_size = args.image_size or preset['image_size']
        crop_mode = not args.no_crop and preset['crop_mode']
    else:
        # Use preset
        preset = RESOLUTION_PRESETS[args.preset]
        base_size = preset['base_size']
        image_size = preset['image_size']
        crop_mode = preset['crop_mode']
    
    # Get prompt
    if args.prompt in PROMPT_TEMPLATES:
        prompt = PROMPT_TEMPLATES[args.prompt]
    else:
        prompt = args.prompt
    
    # Load model
    model, tokenizer = setup_model(args.model, args.device)
    
    # Run inference
    result = infer_with_anti_hallucination(
        model=model,
        tokenizer=tokenizer,
        image_file=args.image,
        prompt=prompt,
        output_path=args.output,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=not args.no_save,
        test_compress=not args.no_compress,
        ngram_size=args.ngram,
        window_size=args.window,
        mode=args.mode
    )
    
    # Print result
    print("\n" + "="*80)
    print("OCR RESULT:")
    print("="*80)
    print(result)
    print("="*80)


if __name__ == "__main__":
    main()
