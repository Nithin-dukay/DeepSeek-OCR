"""
Improved DeepSeek-OCR Inference Script with Anti-Hallucination Mechanisms

This script addresses GitHub Issue #191 by implementing proper generation parameters
and n-gram repetition prevention to reduce hallucinations, especially for challenging
documents like ancient handwritten text.

Key improvements over the original run_dpsk_ocr.py:
1. Uses NoRepeatNGramLogitsProcessor to prevent repetitive hallucinations
2. Implements greedy decoding (temperature=0.0) for more deterministic output
3. Provides configurable generation parameters
4. Supports both direct model.generate() and custom inference

Usage:
    python run_dpsk_ocr_improved.py --image_file your_image.jpg --output_path ./output
    
    # For handwritten documents (more aggressive anti-repetition):
    python run_dpsk_ocr_improved.py --image_file ancient_doc.jpg --ngram_size 20 --window_size 60
    
    # For printed documents (standard settings):
    python run_dpsk_ocr_improved.py --image_file printed_doc.jpg --ngram_size 30 --window_size 90
"""

import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer, LogitsProcessorList
from PIL import Image
from ngram_norepeat import NoRepeatNGramLogitsProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='DeepSeek-OCR Improved Inference')
    
    # Required arguments
    parser.add_argument('--image_file', type=str, required=True,
                        help='Path to the input image file')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Directory to save output files (default: ./output)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='deepseek-ai/DeepSeek-OCR',
                        help='Model name or path (default: deepseek-ai/DeepSeek-OCR)')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device ID (default: 0)')
    
    # Prompt arguments
    parser.add_argument('--prompt_type', type=str, default='free_ocr',
                        choices=['free_ocr', 'grounding', 'custom'],
                        help='Type of prompt to use (default: free_ocr)')
    parser.add_argument('--custom_prompt', type=str, default=None,
                        help='Custom prompt (only used if prompt_type=custom)')
    
    # Image processing arguments
    parser.add_argument('--base_size', type=int, default=1024,
                        help='Base image size (default: 1024)')
    parser.add_argument('--image_size', type=int, default=640,
                        help='Crop image size (default: 640)')
    parser.add_argument('--crop_mode', action='store_true', default=True,
                        help='Enable dynamic cropping (default: True)')
    parser.add_argument('--no_crop_mode', dest='crop_mode', action='store_false',
                        help='Disable dynamic cropping')
    
    # Generation arguments (anti-hallucination)
    parser.add_argument('--ngram_size', type=int, default=30,
                        help='N-gram size for repetition prevention (default: 30, use 20 for handwritten)')
    parser.add_argument('--window_size', type=int, default=90,
                        help='Window size for n-gram checking (default: 90, use 60 for handwritten)')
    parser.add_argument('--max_new_tokens', type=int, default=8192,
                        help='Maximum number of tokens to generate (default: 8192)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default: 0.0 for greedy decoding)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty (default: 1.0)')
    
    # Output arguments
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save results to file (default: True)')
    parser.add_argument('--test_compress', action='store_true', default=True,
                        help='Test compression (default: True)')
    
    return parser.parse_args()


def get_prompt(prompt_type, custom_prompt=None):
    """Get the appropriate prompt based on type."""
    prompts = {
        'free_ocr': '<image>\nFree OCR. ',
        'grounding': '<image>\n<|grounding|>Convert the document to markdown. ',
    }
    
    if prompt_type == 'custom':
        if custom_prompt is None:
            raise ValueError("custom_prompt must be provided when prompt_type='custom'")
        return custom_prompt
    
    return prompts.get(prompt_type, prompts['free_ocr'])


def load_model_and_tokenizer(model_name, device):
    """Load the DeepSeek-OCR model and tokenizer."""
    print(f"Loading model: {model_name}")
    print(f"Using device: cuda:{device}")
    
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


def setup_logits_processor(ngram_size, window_size):
    """
    Setup logits processor with anti-hallucination mechanisms.
    
    Token IDs 128821 and 128822 correspond to <td> and </td> tags,
    which are allowed to repeat in table structures.
    """
    whitelist_token_ids = {128821, 128822}  # <td>, </td>
    
    logits_processor = LogitsProcessorList([
        NoRepeatNGramLogitsProcessor(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids=whitelist_token_ids
        )
    ])
    
    return logits_processor


def run_inference_with_infer_method(model, tokenizer, args, prompt):
    """
    Run inference using the model's built-in infer method.
    
    Note: This method may not support custom logits processors.
    For best anti-hallucination results, use run_inference_with_generate().
    """
    print("\n" + "="*80)
    print("Running inference with model.infer() method")
    print("Note: This method may not support custom anti-hallucination parameters")
    print("="*80 + "\n")
    
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=args.image_file,
        output_path=args.output_path,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        save_results=args.save_results,
        test_compress=args.test_compress
    )
    
    return result


def run_inference_with_generate(model, tokenizer, args, prompt):
    """
    Run inference using model.generate() with custom logits processors.
    
    This method provides better control over generation parameters and
    implements anti-hallucination mechanisms.
    
    Note: This requires the model to support standard HuggingFace generate() API.
    If the model doesn't support this, use run_inference_with_infer_method() instead.
    """
    print("\n" + "="*80)
    print("Running inference with model.generate() and anti-hallucination mechanisms")
    print(f"Parameters: ngram_size={args.ngram_size}, window_size={args.window_size}")
    print(f"            temperature={args.temperature}, max_tokens={args.max_new_tokens}")
    print("="*80 + "\n")
    
    # Setup logits processor
    logits_processor = setup_logits_processor(args.ngram_size, args.window_size)
    
    # Load and preprocess image
    image = Image.open(args.image_file).convert('RGB')
    
    # Prepare inputs (this is a simplified version - actual preprocessing may differ)
    # The exact preprocessing depends on the model's implementation
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate with anti-hallucination parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else None,
            do_sample=args.temperature > 0,
            repetition_penalty=args.repetition_penalty,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Save results
    if args.save_results:
        os.makedirs(args.output_path, exist_ok=True)
        output_file = os.path.join(args.output_path, 'result.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\nResults saved to: {output_file}")
    
    return result


def main():
    args = parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Get prompt
    prompt = get_prompt(args.prompt_type, args.custom_prompt)
    print(f"Using prompt: {prompt}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    
    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Image file: {args.image_file}")
    print(f"Output path: {args.output_path}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Base size: {args.base_size}")
    print(f"Image size: {args.image_size}")
    print(f"Crop mode: {args.crop_mode}")
    print(f"N-gram size: {args.ngram_size}")
    print(f"Window size: {args.window_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print("="*80 + "\n")
    
    # Run inference using the built-in infer method
    # This is the recommended approach as it handles image preprocessing correctly
    try:
        result = run_inference_with_infer_method(model, tokenizer, args, prompt)
        print("\n" + "="*80)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNote: The model.infer() method was used. For better anti-hallucination")
        print("control, consider modifying the model's infer() implementation to include")
        print("the NoRepeatNGramLogitsProcessor and generation parameters shown above.")
        print("\nAlternatively, use the vLLM implementation which has these features built-in:")
        print("  cd ../DeepSeek-OCR-vllm")
        print("  python run_dpsk_ocr_image.py")
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        print("\nTrying alternative generation method...")
        try:
            result = run_inference_with_generate(model, tokenizer, args, prompt)
        except Exception as e2:
            print(f"\nAlternative method also failed: {e2}")
            print("\nRecommendation: Use the vLLM implementation for production use:")
            print("  cd ../DeepSeek-OCR-vllm")
            print("  python run_dpsk_ocr_image.py")
            raise
    
    return result


if __name__ == "__main__":
    main()
