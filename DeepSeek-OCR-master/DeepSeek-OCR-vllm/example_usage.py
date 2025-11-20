#!/usr/bin/env python3
"""
Example usage of DeepSeek-OCR with vLLM (Fix for Issue #244)

This script demonstrates the correct way to use DeepSeek-OCR with vLLM,
including proper model registration.

Usage:
    python example_usage.py --image path/to/image.png
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from PIL import Image

# Register the model with vLLM
print("Registering DeepseekOCRForCausalLM with vLLM...")
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
print("✓ Model registered successfully\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR Example Usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic OCR
  python example_usage.py --image document.png

  # OCR with markdown conversion
  python example_usage.py --image document.png --prompt "<image>\\n<|grounding|>Convert the document to markdown."

  # Use local model
  python example_usage.py --image document.png --model /path/to/local/model
        """
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Model name or path (default: deepseek-ai/DeepSeek-OCR)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\\nFree OCR.",
        help="Prompt template (default: '<image>\\nFree OCR.')"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional, prints to stdout if not specified)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("="*60)
    print("DeepSeek-OCR with vLLM - Example Usage")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print("="*60 + "\n")
    
    # Load image
    print("Loading image...")
    try:
        image = Image.open(args.image).convert("RGB")
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]} pixels\n")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        sys.exit(1)
    
    # Create LLM instance
    print("Initializing model...")
    try:
        llm = LLM(
            model=args.model,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            trust_remote_code=True,
            logits_processors=[NoRepeatNGramLogitsProcessor],
            gpu_memory_utilization=0.75,
        )
        print("✓ Model initialized\n")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Prepare input
    model_input = [{
        "prompt": args.prompt,
        "multi_modal_data": {"image": image}
    }]
    
    # Configure sampling
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # <td>, </td>
        ),
        skip_special_tokens=False,
    )
    
    # Generate output
    print("Generating output...")
    print("-"*60)
    try:
        outputs = llm.generate(model_input, sampling_params)
        result = outputs[0].outputs[0].text
        
        print(result)
        print("-"*60)
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n✓ Output saved to: {args.output}")
        
        print("\n✓ Generation completed successfully")
        
    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
