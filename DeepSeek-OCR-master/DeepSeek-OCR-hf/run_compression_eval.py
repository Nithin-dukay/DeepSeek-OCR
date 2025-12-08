#!/usr/bin/env python3
"""
Fox Dataset Compression Evaluation Script

This script evaluates the DeepSeek-OCR model on the Fox dataset with different
resolution modes to measure compression efficiency.

Usage:
    python run_compression_eval.py --dataset_path /path/to/fox --output_dir ./results
    
    python run_compression_eval.py --dataset_path /path/to/fox --mode Gundam
    
    python run_compression_eval.py --help

GitHub Issue #285: Query about Compression Study Evaluation
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

# Import compression utilities
from compression_utils import (
    patch_infer_method,
    evaluate_compression_batch,
    save_compression_results,
    print_compression_summary,
    calculate_average_compression
)


# Resolution modes configuration
RESOLUTION_MODES = {
    "Tiny": {
        "base_size": 512,
        "image_size": 512,
        "crop_mode": False,
        "description": "512×512 native resolution (64 vision tokens)"
    },
    "Small": {
        "base_size": 640,
        "image_size": 640,
        "crop_mode": False,
        "description": "640×640 native resolution (100 vision tokens)"
    },
    "Base": {
        "base_size": 1024,
        "image_size": 1024,
        "crop_mode": False,
        "description": "1024×1024 native resolution (256 vision tokens)"
    },
    "Large": {
        "base_size": 1280,
        "image_size": 1280,
        "crop_mode": False,
        "description": "1280×1280 native resolution (400 vision tokens)"
    },
    "Gundam": {
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "description": "Dynamic resolution: n×640×640 + 1×1024×1024"
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-OCR compression on Fox dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all modes on Fox dataset
  python run_compression_eval.py --dataset_path /data/fox --output_dir ./results
  
  # Evaluate only Gundam mode
  python run_compression_eval.py --dataset_path /data/fox --mode Gundam
  
  # Evaluate with custom prompt
  python run_compression_eval.py --dataset_path /data/fox --prompt "<image>\\nFree OCR."
  
  # Evaluate specific images
  python run_compression_eval.py --image_files img1.jpg img2.jpg --mode Base
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset_path",
        type=str,
        help="Path to Fox dataset directory (will search for all .jpg and .png files)"
    )
    input_group.add_argument(
        "--image_files",
        type=str,
        nargs="+",
        help="List of specific image files to evaluate"
    )
    
    # Model options
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="HuggingFace model name (default: deepseek-ai/DeepSeek-OCR)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device ID (default: 0)"
    )
    
    # Resolution mode options
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(RESOLUTION_MODES.keys()) + ["all"],
        default="all",
        help="Resolution mode to evaluate (default: all)"
    )
    
    # Inference options
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\n<|grounding|>Convert the document to markdown.",
        help="Prompt template for inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./compression_results",
        help="Output directory for results (default: ./compression_results)"
    )
    
    # Processing options
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images that already have results"
    )
    
    return parser.parse_args()


def find_images(dataset_path):
    """Find all image files in the dataset directory."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(dataset_path, "**", ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    return image_files


def load_model(model_name, device):
    """Load and prepare the DeepSeek-OCR model."""
    print(f"\nLoading model: {model_name}")
    print(f"Device: cuda:{device}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    
    # Move to GPU and set to eval mode
    print("Moving model to GPU...")
    model = model.eval().cuda().to(torch.bfloat16)
    
    # Apply compression patch
    print("Applying compression patch...")
    model = patch_infer_method(model)
    
    print("Model loaded successfully!\n")
    
    return model, tokenizer


def evaluate_mode(model, tokenizer, image_files, mode_name, mode_config, 
                 prompt, output_dir):
    """Evaluate a single resolution mode."""
    print("\n" + "=" * 80)
    print(f"Evaluating Mode: {mode_name}")
    print(f"Description: {mode_config['description']}")
    print(f"Configuration:")
    print(f"  - base_size: {mode_config['base_size']}")
    print(f"  - image_size: {mode_config['image_size']}")
    print(f"  - crop_mode: {mode_config['crop_mode']}")
    print(f"Number of images: {len(image_files)}")
    print("=" * 80)
    
    # Create output directory for this mode
    mode_output_dir = os.path.join(output_dir, mode_name)
    Path(mode_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    results = evaluate_compression_batch(
        model,
        tokenizer,
        image_files,
        prompt=prompt,
        output_dir=mode_output_dir,
        base_size=mode_config['base_size'],
        image_size=mode_config['image_size'],
        crop_mode=mode_config['crop_mode']
    )
    
    # Save results
    json_file = os.path.join(output_dir, f"results_{mode_name}.json")
    csv_file = os.path.join(output_dir, f"results_{mode_name}.csv")
    
    save_compression_results(results, json_file, csv_file)
    
    # Print summary
    print_compression_summary(results)
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 80)
    print("DeepSeek-OCR Compression Evaluation")
    print("Fox Dataset Evaluation Script")
    print("=" * 80)
    
    # Get image files
    if args.dataset_path:
        print(f"\nSearching for images in: {args.dataset_path}")
        image_files = find_images(args.dataset_path)
        print(f"Found {len(image_files)} images")
    else:
        image_files = args.image_files
        print(f"\nEvaluating {len(image_files)} specified images")
    
    if not image_files:
        print("Error: No images found!")
        sys.exit(1)
    
    # Limit number of images if specified
    if args.max_images:
        image_files = image_files[:args.max_images]
        print(f"Limited to {len(image_files)} images")
    
    # Print sample of images
    print("\nSample images:")
    for img in image_files[:5]:
        print(f"  - {img}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more")
    
    # Load model
    try:
        model, tokenizer = load_model(args.model_name, args.device)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nMake sure you have:")
        print("  1. Installed all requirements: pip install -r requirements.txt")
        print("  2. Installed flash-attn: pip install flash-attn --no-build-isolation")
        print("  3. CUDA is available and properly configured")
        sys.exit(1)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine which modes to evaluate
    if args.mode == "all":
        modes_to_eval = list(RESOLUTION_MODES.keys())
    else:
        modes_to_eval = [args.mode]
    
    print(f"\nModes to evaluate: {', '.join(modes_to_eval)}")
    print(f"Prompt: {args.prompt}")
    print(f"Output directory: {args.output_dir}")
    
    # Evaluate each mode
    all_results = {}
    
    for mode_name in modes_to_eval:
        mode_config = RESOLUTION_MODES[mode_name]
        
        try:
            results = evaluate_mode(
                model,
                tokenizer,
                image_files,
                mode_name,
                mode_config,
                args.prompt,
                args.output_dir
            )
            all_results[mode_name] = results
            
            # Clear CUDA cache between modes
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError evaluating mode {mode_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final comparison
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("FINAL COMPARISON ACROSS MODES")
        print("=" * 80)
        
        comparison_data = []
        for mode_name, results in all_results.items():
            avg_stats = calculate_average_compression(results)
            if avg_stats:
                comparison_data.append({
                    'mode': mode_name,
                    'avg_image_tokens': avg_stats['avg_image_tokens'],
                    'avg_output_tokens': avg_stats['avg_output_tokens'],
                    'avg_compression_ratio': avg_stats['avg_compression_ratio']
                })
        
        # Print comparison table
        print(f"\n{'Mode':<10} {'Avg Image Tokens':<20} {'Avg Output Tokens':<20} {'Avg Compression Ratio':<25}")
        print("-" * 80)
        for data in comparison_data:
            print(f"{data['mode']:<10} {data['avg_image_tokens']:<20.2f} "
                  f"{data['avg_output_tokens']:<20.2f} {data['avg_compression_ratio']:<25.2f}")
        
        print("=" * 80)
    
    print(f"\n✓ Evaluation complete! Results saved to: {args.output_dir}")
    print("\nFiles generated:")
    print(f"  - JSON results: results_<mode>.json")
    print(f"  - CSV summary: results_<mode>.csv")
    print(f"  - Output files: {args.output_dir}/<mode>/")


if __name__ == "__main__":
    main()
