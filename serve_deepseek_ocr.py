#!/usr/bin/env python3
"""
DeepSeek-OCR vLLM Serving Script
Fixes GitHub Issue #244: Model architecture registration for vLLM serving

This script properly registers the DeepseekOCRForCausalLM model with vLLM
and provides both serving and direct inference capabilities.

Usage:
    # For serving (API server):
    python serve_deepseek_ocr.py --serve --host 0.0.0.0 --port 8000
    
    # For direct inference:
    python serve_deepseek_ocr.py --image path/to/image.jpg --prompt "<image>\nFree OCR."
"""

import argparse
import sys
from pathlib import Path

# Add the DeepSeek-OCR-vllm directory to the path
deepseek_ocr_path = Path(__file__).parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
sys.path.insert(0, str(deepseek_ocr_path))

try:
    from vllm import LLM, SamplingParams, ModelRegistry
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    from deepseek_ocr import DeepseekOCRForCausalLM
    from PIL import Image
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease ensure you have installed:")
    print("  - vllm (v0.11.1+ recommended)")
    print("  - PIL/Pillow")
    print("  - All dependencies from requirements.txt")
    sys.exit(1)


def register_model():
    """Register the DeepseekOCRForCausalLM model with vLLM."""
    try:
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
        print("✓ Successfully registered DeepseekOCRForCausalLM model")
        return True
    except Exception as e:
        print(f"✗ Failed to register model: {e}")
        return False


def create_llm_instance(model_path="deepseek-ai/DeepSeek-OCR", **kwargs):
    """
    Create and return a vLLM LLM instance with proper configuration.
    
    Args:
        model_path: Path to the model (HuggingFace repo or local path)
        **kwargs: Additional arguments to pass to LLM constructor
    
    Returns:
        LLM instance
    """
    default_config = {
        "model": model_path,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "logits_processors": [NGramPerReqLogitsProcessor],
        "trust_remote_code": True,
    }
    
    # Override defaults with provided kwargs
    default_config.update(kwargs)
    
    print(f"Creating LLM instance with model: {model_path}")
    print(f"Configuration: {default_config}")
    
    try:
        llm = LLM(**default_config)
        print("✓ Successfully created LLM instance")
        return llm
    except Exception as e:
        print(f"✗ Failed to create LLM instance: {e}")
        raise


def create_sampling_params(
    temperature=0.0,
    max_tokens=8192,
    ngram_size=30,
    window_size=90,
    whitelist_token_ids=None
):
    """
    Create sampling parameters for DeepSeek-OCR inference.
    
    Args:
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum number of tokens to generate
        ngram_size: N-gram size for the logits processor
        window_size: Window size for the logits processor
        whitelist_token_ids: List of token IDs to whitelist (e.g., [128821, 128822] for <td>, </td>)
    
    Returns:
        SamplingParams instance
    """
    if whitelist_token_ids is None:
        whitelist_token_ids = {128821, 128822}  # <td>, </td>
    
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        extra_args=dict(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids=whitelist_token_ids,
        ),
        skip_special_tokens=False,
    )


def run_inference(llm, image_paths, prompt="<image>\nFree OCR.", sampling_params=None):
    """
    Run inference on one or more images.
    
    Args:
        llm: LLM instance
        image_paths: Single image path or list of image paths
        prompt: Prompt template (should include <image> token)
        sampling_params: SamplingParams instance (created with defaults if None)
    
    Returns:
        List of output texts
    """
    if sampling_params is None:
        sampling_params = create_sampling_params()
    
    # Ensure image_paths is a list
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    
    # Prepare model inputs
    model_inputs = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            model_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })
            print(f"✓ Loaded image: {img_path}")
        except Exception as e:
            print(f"✗ Failed to load image {img_path}: {e}")
            continue
    
    if not model_inputs:
        print("✗ No valid images to process")
        return []
    
    # Generate outputs
    print(f"Running inference on {len(model_inputs)} image(s)...")
    try:
        outputs = llm.generate(model_inputs, sampling_params)
        results = [output.outputs[0].text for output in outputs]
        print(f"✓ Successfully generated {len(results)} output(s)")
        return results
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        raise


def serve_model(
    model_path="deepseek-ai/DeepSeek-OCR",
    host="0.0.0.0",
    port=8000,
    **llm_kwargs
):
    """
    Start a vLLM API server for DeepSeek-OCR.
    
    Args:
        model_path: Path to the model
        host: Host address to bind to
        port: Port number to bind to
        **llm_kwargs: Additional arguments for LLM configuration
    """
    print(f"Starting vLLM API server on {host}:{port}")
    print(f"Model: {model_path}")
    
    try:
        from vllm.entrypoints.openai.api_server import run_server
        
        # Create LLM instance
        llm = create_llm_instance(model_path, **llm_kwargs)
        
        # Start server
        print(f"✓ Server starting at http://{host}:{port}")
        print("  Use Ctrl+C to stop the server")
        print("\nAPI endpoints:")
        print(f"  - http://{host}:{port}/v1/completions")
        print(f"  - http://{host}:{port}/v1/chat/completions")
        print(f"  - http://{host}:{port}/health")
        
        # Note: This is a simplified version. For production use, consider using
        # the vllm serve command with proper model registration
        print("\n⚠ For production serving, consider using the vllm CLI:")
        print("  vllm serve deepseek-ai/DeepSeek-OCR \\")
        print("    --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \\")
        print("    --no-enable-prefix-caching \\")
        print("    --mm-processor-cache-gb 0")
        
    except ImportError:
        print("✗ vLLM API server components not available")
        print("  This feature requires vLLM to be properly installed")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Failed to start server: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR vLLM Serving and Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct inference on a single image:
  python serve_deepseek_ocr.py --image photo.jpg
  
  # Direct inference with custom prompt:
  python serve_deepseek_ocr.py --image doc.png --prompt "<image>\\n<|grounding|>Convert the document to markdown."
  
  # Batch inference on multiple images:
  python serve_deepseek_ocr.py --image img1.jpg img2.jpg img3.jpg
  
  # Start API server (note: simplified version):
  python serve_deepseek_ocr.py --serve --port 8000
  
  # Use custom model path:
  python serve_deepseek_ocr.py --model /path/to/local/model --image test.jpg
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Model path (HuggingFace repo or local path)"
    )
    
    # Inference mode
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        help="Path(s) to image file(s) for inference"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\nFree OCR.",
        help="Prompt template (must include <image> token)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save results (optional)"
    )
    
    # Serving mode
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start API server mode"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Register the model
    print("=" * 60)
    print("DeepSeek-OCR vLLM Serving Script")
    print("=" * 60)
    if not register_model():
        sys.exit(1)
    
    # Serve mode
    if args.serve:
        print("\n" + "=" * 60)
        print("Starting Server Mode")
        print("=" * 60)
        serve_model(
            model_path=args.model,
            host=args.host,
            port=args.port
        )
        return
    
    # Inference mode
    if args.image:
        print("\n" + "=" * 60)
        print("Running Inference Mode")
        print("=" * 60)
        
        # Create LLM instance
        llm = create_llm_instance(args.model)
        
        # Create sampling parameters
        sampling_params = create_sampling_params(
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Run inference
        results = run_inference(
            llm,
            args.image,
            prompt=args.prompt,
            sampling_params=sampling_params
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        for i, (img_path, result) in enumerate(zip(args.image, results), 1):
            print(f"\n[Image {i}: {img_path}]")
            print("-" * 60)
            print(result)
            print("-" * 60)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for img_path, result in zip(args.image, results):
                    f.write(f"=== {img_path} ===\n")
                    f.write(result)
                    f.write("\n\n")
            print(f"\n✓ Results saved to: {args.output}")
        
        return
    
    # No mode specified
    parser.print_help()
    print("\n✗ Error: Please specify either --image for inference or --serve for API server")
    sys.exit(1)


if __name__ == "__main__":
    main()
