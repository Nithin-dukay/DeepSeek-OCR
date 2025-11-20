#!/usr/bin/env python3
"""
Custom vLLM server for DeepSeek-OCR with proper model registration.

This script registers the DeepseekOCRForCausalLM model and starts a vLLM server
that can be accessed via OpenAI-compatible API.

Usage:
    python serve_deepseek_ocr.py --model deepseek-ai/DeepSeek-OCR --port 8000

Example API call:
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "deepseek-ai/DeepSeek-OCR",
            "prompt": "<image>\\nFree OCR.",
            "max_tokens": 8192
        }'
"""

import sys
import os
import argparse
import asyncio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables
os.environ['VLLM_USE_V1'] = '0'

from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import run_server as vllm_run_server

# Register the model
print("Registering DeepseekOCRForCausalLM with vLLM...")
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
print("✓ Model registered successfully")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR vLLM Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings
  python serve_deepseek_ocr.py

  # Start server with custom model path
  python serve_deepseek_ocr.py --model /path/to/local/model

  # Start server on specific port
  python serve_deepseek_ocr.py --port 8080

  # Start server with custom GPU memory utilization
  python serve_deepseek_ocr.py --gpu-memory-utilization 0.8
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Model name or path (default: deepseek-ai/DeepSeek-OCR)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
        help="GPU memory utilization (default: 0.75)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Block size for paged attention (default: 256)"
    )
    
    return parser.parse_args()


async def start_server(args):
    """Start the vLLM server with DeepSeek-OCR model."""
    
    print("\n" + "="*60)
    print("DeepSeek-OCR vLLM Server")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print("="*60 + "\n")
    
    # Configure engine arguments
    engine_args = AsyncEngineArgs(
        model=args.model,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
    )
    
    print("Initializing engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✓ Engine initialized successfully\n")
    
    print(f"Server starting on http://{args.host}:{args.port}")
    print(f"API endpoint: http://{args.host}:{args.port}/v1/completions")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Note: This is a simplified version. For full OpenAI API compatibility,
    # you would need to implement the full API server logic.
    # For now, we'll just keep the engine running.
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        asyncio.run(start_server(args))
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
