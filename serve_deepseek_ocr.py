#!/usr/bin/env python3
"""
DeepSeek-OCR vLLM Serving Script

This script properly registers the DeepSeek-OCR model and starts a vLLM server
with the correct configuration for OCR tasks.

Usage:
    python serve_deepseek_ocr.py [options]

Options:
    --model MODEL_PATH          Path or HuggingFace model ID (default: deepseek-ai/DeepSeek-OCR)
    --host HOST                 Server host (default: 0.0.0.0)
    --port PORT                 Server port (default: 8000)
    --gpu-memory-utilization    GPU memory utilization (default: 0.9)
    --max-model-len            Maximum model length (default: 8192)
    --tensor-parallel-size     Tensor parallel size (default: 1)

Example:
    python serve_deepseek_ocr.py --model deepseek-ai/DeepSeek-OCR --port 8000
"""

import argparse
import sys
import os

# Add the DeepSeek-OCR-vllm directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
vllm_dir = os.path.join(current_dir, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")
if os.path.exists(vllm_dir):
    sys.path.insert(0, vllm_dir)

# Set environment variables for optimal performance
os.environ['VLLM_USE_V1'] = '0'

# Register the model first
import register_deepseek_ocr

from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve DeepSeek-OCR model with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code from HuggingFace"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("DeepSeek-OCR vLLM Server")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Max Model Length: {args.max_model_len}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print("=" * 60)
    
    # Import vLLM serve command
    try:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.entrypoints.openai.api_server import run_server as vllm_run_server
        from vllm.utils import FlexibleArgumentParser
        
        # Build command line arguments for vLLM
        vllm_args = [
            "--model", args.model,
            "--host", args.host,
            "--port", str(args.port),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--trust-remote-code",
            "--disable-log-requests",
            # OCR-specific settings
            "--enable-prefix-caching", "false",
            "--mm-processor-cache-gb", "0",
        ]
        
        # Override architecture in config
        os.environ["VLLM_OVERRIDE_ARCHITECTURES"] = "DeepseekOCRForCausalLM"
        
        print("\nStarting vLLM server...")
        print("Note: The server will be available at the OpenAI-compatible API endpoint")
        print(f"      API Base URL: http://{args.host}:{args.port}/v1")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Run vLLM serve command programmatically
        import subprocess
        cmd = [
            "vllm", "serve", args.model,
            "--host", args.host,
            "--port", str(args.port),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--tensor-parallel-size", str(args.tensor_parallel_size),
            "--trust-remote-code",
            "--enable-prefix-caching", "false",
            "--mm-processor-cache-gb", "0",
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"\nError starting server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure vLLM is installed: pip install vllm")
        print("2. Ensure the model files are accessible")
        print("3. Check that you have sufficient GPU memory")
        print("4. Try using the fix_model_config.py script if architecture errors persist")
        sys.exit(1)


if __name__ == "__main__":
    main()
