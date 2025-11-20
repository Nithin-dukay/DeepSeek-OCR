"""
Model Registration Script for DeepSeek-OCR with vLLM

This script registers the DeepseekOCRForCausalLM model with vLLM's ModelRegistry.
Run this before using `vllm serve` command.

Usage:
    python register_model.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM

def register_deepseek_ocr():
    """Register DeepseekOCRForCausalLM with vLLM's ModelRegistry."""
    try:
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
        print("✓ Successfully registered DeepseekOCRForCausalLM with vLLM")
        return True
    except Exception as e:
        print(f"✗ Failed to register model: {e}")
        return False

if __name__ == "__main__":
    success = register_deepseek_ocr()
    sys.exit(0 if success else 1)
