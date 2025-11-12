"""
Model registration module for DeepSeek-OCR.

This module automatically registers the DeepseekOCRForCausalLM model with vLLM's ModelRegistry
when imported. This is necessary because the model's config.json may have a typo in the
architecture name (DeepseekOCRForCausallM instead of DeepseekOCRForCausalLM).

Usage:
    Simply import this module before using vLLM with DeepSeek-OCR:
    
    >>> import register_deepseek_ocr
    >>> from vllm import LLM
    >>> llm = LLM(model="deepseek-ai/DeepSeek-OCR")
"""

import sys
import os

# Add the DeepSeek-OCR-vllm directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
vllm_dir = os.path.join(current_dir, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")
if os.path.exists(vllm_dir):
    sys.path.insert(0, vllm_dir)

try:
    from vllm.model_executor.models.registry import ModelRegistry
    from deepseek_ocr import DeepseekOCRForCausalLM
    
    # Register the model with both possible architecture names to handle the typo
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
    # Also register with the typo version in case it's in the config
    ModelRegistry.register_model("DeepseekOCRForCausallM", DeepseekOCRForCausalLM)
    
    print("✓ DeepSeek-OCR model registered successfully with vLLM")
    
except ImportError as e:
    print(f"Warning: Could not register DeepSeek-OCR model: {e}")
    print("Make sure vLLM is installed and the deepseek_ocr module is available")
except Exception as e:
    print(f"Error registering DeepSeek-OCR model: {e}")
