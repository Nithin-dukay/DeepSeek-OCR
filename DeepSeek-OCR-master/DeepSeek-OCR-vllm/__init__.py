"""
DeepSeek-OCR vLLM Implementation

This module provides the vLLM-based implementation of DeepSeek-OCR
for high-performance batch inference.
"""

from .deepseek_ocr import DeepseekOCRForCausalLM
from .config import *

__all__ = [
    "DeepseekOCRForCausalLM",
]
