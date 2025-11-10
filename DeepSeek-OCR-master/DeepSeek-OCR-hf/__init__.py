"""
DeepSeek-OCR Transformers Integration

This package provides a clean interface to use DeepSeek-OCR with Transformers
without warnings.

Usage:
    from DeepSeek_OCR_hf import DeepSeekOCRForCausalLM, DeepSeekOCRConfig
    from transformers import AutoTokenizer
    
    model = DeepSeekOCRForCausalLM.from_pretrained('deepseek-ai/DeepSeek-OCR')
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR')
"""

from .configuration_deepseek_ocr import DeepSeekOCRConfig
from .modeling_deepseek_ocr import DeepSeekOCRForCausalLM

__version__ = "1.0.0"
__all__ = ["DeepSeekOCRConfig", "DeepSeekOCRForCausalLM"]
