"""
DeepSeek-OCR vLLM Plugin

This module registers the DeepseekOCRForCausalLM model with vLLM's ModelRegistry,
allowing it to be used with the `vllm serve` command.
"""

from vllm.model_executor.models.registry import ModelRegistry
from .deepseek_ocr import DeepseekOCRForCausalLM
from .process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Register the model with vLLM
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Export for external use
__all__ = [
    "DeepseekOCRForCausalLM",
    "NoRepeatNGramLogitsProcessor",
]
