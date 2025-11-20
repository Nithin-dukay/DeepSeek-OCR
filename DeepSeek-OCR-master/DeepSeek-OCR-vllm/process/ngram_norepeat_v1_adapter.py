"""
vLLM v1 Engine Adapter for NoRepeatNGramLogitsProcessor.

This adapter enables DeepSeek-OCR's n-gram no-repeat logits processor
to work with vLLM 0.11.0's new v1 engine architecture.
"""

from .ngram_norepeat import NoRepeatNGramLogitsProcessor
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class NoRepeatNGramAdaptor(AdapterLogitsProcessor):
    """
    Adapter for NoRepeatNGramLogitsProcessor to work with vLLM v1 engine.
    
    The v1 engine uses global logits processors via the AdapterLogitsProcessor
    interface, which creates per-request processors from sampling parameters.
    """
    
    def is_argmax_invariant(self) -> bool:
        """
        Indicates whether this processor affects argmax sampling.
        
        Returns:
            True since n-gram filtering can change the argmax token.
        """
        return True

    def new_req_logits_processor(self, params):
        """
        Create a new per-request logits processor instance.
        
        Args:
            params: Sampling parameters containing extra_args with:
                - ngram_size: Size of n-grams to check for repetition
                - window_size: Window size for checking repetitions
                - whitelist_token_ids: Set of token IDs to exclude from filtering
        
        Returns:
            NoRepeatNGramLogitsProcessor instance configured with the parameters.
        """
        return NoRepeatNGramLogitsProcessor(
            ngram_size=params.extra_args["ngram_size"],
            window_size=params.extra_args["window_size"],
            whitelist_token_ids=params.extra_args["whitelist_token_ids"],
        )
