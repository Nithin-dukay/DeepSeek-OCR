"""
NoRepeatNGramLogitsProcessor for Transformers
This processor prevents hallucinations by blocking repeated n-grams during generation.
Compatible with Transformers' generation API.
"""

import torch
from transformers import LogitsProcessor
from typing import List, Set, Optional


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Logits processor that prevents the generation of repeated n-grams.
    
    This is crucial for OCR tasks to prevent hallucinations, especially with:
    - Handwritten documents
    - Ancient or degraded text
    - Complex layouts
    
    Args:
        ngram_size (int): Size of n-grams to track. Larger values are more strict.
            - For handwritten/complex documents: 30-40
            - For printed documents: 20-30
        window_size (int): How far back to look for repeated n-grams.
            - For handwritten/complex documents: 90
            - For printed documents: 50-90
        whitelist_token_ids (set, optional): Token IDs that are allowed to repeat.
            - For tables: {128821, 128822} for <td> and </td> tags
    """

    def __init__(
        self, 
        ngram_size: int, 
        window_size: int = 100, 
        whitelist_token_ids: Optional[Set[int]] = None
    ):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` must be a strictly positive integer, but got {ngram_size}"
            )
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(
                f"`window_size` must be a strictly positive integer, but got {window_size}"
            )
        
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to prevent n-gram repetition.
        
        Args:
            input_ids: Token IDs generated so far. Shape: (batch_size, seq_len)
            scores: Logits for next token. Shape: (batch_size, vocab_size)
            
        Returns:
            Modified scores with banned tokens set to -inf
        """
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[-1]
        
        # Process each sequence in the batch
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()
            
            # Need at least ngram_size tokens to check for repetition
            if len(sequence) < self.ngram_size:
                continue
            
            # Get the current prefix (last ngram_size - 1 tokens)
            current_prefix = tuple(sequence[-(self.ngram_size - 1):])
            
            # Define search window
            search_start = max(0, len(sequence) - self.window_size)
            search_end = len(sequence) - self.ngram_size + 1
            
            # Find all tokens that would complete a repeated n-gram
            banned_tokens = set()
            for i in range(search_start, search_end):
                ngram = tuple(sequence[i:i + self.ngram_size])
                if ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])
            
            # Remove whitelisted tokens from banned set
            banned_tokens = banned_tokens - self.whitelist_token_ids
            
            # Apply banning by setting scores to -inf
            if banned_tokens:
                for token_id in banned_tokens:
                    if token_id < vocab_size:
                        scores[batch_idx, token_id] = float('-inf')
        
        return scores


def get_recommended_params(document_type: str = "general") -> dict:
    """
    Get recommended parameters for different document types.
    
    Args:
        document_type: Type of document being processed
            - "handwritten": Ancient or handwritten documents
            - "printed": Modern printed documents
            - "table": Documents with tables
            - "general": General purpose (default)
            
    Returns:
        Dictionary with recommended ngram_size, window_size, and whitelist_token_ids
    """
    params = {
        "handwritten": {
            "ngram_size": 35,
            "window_size": 90,
            "whitelist_token_ids": set(),
        },
        "printed": {
            "ngram_size": 25,
            "window_size": 70,
            "whitelist_token_ids": set(),
        },
        "table": {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},  # <td>, </td>
        },
        "general": {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": set(),
        },
    }
    
    return params.get(document_type, params["general"])
