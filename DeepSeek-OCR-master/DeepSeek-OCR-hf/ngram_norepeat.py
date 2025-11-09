"""
NoRepeatNGramLogitsProcessor - Prevents repetitive hallucinations in OCR output

This logits processor prevents the model from generating repetitive n-grams,
which is crucial for preventing hallucinations in OCR tasks, especially with
challenging documents like handwritten text.

Usage:
    from transformers import LogitsProcessorList
    from ngram_norepeat import NoRepeatNGramLogitsProcessor
    
    logits_processor = LogitsProcessorList([
        NoRepeatNGramLogitsProcessor(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822}  # <td>, </td>
        )
    ])
"""

import torch
from transformers import LogitsProcessor
from typing import List, Set


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Logits processor that prevents repetition of n-grams within a sliding window.
    
    This processor scans the generated sequence within a sliding window and bans
    tokens that would complete n-grams that have already appeared in that window.
    Certain tokens can be whitelisted to allow repetition (e.g., table delimiters).
    
    Args:
        ngram_size (int): Size of n-grams to track (e.g., 30 means sequences of 30 tokens)
        window_size (int): Size of the sliding window to check for repetitions (default: 100)
        whitelist_token_ids (set): Set of token IDs that are allowed to repeat (default: None)
    
    Example:
        >>> processor = NoRepeatNGramLogitsProcessor(
        ...     ngram_size=30,
        ...     window_size=90,
        ...     whitelist_token_ids={128821, 128822}
        ... )
    """

    def __init__(self, ngram_size: int, window_size: int = 100, whitelist_token_ids: set = None):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to ban tokens that would create repeated n-grams.
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length) containing generated token IDs
            scores: Tensor of shape (batch_size, vocab_size) containing logits for next token
            
        Returns:
            Modified scores with banned tokens set to -inf
        """
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            # Get the sequence for this batch item
            sequence = input_ids[batch_idx].tolist()
            
            # Skip if sequence is too short
            if len(sequence) < self.ngram_size:
                continue
            
            # Get the current prefix (last ngram_size - 1 tokens)
            current_prefix = tuple(sequence[-(self.ngram_size - 1):])
            
            # Define the search window
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
            
            # Apply bans by setting scores to -inf
            if banned_tokens:
                for token in banned_tokens:
                    scores[batch_idx, token] = float("-inf")
        
        return scores
