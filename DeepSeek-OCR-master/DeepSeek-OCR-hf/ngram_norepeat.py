"""
N-gram Repetition Prevention for HuggingFace Transformers

This module provides the NoRepeatNGramLogitsProcessor for use with
HuggingFace transformers generation to prevent infinite loops and
excessive repetition in generated text.
"""

import torch
from transformers import LogitsProcessor
from typing import List, Set, Optional


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Enhanced N-gram repetition prevention processor for HuggingFace transformers.
    
    This processor prevents the model from generating repetitive patterns by:
    1. Standard n-gram blocking (prevents exact n-gram repetition)
    2. Short pattern detection (prevents single token loops like '. . . .')
    3. Consecutive token blocking (prevents immediate repetition)
    
    Args:
        ngram_size: Size of n-grams to track for repetition
        window_size: How far back to look for repeated n-grams
        whitelist_token_ids: Token IDs that are allowed to repeat (e.g., table tags)
        min_ngram_size: Minimum n-gram size for short pattern detection (default: 2)
        max_consecutive_repeats: Maximum allowed consecutive identical tokens (default: 3)
    """

    def __init__(
        self, 
        ngram_size: int, 
        window_size: int = 100, 
        whitelist_token_ids: Optional[set] = None,
        min_ngram_size: int = 2,
        max_consecutive_repeats: int = 3
    ):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        if not isinstance(min_ngram_size, int) or min_ngram_size <= 0:
            raise ValueError(f"`min_ngram_size` has to be a strictly positive integer, but is {min_ngram_size}")
        if not isinstance(max_consecutive_repeats, int) or max_consecutive_repeats <= 0:
            raise ValueError(f"`max_consecutive_repeats` has to be a strictly positive integer, but is {max_consecutive_repeats}")
            
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
        self.min_ngram_size = min_ngram_size
        self.max_consecutive_repeats = max_consecutive_repeats
    
    def _detect_consecutive_repeats(self, input_ids: torch.LongTensor) -> Set[int]:
        """Detect tokens that have been repeated consecutively too many times."""
        banned = set()
        if input_ids.shape[-1] < self.max_consecutive_repeats:
            return banned
        
        # Check if the last N tokens are all the same
        recent_tokens = input_ids[0, -self.max_consecutive_repeats:].tolist()
        if len(set(recent_tokens)) == 1:
            # All recent tokens are identical, ban this token
            token = recent_tokens[0]
            if token not in self.whitelist_token_ids:
                banned.add(token)
        
        return banned
    
    def _detect_short_patterns(self, input_ids: torch.LongTensor) -> Set[int]:
        """Detect short repetitive patterns (e.g., 'A B A B A B')."""
        banned = set()
        input_ids_list = input_ids[0].tolist()
        
        # Check for 2-gram patterns (most common for dots: '. . . .')
        if len(input_ids_list) >= self.min_ngram_size * 3:
            for pattern_size in range(self.min_ngram_size, min(5, len(input_ids_list) // 3)):
                # Get the last pattern
                last_pattern = tuple(input_ids_list[-pattern_size:])
                
                # Check if this pattern repeats multiple times at the end
                repeat_count = 0
                for i in range(len(input_ids_list) - pattern_size, -1, -pattern_size):
                    if i < 0:
                        break
                    pattern = tuple(input_ids_list[i:i + pattern_size])
                    if pattern == last_pattern:
                        repeat_count += 1
                    else:
                        break
                
                # If pattern repeats 3+ times, ban the next token in the pattern
                if repeat_count >= 3:
                    # Ban all tokens in the pattern
                    for token in last_pattern:
                        if token not in self.whitelist_token_ids:
                            banned.add(token)
        
        return banned
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to prevent n-gram repetition.
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
            scores: Tensor of shape (batch_size, vocab_size)
        
        Returns:
            Modified scores with banned tokens set to -inf
        """
        batch_size = scores.shape[0]
        
        for batch_idx in range(batch_size):
            banned_tokens = set()
            
            # Get the sequence for this batch item
            sequence = input_ids[batch_idx]
            
            # 1. Standard n-gram blocking
            if sequence.shape[-1] >= self.ngram_size:
                current_prefix = tuple(sequence[-(self.ngram_size - 1):].tolist())
                
                search_start = max(0, sequence.shape[-1] - self.window_size)
                search_end = sequence.shape[-1] - self.ngram_size + 1
                
                sequence_list = sequence.tolist()
                for i in range(search_start, search_end):
                    ngram = tuple(sequence_list[i:i + self.ngram_size])
                    if ngram[:-1] == current_prefix:
                        banned_tokens.add(ngram[-1])
            
            # 2. Detect consecutive repeats (e.g., '. . . .')
            consecutive_banned = self._detect_consecutive_repeats(input_ids[batch_idx:batch_idx+1])
            banned_tokens.update(consecutive_banned)
            
            # 3. Detect short repetitive patterns
            pattern_banned = self._detect_short_patterns(input_ids[batch_idx:batch_idx+1])
            banned_tokens.update(pattern_banned)
            
            # Remove whitelisted tokens
            banned_tokens = banned_tokens - self.whitelist_token_ids
            
            # Apply bans to scores
            if banned_tokens:
                for token in banned_tokens:
                    scores[batch_idx, token] = -float("inf")
        
        return scores
