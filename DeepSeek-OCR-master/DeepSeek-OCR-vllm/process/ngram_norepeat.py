import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from typing import List, Set


class NoRepeatNGramLogitsProcessor(LogitsProcessor):

    def __init__(self, ngram_size: int, window_size: int = 100, whitelist_token_ids: set = None, 
                 min_generated_tokens: int = 10, enable_adaptive_ngram: bool = True):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
        self.min_generated_tokens = min_generated_tokens
        self.enable_adaptive_ngram = enable_adaptive_ngram
        self.prompt_length = None
    
    def __call__(self, input_ids: List[int], scores: torch.FloatTensor) -> torch.FloatTensor:
        # Store the prompt length on first call (when we have the full input including prompt)
        if self.prompt_length is None:
            # Estimate prompt length - we'll update this as we generate
            # For now, we consider the first call to have mostly prompt tokens
            self.prompt_length = len(input_ids)
        
        # Calculate how many tokens have been generated
        generated_length = len(input_ids) - self.prompt_length
        
        # Don't apply n-gram blocking until we've generated enough tokens
        # This prevents the processor from being too aggressive early in generation
        if generated_length < self.min_generated_tokens:
            return scores
        
        # Don't apply if we don't have enough tokens for an n-gram
        if len(input_ids) < self.ngram_size:
            return scores
        
        # Adaptive n-gram size: use smaller n-gram size early in generation
        effective_ngram_size = self.ngram_size
        if self.enable_adaptive_ngram and generated_length < self.window_size // 2:
            # Use a smaller n-gram size early in generation to be less restrictive
            effective_ngram_size = max(self.ngram_size // 2, 3)
        
        current_prefix = tuple(input_ids[-(effective_ngram_size - 1):])
        
        # Only look at generated tokens, not the prompt
        # This prevents prompt content from interfering with generation
        search_start = max(self.prompt_length, len(input_ids) - self.window_size)
        search_end = len(input_ids) - effective_ngram_size + 1
        
        # If search range is invalid, don't apply blocking
        if search_start >= search_end:
            return scores
        
        banned_tokens = set()
        repetition_count = {}
        
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + effective_ngram_size])
            if ngram[:-1] == current_prefix:
                last_token = ngram[-1]
                repetition_count[last_token] = repetition_count.get(last_token, 0) + 1
        
        # Only ban tokens that appear multiple times (more aggressive filtering)
        # This allows single occurrences to pass through
        for token, count in repetition_count.items():
            if count >= 2:  # Only ban if the pattern repeats at least twice
                banned_tokens.add(token)
        
        banned_tokens = banned_tokens - self.whitelist_token_ids
        
        if banned_tokens:
            scores = scores.clone()
            for token in banned_tokens:
                scores[token] = -float("inf")
        
        return scores