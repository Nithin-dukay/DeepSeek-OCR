import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from typing import List, Set


class NoRepeatNGramLogitsProcessor(LogitsProcessor):

    def __init__(self, ngram_size: int, window_size: int = 100, whitelist_token_ids: set = None, 
                 min_generated_tokens: int = 10, enable_adaptive: bool = True):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
        self.min_generated_tokens = min_generated_tokens
        self.enable_adaptive = enable_adaptive
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
        
        if len(input_ids) < self.ngram_size:
            return scores
        
        current_prefix = tuple(input_ids[-(self.ngram_size - 1):])
        
        # Only search within generated tokens, not the prompt
        # This prevents prompt content from interfering with generation
        generation_start = self.prompt_length
        search_start = max(generation_start, len(input_ids) - self.window_size)
        search_end = len(input_ids) - self.ngram_size + 1
        
        # If search range is invalid or too small, skip processing
        if search_end <= search_start:
            return scores
        
        banned_tokens = set()
        repetition_count = {}
        
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                next_token = ngram[-1]
                repetition_count[next_token] = repetition_count.get(next_token, 0) + 1
                
                # Adaptive blocking: only ban if repeated multiple times
                if self.enable_adaptive:
                    # Ban only if the token appears more than once in the pattern
                    if repetition_count[next_token] > 1:
                        banned_tokens.add(next_token)
                else:
                    # Original behavior: ban on first occurrence
                    banned_tokens.add(next_token)
        
        banned_tokens = banned_tokens - self.whitelist_token_ids
        
        if banned_tokens:
            scores = scores.clone()
            for token in banned_tokens:
                scores[token] = -float("inf")
        
        return scores