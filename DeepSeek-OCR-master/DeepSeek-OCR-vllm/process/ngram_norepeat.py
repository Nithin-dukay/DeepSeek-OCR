import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from typing import List, Set


class NoRepeatNGramLogitsProcessor(LogitsProcessor):

    def __init__(self, ngram_size: int, window_size: int = 100, whitelist_token_ids: set = None, 
                 min_generated_tokens: int = 10, prompt_length: int = 0):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
        self.min_generated_tokens = min_generated_tokens
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids: List[int], scores: torch.FloatTensor) -> torch.FloatTensor:
        # Only apply n-gram blocking after generating minimum tokens
        # This prevents the processor from interfering with the initial generation
        generated_length = len(input_ids) - self.prompt_length
        if generated_length < self.min_generated_tokens:
            return scores
            
        if len(input_ids) < self.ngram_size:
            return scores
        
        current_prefix = tuple(input_ids[-(self.ngram_size - 1):])
        
        # Only search within the generated portion, not the prompt
        search_start = max(self.prompt_length, len(input_ids) - self.window_size)
        search_end = len(input_ids) - self.ngram_size + 1
        
        # Skip if search range is invalid
        if search_start >= search_end:
            return scores
        
        banned_tokens = set()
        repetition_count = {}
        
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                next_token = ngram[-1]
                repetition_count[next_token] = repetition_count.get(next_token, 0) + 1
                # Only ban tokens that appear multiple times (actual repetition)
                if repetition_count[next_token] >= 2:
                    banned_tokens.add(next_token)
        
        banned_tokens = banned_tokens - self.whitelist_token_ids
        
        if banned_tokens:
            scores = scores.clone()
            for token in banned_tokens:
                scores[token] = -float("inf")
        
        return scores