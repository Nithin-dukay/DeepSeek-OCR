import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from typing import List, Set
from collections import Counter


class NoRepeatNGramLogitsProcessor(LogitsProcessor):

    def __init__(
        self, 
        ngram_size: int, 
        window_size: int = 100, 
        whitelist_token_ids: set = None,
        max_token_repetition_ratio: float = 0.4,
        max_consecutive_repetitions: int = 5,
        enable_enhanced_detection: bool = True
    ):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
        self.max_token_repetition_ratio = max_token_repetition_ratio
        self.max_consecutive_repetitions = max_consecutive_repetitions
        self.enable_enhanced_detection = enable_enhanced_detection
    
    def __call__(self, input_ids: List[int], scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(input_ids) < self.ngram_size:
            return scores
        
        banned_tokens = set()
        
        # Original n-gram based detection
        current_prefix = tuple(input_ids[-(self.ngram_size - 1):])
        
        search_start = max(0, len(input_ids) - self.window_size)
        search_end = len(input_ids) - self.ngram_size + 1
        
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])
        
        # Enhanced detection for repetitive patterns
        if self.enable_enhanced_detection:
            # Get the recent window of tokens
            window_start = max(0, len(input_ids) - self.window_size)
            recent_tokens = input_ids[window_start:]
            
            # 1. Check for excessive token frequency in the window
            if len(recent_tokens) >= 10:  # Only check if we have enough tokens
                token_counts = Counter(recent_tokens)
                window_length = len(recent_tokens)
                
                for token_id, count in token_counts.items():
                    if token_id not in self.whitelist_token_ids:
                        # If a token appears too frequently in the window, ban it
                        if count / window_length > self.max_token_repetition_ratio:
                            banned_tokens.add(token_id)
            
            # 2. Check for consecutive repetitions
            if len(input_ids) >= self.max_consecutive_repetitions:
                # Check if the last N tokens are all the same
                last_n_tokens = input_ids[-self.max_consecutive_repetitions:]
                if len(set(last_n_tokens)) == 1:  # All tokens are identical
                    repeated_token = last_n_tokens[0]
                    if repeated_token not in self.whitelist_token_ids:
                        banned_tokens.add(repeated_token)
            
            # 3. Check for short repeating patterns (e.g., "A B A B A B")
            # This catches patterns like ". <space> . <space> . <space>"
            if len(input_ids) >= 6:
                # Check for 2-token patterns repeating
                pattern_len = 2
                last_pattern = tuple(input_ids[-pattern_len:])
                
                # Count how many times this pattern repeats at the end
                repetition_count = 1
                pos = len(input_ids) - pattern_len * 2
                while pos >= 0:
                    check_pattern = tuple(input_ids[pos:pos + pattern_len])
                    if check_pattern == last_pattern:
                        repetition_count += 1
                        pos -= pattern_len
                    else:
                        break
                
                # If the pattern repeats too many times, ban the tokens in it
                if repetition_count >= 3:  # Pattern repeats 3+ times
                    for token_id in last_pattern:
                        if token_id not in self.whitelist_token_ids:
                            banned_tokens.add(token_id)
            
            # 4. Check for 3-token patterns repeating
            if len(input_ids) >= 9:
                pattern_len = 3
                last_pattern = tuple(input_ids[-pattern_len:])
                
                repetition_count = 1
                pos = len(input_ids) - pattern_len * 2
                while pos >= 0:
                    check_pattern = tuple(input_ids[pos:pos + pattern_len])
                    if check_pattern == last_pattern:
                        repetition_count += 1
                        pos -= pattern_len
                    else:
                        break
                
                if repetition_count >= 3:
                    for token_id in last_pattern:
                        if token_id not in self.whitelist_token_ids:
                            banned_tokens.add(token_id)
        
        # Remove whitelisted tokens from banned set
        banned_tokens = banned_tokens - self.whitelist_token_ids
        
        # Apply bans to scores
        if banned_tokens:
            scores = scores.clone()
            for token in banned_tokens:
                scores[token] = -float("inf")
        
        return scores