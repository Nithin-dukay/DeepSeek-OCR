import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from typing import List, Set, Optional
from collections import deque


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Enhanced N-gram repetition prevention with pattern detection.
    
    This processor prevents repetitive output by:
    1. Banning tokens that would complete previously seen n-grams
    2. Detecting consecutive repetition patterns
    3. Applying stronger penalties for excessive repetition
    """

    def __init__(
        self, 
        ngram_size: int, 
        window_size: int = 100, 
        whitelist_token_ids: set = None,
        max_consecutive_repeats: int = 3,
        repetition_penalty_scale: float = 2.0,
        pattern_detection_size: int = 20
    ):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
        self.max_consecutive_repeats = max_consecutive_repeats
        self.repetition_penalty_scale = repetition_penalty_scale
        self.pattern_detection_size = pattern_detection_size
        
        # Track recent patterns for consecutive repetition detection
        self.recent_patterns = deque(maxlen=10)
        self.repetition_count = 0
    
    def _detect_pattern_repetition(self, input_ids: List[int]) -> int:
        """
        Detect if recent tokens form a repeating pattern.
        Returns the number of consecutive repetitions detected.
        """
        if len(input_ids) < self.pattern_detection_size * 2:
            return 0
        
        # Check for repeating patterns of various sizes
        for pattern_size in range(self.pattern_detection_size, 5, -1):
            if len(input_ids) < pattern_size * 2:
                continue
            
            # Get the most recent pattern
            recent_pattern = tuple(input_ids[-pattern_size:])
            
            # Check how many times this pattern repeats consecutively
            repeat_count = 1
            offset = pattern_size
            
            while offset + pattern_size <= len(input_ids):
                prev_pattern = tuple(input_ids[-(offset + pattern_size):-offset])
                
                # Calculate similarity (allow some variation for numbers)
                similarity = sum(1 for a, b in zip(recent_pattern, prev_pattern) if a == b)
                similarity_ratio = similarity / pattern_size
                
                # If patterns are very similar (>80% match), count as repetition
                if similarity_ratio > 0.8:
                    repeat_count += 1
                    offset += pattern_size
                else:
                    break
            
            if repeat_count >= 2:
                return repeat_count
        
        return 0
    
    def __call__(self, input_ids: List[int], scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(input_ids) < self.ngram_size:
            return scores
        
        # Standard n-gram blocking
        current_prefix = tuple(input_ids[-(self.ngram_size - 1):])
        
        search_start = max(0, len(input_ids) - self.window_size)
        search_end = len(input_ids) - self.ngram_size + 1
        
        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])
        
        banned_tokens = banned_tokens - self.whitelist_token_ids
        
        # Detect pattern repetition
        repeat_count = self._detect_pattern_repetition(input_ids)
        
        # Apply penalties
        if banned_tokens or repeat_count >= self.max_consecutive_repeats:
            scores = scores.clone()
            
            # Ban exact n-gram matches
            for token in banned_tokens:
                scores[token] = -float("inf")
            
            # Apply stronger penalty if excessive repetition is detected
            if repeat_count >= self.max_consecutive_repeats:
                # Get tokens from the repeating pattern
                pattern_size = min(self.pattern_detection_size, len(input_ids) // 2)
                recent_pattern_tokens = set(input_ids[-pattern_size:])
                
                # Apply scaled penalty to tokens in the repeating pattern
                for token in recent_pattern_tokens:
                    if token not in self.whitelist_token_ids and scores[token] > -float("inf"):
                        # Apply logarithmic penalty based on repetition count
                        penalty = self.repetition_penalty_scale * (1 + torch.log(torch.tensor(float(repeat_count))))
                        scores[token] = scores[token] / penalty
        
        return scores