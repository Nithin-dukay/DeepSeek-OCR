import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens
from typing import List, Set, Optional
from collections import Counter


class NoRepeatNGramLogitsProcessor(LogitsProcessor):

    def __init__(
        self, 
        ngram_size: int, 
        window_size: int = 100, 
        whitelist_token_ids: set = None,
        adaptive: bool = True,
        repetition_threshold: int = 3
    ):
        """
        Enhanced N-gram repetition prevention processor.
        
        Args:
            ngram_size: Size of n-grams to check for repetition
            window_size: Size of the sliding window to check for repetitions
            whitelist_token_ids: Token IDs that are allowed to repeat
            adaptive: If True, dynamically adjust parameters based on detected patterns
            repetition_threshold: Number of times a token can repeat before being banned
        """
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()
        self.adaptive = adaptive
        self.repetition_threshold = repetition_threshold
        
        # Track repetition patterns for adaptive behavior
        self.repetition_history = []
        self.consecutive_repetitions = 0
    
    def _detect_numeric_repetition(self, input_ids: List[int], window: int = 20) -> bool:
        """
        Detect if the recent output consists primarily of repetitive numeric patterns.
        
        Args:
            input_ids: List of token IDs
            window: Number of recent tokens to check
            
        Returns:
            True if numeric repetition is detected
        """
        if len(input_ids) < window:
            return False
        
        recent_tokens = input_ids[-window:]
        
        # Count unique tokens in recent window
        unique_ratio = len(set(recent_tokens)) / len(recent_tokens)
        
        # If less than 20% unique tokens, likely repetitive
        if unique_ratio < 0.2:
            return True
        
        # Check for repeating sequences
        for seq_len in range(2, min(10, window // 2)):
            for i in range(len(recent_tokens) - seq_len * 2):
                seq1 = tuple(recent_tokens[i:i + seq_len])
                seq2 = tuple(recent_tokens[i + seq_len:i + seq_len * 2])
                if seq1 == seq2:
                    return True
        
        return False
    
    def _get_adaptive_params(self, input_ids: List[int]) -> tuple:
        """
        Dynamically adjust ngram_size and window_size based on generation patterns.
        
        Args:
            input_ids: List of token IDs
            
        Returns:
            Tuple of (adjusted_ngram_size, adjusted_window_size)
        """
        if not self.adaptive:
            return self.ngram_size, self.window_size
        
        # Detect if we're in a repetitive state
        is_repetitive = self._detect_numeric_repetition(input_ids)
        
        if is_repetitive:
            self.consecutive_repetitions += 1
            # Increase strictness when repetition is detected
            adjusted_ngram = max(3, self.ngram_size - 5)  # Smaller n-grams catch more patterns
            adjusted_window = min(self.window_size * 2, 200)  # Larger window
        else:
            self.consecutive_repetitions = max(0, self.consecutive_repetitions - 1)
            adjusted_ngram = self.ngram_size
            adjusted_window = self.window_size
        
        return adjusted_ngram, adjusted_window
    
    def _get_token_frequency(self, input_ids: List[int], window: int) -> Counter:
        """
        Get frequency count of tokens in recent window.
        
        Args:
            input_ids: List of token IDs
            window: Size of window to analyze
            
        Returns:
            Counter object with token frequencies
        """
        search_start = max(0, len(input_ids) - window)
        recent_tokens = input_ids[search_start:]
        return Counter(recent_tokens)
    
    def __call__(self, input_ids: List[int], scores: torch.FloatTensor) -> torch.FloatTensor:
        # Get adaptive parameters
        ngram_size, window_size = self._get_adaptive_params(input_ids)
        
        if len(input_ids) < ngram_size:
            return scores
        
        current_prefix = tuple(input_ids[-(ngram_size - 1):])
        
        search_start = max(0, len(input_ids) - window_size)
        search_end = len(input_ids) - ngram_size + 1
        
        # Standard n-gram based banning
        banned_tokens = set()
        token_counts = Counter()
        
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + ngram_size])
            if ngram[:-1] == current_prefix:
                token = ngram[-1]
                token_counts[token] += 1
                # Ban tokens that appear more than threshold times
                if token_counts[token] >= self.repetition_threshold:
                    banned_tokens.add(token)
        
        # Additional frequency-based banning for highly repetitive tokens
        if self.consecutive_repetitions > 2:
            token_freq = self._get_token_frequency(input_ids, window_size)
            # Ban tokens that appear too frequently in recent history
            for token, count in token_freq.items():
                if count > window_size * 0.3:  # More than 30% of recent tokens
                    banned_tokens.add(token)
        
        # Remove whitelisted tokens
        banned_tokens = banned_tokens - self.whitelist_token_ids
        
        # Apply banning
        if banned_tokens:
            scores = scores.clone()
            for token in banned_tokens:
                if token < len(scores):  # Safety check
                    scores[token] = -float("inf")
        
        return scores