"""
Repetition Detection Utility for Early Stopping

This module provides utilities to detect infinite loops and excessive repetition
in generated text, allowing for early stopping before consuming all max_tokens.
"""

from typing import List, Optional, Tuple
from collections import Counter
import re


class RepetitionDetector:
    """
    Detects excessive repetition patterns in generated token sequences.
    
    This detector can identify:
    1. Infinite loops (same pattern repeating many times)
    2. Excessive consecutive identical tokens
    3. Alternating patterns (A B A B A B...)
    4. Character-level repetition in decoded text
    
    Args:
        max_consecutive_tokens: Maximum allowed consecutive identical tokens (default: 5)
        max_pattern_repeats: Maximum allowed pattern repetitions (default: 10)
        pattern_sizes: List of pattern sizes to check (default: [1, 2, 3, 4, 5])
        min_sequence_length: Minimum sequence length before checking (default: 20)
    """
    
    def __init__(
        self,
        max_consecutive_tokens: int = 5,
        max_pattern_repeats: int = 10,
        pattern_sizes: Optional[List[int]] = None,
        min_sequence_length: int = 20
    ):
        self.max_consecutive_tokens = max_consecutive_tokens
        self.max_pattern_repeats = max_pattern_repeats
        self.pattern_sizes = pattern_sizes or [1, 2, 3, 4, 5]
        self.min_sequence_length = min_sequence_length
    
    def detect_consecutive_repetition(self, tokens: List[int]) -> Tuple[bool, int]:
        """
        Detect if there are too many consecutive identical tokens.
        
        Returns:
            (is_repetitive, count): Whether repetition detected and the count
        """
        if len(tokens) < self.max_consecutive_tokens:
            return False, 0
        
        # Check the last N tokens
        recent = tokens[-self.max_consecutive_tokens:]
        if len(set(recent)) == 1:
            # All are the same, count how many in total
            token = recent[0]
            count = 0
            for t in reversed(tokens):
                if t == token:
                    count += 1
                else:
                    break
            
            if count >= self.max_consecutive_tokens:
                return True, count
        
        return False, 0
    
    def detect_pattern_repetition(self, tokens: List[int]) -> Tuple[bool, int, int]:
        """
        Detect if a pattern is repeating excessively.
        
        Returns:
            (is_repetitive, pattern_size, repeat_count): Detection result
        """
        if len(tokens) < self.min_sequence_length:
            return False, 0, 0
        
        for pattern_size in self.pattern_sizes:
            if len(tokens) < pattern_size * 3:
                continue
            
            # Get the last pattern
            last_pattern = tuple(tokens[-pattern_size:])
            
            # Count how many times this pattern repeats at the end
            repeat_count = 0
            for i in range(len(tokens) - pattern_size, -1, -pattern_size):
                if i < 0:
                    break
                pattern = tuple(tokens[i:i + pattern_size])
                if pattern == last_pattern:
                    repeat_count += 1
                else:
                    break
            
            if repeat_count >= self.max_pattern_repeats:
                return True, pattern_size, repeat_count
        
        return False, 0, 0
    
    def detect_text_repetition(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect repetition in decoded text (character-level).
        
        This catches cases like ". . . . . . . . . ." or "abc abc abc abc"
        
        Returns:
            (is_repetitive, pattern): Whether repetition detected and the pattern
        """
        if len(text) < 50:
            return False, None
        
        # Check for patterns like ". . . . ." (single char with spaces)
        single_char_pattern = re.search(r'(.)\s+\1\s+\1\s+\1\s+\1', text[-100:])
        if single_char_pattern:
            return True, single_char_pattern.group(0)
        
        # Check for repeated short strings
        for length in [2, 3, 4, 5, 10]:
            if len(text) < length * 5:
                continue
            
            # Get the last segment
            segment = text[-length:]
            
            # Count occurrences in the recent text
            recent_text = text[-length * 15:]
            count = recent_text.count(segment)
            
            if count >= 8:  # If appears 8+ times in recent text
                return True, segment
        
        return False, None
    
    def should_stop_generation(
        self, 
        tokens: List[int], 
        decoded_text: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Determine if generation should stop due to excessive repetition.
        
        Args:
            tokens: List of generated token IDs
            decoded_text: Optional decoded text for character-level checking
        
        Returns:
            (should_stop, reason): Whether to stop and the reason
        """
        # Check consecutive repetition
        is_consecutive, count = self.detect_consecutive_repetition(tokens)
        if is_consecutive:
            return True, f"Consecutive token repetition detected ({count} times)"
        
        # Check pattern repetition
        is_pattern, pattern_size, repeat_count = self.detect_pattern_repetition(tokens)
        if is_pattern:
            return True, f"Pattern repetition detected (size={pattern_size}, repeats={repeat_count})"
        
        # Check text-level repetition if text provided
        if decoded_text:
            is_text_rep, pattern = self.detect_text_repetition(decoded_text)
            if is_text_rep:
                return True, f"Text repetition detected: '{pattern[:50]}...'"
        
        return False, ""
    
    def analyze_sequence(self, tokens: List[int]) -> dict:
        """
        Analyze a token sequence for various repetition metrics.
        
        Returns:
            Dictionary with analysis results
        """
        if len(tokens) < 10:
            return {
                "length": len(tokens),
                "unique_tokens": len(set(tokens)),
                "has_repetition": False
            }
        
        # Basic stats
        unique_tokens = len(set(tokens))
        token_counts = Counter(tokens)
        most_common = token_counts.most_common(3)
        
        # Check for issues
        consecutive_rep, cons_count = self.detect_consecutive_repetition(tokens)
        pattern_rep, pat_size, pat_count = self.detect_pattern_repetition(tokens)
        
        return {
            "length": len(tokens),
            "unique_tokens": unique_tokens,
            "diversity_ratio": unique_tokens / len(tokens),
            "most_common_tokens": most_common,
            "has_consecutive_repetition": consecutive_rep,
            "consecutive_count": cons_count,
            "has_pattern_repetition": pattern_rep,
            "pattern_size": pat_size,
            "pattern_repeats": pat_count,
            "has_repetition": consecutive_rep or pattern_rep
        }


def create_default_detector() -> RepetitionDetector:
    """Create a RepetitionDetector with default settings for OCR tasks."""
    return RepetitionDetector(
        max_consecutive_tokens=5,
        max_pattern_repeats=10,
        pattern_sizes=[1, 2, 3, 4, 5],
        min_sequence_length=20
    )
