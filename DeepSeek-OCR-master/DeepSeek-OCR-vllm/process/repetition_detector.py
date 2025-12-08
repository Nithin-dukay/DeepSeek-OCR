"""
Repetition detection utility for early stopping in OCR generation.

This module provides utilities to detect excessive repetition patterns
in generated text and trigger early stopping to prevent runaway generation.
"""

import re
from typing import List, Tuple, Optional
from collections import Counter


class RepetitionDetector:
    """
    Detects various forms of repetition in generated text.
    
    This class can identify:
    1. Exact phrase repetition
    2. Structural repetition (same pattern with different values)
    3. Excessive consecutive similar lines
    4. Runaway number sequences
    """
    
    def __init__(
        self,
        max_consecutive_similar_lines: int = 5,
        similarity_threshold: float = 0.7,
        max_number_sequence_length: int = 20,
        enable_early_stopping: bool = True
    ):
        """
        Initialize the repetition detector.
        
        Args:
            max_consecutive_similar_lines: Maximum allowed consecutive similar lines
            similarity_threshold: Threshold for considering two lines similar (0-1)
            max_number_sequence_length: Maximum allowed length of incrementing number sequences
            enable_early_stopping: Whether to enable early stopping detection
        """
        self.max_consecutive_similar_lines = max_consecutive_similar_lines
        self.similarity_threshold = similarity_threshold
        self.max_number_sequence_length = max_number_sequence_length
        self.enable_early_stopping = enable_early_stopping
    
    def _calculate_line_similarity(self, line1: str, line2: str) -> float:
        """
        Calculate similarity between two lines, ignoring numbers.
        
        Returns a value between 0 and 1, where 1 means identical structure.
        """
        # Remove numbers, currency symbols, and extra whitespace for comparison
        pattern1 = re.sub(r'[\d\$\.]+', 'N', line1).strip()
        pattern2 = re.sub(r'[\d\$\.]+', 'N', line2).strip()
        
        if not pattern1 or not pattern2:
            return 0.0
        
        # Simple character-level similarity
        if pattern1 == pattern2:
            return 1.0
        
        # Calculate Jaccard similarity on words
        words1 = set(pattern1.split())
        words2 = set(pattern2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_number_sequence(self, text: str) -> Optional[Tuple[int, int, int]]:
        """
        Detect incrementing number sequences in text.
        
        Returns (start_value, end_value, sequence_length) if found, None otherwise.
        """
        # Find all numbers in the text (including decimals)
        numbers = []
        for m in re.finditer(r'\d+(?:\.\d+)?', text):
            try:
                num = float(m.group())
                numbers.append(num)
            except ValueError:
                continue
        
        if len(numbers) < 3:
            return None
        
        # Look for incrementing sequences (allowing for small variations)
        max_sequence_length = 0
        start_val = None
        end_val = None
        
        for i in range(len(numbers) - 2):
            sequence_length = 1
            current_diff = numbers[i + 1] - numbers[i]
            
            if current_diff <= 0:
                continue
            
            for j in range(i + 1, len(numbers) - 1):
                next_diff = numbers[j + 1] - numbers[j]
                # Allow for small variations in the increment (within 20%)
                if abs(next_diff - current_diff) <= abs(current_diff * 0.2):
                    sequence_length += 1
                else:
                    break
            
            if sequence_length > max_sequence_length:
                max_sequence_length = sequence_length
                start_val = int(numbers[i])
                end_val = int(numbers[i + sequence_length])
        
        if max_sequence_length >= 3:
            return (start_val, end_val, max_sequence_length)
        
        return None
    
    def detect_excessive_repetition(self, text: str) -> Tuple[bool, str]:
        """
        Detect if text contains excessive repetition.
        
        Returns:
            (should_stop, reason) tuple where should_stop is True if excessive
            repetition is detected, and reason explains why.
        """
        if not self.enable_early_stopping:
            return False, ""
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 3:
            return False, ""
        
        # Check for consecutive similar lines
        consecutive_similar = 1
        max_consecutive_similar = 1
        
        for i in range(len(lines) - 1):
            similarity = self._calculate_line_similarity(lines[i], lines[i + 1])
            
            if similarity >= self.similarity_threshold:
                consecutive_similar += 1
                max_consecutive_similar = max(max_consecutive_similar, consecutive_similar)
            else:
                consecutive_similar = 1
        
        if max_consecutive_similar > self.max_consecutive_similar_lines:
            return True, f"Detected {max_consecutive_similar} consecutive similar lines (threshold: {self.max_consecutive_similar_lines})"
        
        # Check for number sequences (like counting from 16 to 293)
        number_sequence = self._detect_number_sequence(text)
        if number_sequence:
            start, end, length = number_sequence
            if length > self.max_number_sequence_length:
                return True, f"Detected incrementing number sequence from {start} to {end} (length: {length})"
        
        # Check for exact phrase repetition
        # Look at the last 20 lines and check if any phrase repeats more than 3 times
        recent_lines = lines[-20:] if len(lines) > 20 else lines
        line_counts = Counter(recent_lines)
        
        for line, count in line_counts.items():
            if count > 3 and len(line) > 5:  # Lowered threshold for short lines
                return True, f"Exact phrase repeated {count} times: '{line[:50]}...'"
        
        # Check for repeating patterns (e.g., same word appearing many times)
        # This catches cases like "HAMANIAN" appearing 278 times
        all_lines_text = ' '.join(lines)
        words = re.findall(r'\b[A-Z]{3,}\b', all_lines_text)  # Find capitalized words
        if words:
            word_counts = Counter(words)
            for word, count in word_counts.items():
                # If a word appears more than 10 times and represents >30% of total lines
                if count > 10 and count > len(lines) * 0.3:
                    return True, f"Word '{word}' repeated excessively ({count} times, {count*100//len(lines)}% of lines)"
        
        return False, ""
    
    def truncate_at_repetition(self, text: str) -> str:
        """
        Truncate text at the point where excessive repetition begins.
        
        Returns the truncated text.
        """
        lines = text.split('\n')
        
        if len(lines) < 3:
            return text
        
        # Find the point where repetition starts
        for i in range(len(lines) - self.max_consecutive_similar_lines):
            consecutive_similar = 0
            
            for j in range(i, min(i + self.max_consecutive_similar_lines + 1, len(lines) - 1)):
                similarity = self._calculate_line_similarity(lines[j], lines[j + 1])
                
                if similarity >= self.similarity_threshold:
                    consecutive_similar += 1
                else:
                    break
            
            if consecutive_similar >= self.max_consecutive_similar_lines:
                # Truncate at this point
                return '\n'.join(lines[:i])
        
        return text
    
    def get_repetition_stats(self, text: str) -> dict:
        """
        Get statistics about repetition in the text.
        
        Returns a dictionary with various repetition metrics.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return {
                'total_lines': 0,
                'unique_lines': 0,
                'max_consecutive_similar': 0,
                'has_number_sequence': False,
                'most_repeated_line': None,
                'max_repetition_count': 0
            }
        
        # Calculate consecutive similar lines
        consecutive_similar = 1
        max_consecutive_similar = 1
        
        for i in range(len(lines) - 1):
            similarity = self._calculate_line_similarity(lines[i], lines[i + 1])
            
            if similarity >= self.similarity_threshold:
                consecutive_similar += 1
                max_consecutive_similar = max(max_consecutive_similar, consecutive_similar)
            else:
                consecutive_similar = 1
        
        # Find most repeated line
        line_counts = Counter(lines)
        most_common = line_counts.most_common(1)
        most_repeated_line = most_common[0][0] if most_common else None
        max_repetition_count = most_common[0][1] if most_common else 0
        
        # Check for number sequences
        number_sequence = self._detect_number_sequence(text)
        
        return {
            'total_lines': len(lines),
            'unique_lines': len(set(lines)),
            'max_consecutive_similar': max_consecutive_similar,
            'has_number_sequence': number_sequence is not None,
            'number_sequence_info': number_sequence,
            'most_repeated_line': most_repeated_line[:100] if most_repeated_line else None,
            'max_repetition_count': max_repetition_count
        }
