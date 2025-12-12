"""
Repetition Detection and Mitigation for DeepSeek-OCR
Addresses GitHub Issue #151: Detects and mitigates catastrophic failures
(loops/duplication) in OCR output from historical documents.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def detect_repetition_patterns(
    text: str,
    threshold: float = 0.3,
    min_pattern_length: int = 10,
    max_pattern_length: int = 200
) -> Tuple[float, List[Dict]]:
    """
    Detect repetitive patterns in generated text.
    
    Returns a score from 0.0 (no repetition) to 1.0 (severe repetition).
    
    Args:
        text: Generated text to analyze
        threshold: Threshold for considering text as repetitive
        min_pattern_length: Minimum character length for patterns
        max_pattern_length: Maximum character length for patterns
        
    Returns:
        Tuple of (repetition_score, list of detected patterns)
    """
    if not text or len(text) < min_pattern_length * 2:
        return 0.0, []
    
    patterns = []
    
    # 1. Detect exact substring repetition
    exact_score, exact_patterns = detect_exact_repetition(
        text, min_pattern_length, max_pattern_length
    )
    patterns.extend(exact_patterns)
    
    # 2. Detect phrase-level repetition
    phrase_score, phrase_patterns = detect_phrase_repetition(text)
    patterns.extend(phrase_patterns)
    
    # 3. Detect line-level repetition
    line_score, line_patterns = detect_line_repetition(text)
    patterns.extend(line_patterns)
    
    # 4. Detect word-level repetition (excessive)
    word_score = detect_word_repetition(text)
    
    # 5. Check for stuck loops (same short phrase many times)
    loop_score, loop_patterns = detect_stuck_loops(text)
    patterns.extend(loop_patterns)
    
    # Combine scores (weighted)
    repetition_score = (
        exact_score * 0.3 +
        phrase_score * 0.2 +
        line_score * 0.2 +
        word_score * 0.1 +
        loop_score * 0.2
    )
    
    return min(repetition_score, 1.0), patterns


def detect_exact_repetition(
    text: str,
    min_length: int = 10,
    max_length: int = 200
) -> Tuple[float, List[Dict]]:
    """
    Detect exact substring repetition.
    
    Returns score and list of repeated substrings.
    """
    patterns = []
    text_len = len(text)
    
    if text_len < min_length * 2:
        return 0.0, []
    
    # Check for repeated substrings of various lengths
    max_repetition_ratio = 0.0
    
    for length in range(min_length, min(max_length, text_len // 2)):
        # Sliding window to find repeated substrings
        seen = defaultdict(list)
        
        for i in range(text_len - length + 1):
            substring = text[i:i+length]
            seen[substring].append(i)
        
        # Find most repeated substring
        for substring, positions in seen.items():
            if len(positions) >= 3:  # Repeated at least 3 times
                repetition_ratio = (len(positions) * length) / text_len
                
                if repetition_ratio > max_repetition_ratio:
                    max_repetition_ratio = repetition_ratio
                
                patterns.append({
                    'type': 'exact_repetition',
                    'pattern': substring[:50] + '...' if len(substring) > 50 else substring,
                    'count': len(positions),
                    'positions': positions[:5],  # First 5 positions
                    'length': length,
                    'ratio': repetition_ratio
                })
    
    return min(max_repetition_ratio * 2, 1.0), patterns


def detect_phrase_repetition(text: str) -> Tuple[float, List[Dict]]:
    """
    Detect repeated phrases (3-10 words).
    """
    patterns = []
    
    # Split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) < 10:
        return 0.0, []
    
    # Check n-grams of different sizes
    max_repetition_ratio = 0.0
    
    for n in range(3, 11):  # 3 to 10 word phrases
        if len(words) < n * 2:
            continue
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        
        # Count occurrences
        ngram_counts = Counter(ngrams)
        
        for ngram, count in ngram_counts.most_common(10):
            if count >= 3:  # Repeated at least 3 times
                repetition_ratio = (count * n) / len(words)
                
                if repetition_ratio > max_repetition_ratio:
                    max_repetition_ratio = repetition_ratio
                
                patterns.append({
                    'type': 'phrase_repetition',
                    'pattern': ' '.join(ngram),
                    'count': count,
                    'n': n,
                    'ratio': repetition_ratio
                })
    
    return min(max_repetition_ratio * 3, 1.0), patterns


def detect_line_repetition(text: str) -> Tuple[float, List[Dict]]:
    """
    Detect repeated lines.
    """
    patterns = []
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if len(lines) < 3:
        return 0.0, []
    
    # Count line occurrences
    line_counts = Counter(lines)
    
    max_repetition_ratio = 0.0
    
    for line, count in line_counts.most_common(10):
        if count >= 3 and len(line) > 10:  # Repeated at least 3 times
            repetition_ratio = count / len(lines)
            
            if repetition_ratio > max_repetition_ratio:
                max_repetition_ratio = repetition_ratio
            
            patterns.append({
                'type': 'line_repetition',
                'pattern': line[:100] + '...' if len(line) > 100 else line,
                'count': count,
                'ratio': repetition_ratio
            })
    
    return min(max_repetition_ratio * 2, 1.0), patterns


def detect_word_repetition(text: str) -> float:
    """
    Detect excessive word repetition.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) < 10:
        return 0.0
    
    # Calculate word diversity
    unique_words = len(set(words))
    total_words = len(words)
    
    diversity_ratio = unique_words / total_words
    
    # Low diversity indicates repetition
    # Normal text has diversity ~0.4-0.6
    # Repetitive text has diversity <0.3
    if diversity_ratio > 0.4:
        return 0.0
    elif diversity_ratio > 0.3:
        return (0.4 - diversity_ratio) / 0.1  # 0.0 to 1.0
    else:
        return 1.0


def detect_stuck_loops(text: str, max_loop_length: int = 50) -> Tuple[float, List[Dict]]:
    """
    Detect stuck loops (same short phrase repeated many times consecutively).
    
    This is the most severe type of failure.
    """
    patterns = []
    
    if len(text) < max_loop_length * 3:
        return 0.0, []
    
    max_loop_score = 0.0
    
    # Check for consecutive repetitions
    for length in range(5, max_loop_length):
        i = 0
        while i < len(text) - length * 2:
            pattern = text[i:i+length]
            
            # Count consecutive repetitions
            consecutive_count = 1
            j = i + length
            
            while j + length <= len(text) and text[j:j+length] == pattern:
                consecutive_count += 1
                j += length
            
            if consecutive_count >= 5:  # Repeated 5+ times consecutively
                loop_score = min(consecutive_count / 10, 1.0)
                
                if loop_score > max_loop_score:
                    max_loop_score = loop_score
                
                patterns.append({
                    'type': 'stuck_loop',
                    'pattern': pattern[:50] + '...' if len(pattern) > 50 else pattern,
                    'consecutive_count': consecutive_count,
                    'position': i,
                    'length': length,
                    'severity': 'critical' if consecutive_count >= 10 else 'high'
                })
                
                i = j  # Skip past this loop
            else:
                i += 1
    
    return max_loop_score, patterns


def calculate_length_anomaly(
    generated_length: int,
    expected_length: Optional[int] = None,
    expected_ratio_range: Tuple[float, float] = (0.5, 2.0)
) -> float:
    """
    Calculate length anomaly score.
    
    Issue #151 reports OCR length often 3-5x ground truth for failures.
    
    Args:
        generated_length: Length of generated text
        expected_length: Expected length (ground truth)
        expected_ratio_range: Acceptable ratio range (min, max)
        
    Returns:
        Anomaly score from 0.0 (normal) to 1.0 (severe anomaly)
    """
    if expected_length is None:
        return 0.0
    
    if expected_length == 0:
        return 1.0 if generated_length > 0 else 0.0
    
    ratio = generated_length / expected_length
    
    min_ratio, max_ratio = expected_ratio_range
    
    if min_ratio <= ratio <= max_ratio:
        return 0.0
    elif ratio < min_ratio:
        # Too short
        return (min_ratio - ratio) / min_ratio
    else:
        # Too long (more concerning for OCR)
        # Issue #151: 3-5x is catastrophic
        if ratio >= 5.0:
            return 1.0
        elif ratio >= 3.0:
            return 0.8
        else:
            return (ratio - max_ratio) / (3.0 - max_ratio) * 0.8


def analyze_ocr_output(
    text: str,
    expected_length: Optional[int] = None,
    repetition_threshold: float = 0.3
) -> Dict:
    """
    Comprehensive analysis of OCR output for quality issues.
    
    Args:
        text: Generated OCR text
        expected_length: Expected text length (if known)
        repetition_threshold: Threshold for repetition detection
        
    Returns:
        Dictionary with analysis results
    """
    # Detect repetition
    repetition_score, patterns = detect_repetition_patterns(
        text, threshold=repetition_threshold
    )
    
    # Calculate length anomaly
    length_anomaly = calculate_length_anomaly(
        len(text), expected_length
    ) if expected_length else 0.0
    
    # Overall quality score (0.0 = good, 1.0 = catastrophic failure)
    quality_score = max(repetition_score, length_anomaly)
    
    # Determine status
    if quality_score < 0.3:
        status = "good"
    elif quality_score < 0.6:
        status = "warning"
    else:
        status = "failure"
    
    return {
        'status': status,
        'quality_score': quality_score,
        'repetition_score': repetition_score,
        'length_anomaly': length_anomaly,
        'text_length': len(text),
        'expected_length': expected_length,
        'length_ratio': len(text) / expected_length if expected_length else None,
        'patterns': patterns,
        'recommendations': generate_recommendations(
            repetition_score, length_anomaly, patterns
        )
    }


def generate_recommendations(
    repetition_score: float,
    length_anomaly: float,
    patterns: List[Dict]
) -> List[str]:
    """
    Generate recommendations based on detected issues.
    """
    recommendations = []
    
    if repetition_score > 0.6:
        recommendations.append(
            "Severe repetition detected. Retry with stricter no_repeat_ngram_size (7-10) "
            "and higher repetition_penalty (1.3-1.5)."
        )
    elif repetition_score > 0.3:
        recommendations.append(
            "Moderate repetition detected. Consider increasing no_repeat_ngram_size to 6-7 "
            "and repetition_penalty to 1.2-1.3."
        )
    
    if length_anomaly > 0.8:
        recommendations.append(
            "Output length is abnormally long (3-5x expected). This indicates catastrophic "
            "failure. Retry with lower max_new_tokens and stricter penalties."
        )
    elif length_anomaly > 0.5:
        recommendations.append(
            "Output length is longer than expected. Consider reducing max_new_tokens."
        )
    
    # Check for stuck loops
    stuck_loops = [p for p in patterns if p['type'] == 'stuck_loop']
    if stuck_loops:
        recommendations.append(
            f"Detected {len(stuck_loops)} stuck loop(s). This is a critical failure. "
            "Retry with different prompt or use column-splitting for tall images."
        )
    
    if not recommendations:
        recommendations.append("Output quality looks good. No issues detected.")
    
    return recommendations


def example_usage():
    """Example usage of repetition detection."""
    
    # Example 1: Good text
    print("=== Example 1: Good text ===")
    good_text = """
    The British Library contains historical newspapers from the 1800s and 1900s.
    These documents provide valuable insights into the social and political climate
    of the era. Researchers can access digitized versions of these newspapers online.
    """
    
    result = analyze_ocr_output(good_text, expected_length=200)
    print(f"Status: {result['status']}")
    print(f"Quality score: {result['quality_score']:.3f}")
    print(f"Repetition score: {result['repetition_score']:.3f}")
    print(f"Recommendations: {result['recommendations']}")
    
    # Example 2: Repetitive text (stuck loop)
    print("\n=== Example 2: Stuck loop ===")
    bad_text = "and the " * 50 + "newspaper article continues..."
    
    result = analyze_ocr_output(bad_text, expected_length=100)
    print(f"Status: {result['status']}")
    print(f"Quality score: {result['quality_score']:.3f}")
    print(f"Repetition score: {result['repetition_score']:.3f}")
    print(f"Patterns detected: {len(result['patterns'])}")
    for pattern in result['patterns'][:3]:
        print(f"  - {pattern['type']}: {pattern.get('pattern', 'N/A')[:50]}")
    print(f"Recommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    
    # Example 3: Length anomaly
    print("\n=== Example 3: Length anomaly ===")
    long_text = "This is a test. " * 200
    
    result = analyze_ocr_output(long_text, expected_length=100)
    print(f"Status: {result['status']}")
    print(f"Quality score: {result['quality_score']:.3f}")
    print(f"Length ratio: {result['length_ratio']:.2f}x")
    print(f"Length anomaly: {result['length_anomaly']:.3f}")
    print(f"Recommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")


if __name__ == "__main__":
    example_usage()
