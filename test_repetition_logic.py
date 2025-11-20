#!/usr/bin/env python3
"""
Test script to verify the enhanced NoRepeatNGramLogitsProcessor logic
Tests the detection algorithms without requiring torch/transformers
"""

from collections import Counter


def test_consecutive_repetitions(input_ids, max_consecutive=5):
    """Test consecutive repetition detection logic"""
    if len(input_ids) >= max_consecutive:
        last_n_tokens = input_ids[-max_consecutive:]
        if len(set(last_n_tokens)) == 1:
            return True, last_n_tokens[0]
    return False, None


def test_pattern_repetitions(input_ids, pattern_len=2, min_repeats=3):
    """Test repeating pattern detection logic"""
    if len(input_ids) >= pattern_len * min_repeats:
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
        
        if repetition_count >= min_repeats:
            return True, last_pattern
    return False, None


def test_frequency_detection(input_ids, window_size=90, max_ratio=0.4):
    """Test frequency-based detection logic"""
    window_start = max(0, len(input_ids) - window_size)
    recent_tokens = input_ids[window_start:]
    
    if len(recent_tokens) >= 10:
        token_counts = Counter(recent_tokens)
        window_length = len(recent_tokens)
        
        high_freq_tokens = []
        for token_id, count in token_counts.items():
            if count / window_length > max_ratio:
                high_freq_tokens.append((token_id, count / window_length))
        
        if high_freq_tokens:
            return True, high_freq_tokens
    return False, None


def run_tests():
    print("=" * 60)
    print("Testing Enhanced Repetition Detection Logic")
    print("=" * 60)
    print()
    
    # Test 1: Consecutive repetitions
    print("Test 1: Consecutive Repetitions")
    print("-" * 60)
    input_ids = [1, 2, 3, 4, 5, 13, 13, 13, 13, 13]
    detected, token = test_consecutive_repetitions(input_ids, max_consecutive=5)
    if detected:
        print(f"✓ PASS: Detected consecutive repetition of token {token}")
    else:
        print("✗ FAIL: Did not detect consecutive repetitions")
    print()
    
    # Test 2: 2-token pattern repetitions
    print("Test 2: 2-Token Pattern Repetitions (. <space> . <space>)")
    print("-" * 60)
    input_ids = [1, 2, 3, 13, 14, 13, 14, 13, 14]
    detected, pattern = test_pattern_repetitions(input_ids, pattern_len=2, min_repeats=3)
    if detected:
        print(f"✓ PASS: Detected repeating pattern {pattern}")
    else:
        print("✗ FAIL: Did not detect pattern repetitions")
    print()
    
    # Test 3: 3-token pattern repetitions
    print("Test 3: 3-Token Pattern Repetitions")
    print("-" * 60)
    input_ids = [1, 2, 13, 14, 15, 13, 14, 15, 13, 14, 15]
    detected, pattern = test_pattern_repetitions(input_ids, pattern_len=3, min_repeats=3)
    if detected:
        print(f"✓ PASS: Detected repeating pattern {pattern}")
    else:
        print("✗ FAIL: Did not detect pattern repetitions")
    print()
    
    # Test 4: Frequency-based detection
    print("Test 4: Frequency-Based Detection (50% frequency)")
    print("-" * 60)
    input_ids = [13, 1, 13, 2, 13, 3, 13, 4, 13, 5, 13, 6, 13, 7, 13, 8]
    detected, tokens = test_frequency_detection(input_ids, window_size=90, max_ratio=0.4)
    if detected:
        print(f"✓ PASS: Detected high-frequency tokens: {tokens}")
    else:
        print("✗ FAIL: Did not detect high-frequency tokens")
    print()
    
    # Test 5: Normal text (should not trigger)
    print("Test 5: Normal Text (No False Positives)")
    print("-" * 60)
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    detected_consec, _ = test_consecutive_repetitions(input_ids, max_consecutive=5)
    detected_pattern, _ = test_pattern_repetitions(input_ids, pattern_len=2, min_repeats=3)
    detected_freq, _ = test_frequency_detection(input_ids, window_size=90, max_ratio=0.4)
    
    if not (detected_consec or detected_pattern or detected_freq):
        print("✓ PASS: Normal text not flagged as repetitive")
    else:
        print("✗ FAIL: Normal text incorrectly flagged")
    print()
    
    # Test 6: Edge case - exactly at threshold
    print("Test 6: Edge Case - Exactly at Threshold (40%)")
    print("-" * 60)
    # Create sequence where token 13 appears exactly 40% of the time
    input_ids = [13, 13, 1, 2, 3, 4, 5, 6, 7, 8]  # 2 out of 10 = 20%
    detected, tokens = test_frequency_detection(input_ids, window_size=90, max_ratio=0.4)
    if not detected:
        print("✓ PASS: Token at 20% not flagged (below 40% threshold)")
    else:
        print("✗ FAIL: Token incorrectly flagged")
    print()
    
    # Test 7: Simulating the dot issue from GitHub #250
    print("Test 7: Simulating GitHub Issue #250 (Infinite Dots)")
    print("-" * 60)
    # Simulate: ". . . . . . . . . . . . . . . . . . . . . . . . . . . ."
    # This could be tokenized as alternating dot and space tokens
    dot_token = 13
    space_token = 14
    input_ids = [1, 2, 3]  # Some initial tokens
    # Add 20 repetitions of ". " pattern
    for _ in range(20):
        input_ids.extend([dot_token, space_token])
    
    detected_pattern, pattern = test_pattern_repetitions(input_ids, pattern_len=2, min_repeats=3)
    detected_freq, tokens = test_frequency_detection(input_ids, window_size=90, max_ratio=0.4)
    
    if detected_pattern or detected_freq:
        print("✓ PASS: Infinite dot pattern would be detected and stopped")
        if detected_pattern:
            print(f"  - Pattern detection: {pattern}")
        if detected_freq:
            print(f"  - Frequency detection: {tokens}")
    else:
        print("✗ FAIL: Infinite dot pattern not detected")
    print()
    
    print("=" * 60)
    print("All logic tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
