#!/usr/bin/env python3
"""
Test script to verify the enhanced NoRepeatNGramLogitsProcessor
fixes the infinite dot repetition issue (GitHub Issue #250)
"""

import sys
import torch
sys.path.insert(0, '/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm')

from process.ngram_norepeat import NoRepeatNGramLogitsProcessor


def test_consecutive_repetitions():
    """Test that consecutive repetitions are detected and banned"""
    print("Test 1: Consecutive Repetitions")
    print("-" * 50)
    
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        max_token_repetition_ratio=0.4,
        max_consecutive_repetitions=5,
        enable_enhanced_detection=True
    )
    
    # Simulate a sequence with 5 consecutive dots (token 13)
    # This should trigger the consecutive repetition detection
    input_ids = [1, 2, 3, 4, 5, 13, 13, 13, 13, 13]
    
    # Create dummy scores
    vocab_size = 150000
    scores = torch.zeros(vocab_size)
    scores[13] = 10.0  # High score for the dot token
    
    # Process the scores
    processed_scores = processor(input_ids, scores)
    
    # Check if token 13 is banned (should be -inf)
    if processed_scores[13] == float('-inf'):
        print("✓ PASS: Consecutive repetitions detected and banned")
    else:
        print("✗ FAIL: Consecutive repetitions not detected")
    
    print()


def test_pattern_repetitions():
    """Test that repeating patterns like '. <space> . <space>' are detected"""
    print("Test 2: Pattern Repetitions (2-token pattern)")
    print("-" * 50)
    
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        max_token_repetition_ratio=0.4,
        max_consecutive_repetitions=5,
        enable_enhanced_detection=True
    )
    
    # Simulate a sequence with repeating pattern: [13, 14] repeated 3 times
    # This mimics ". <space> . <space> . <space>"
    input_ids = [1, 2, 3, 13, 14, 13, 14, 13, 14]
    
    vocab_size = 150000
    scores = torch.zeros(vocab_size)
    scores[13] = 10.0
    scores[14] = 10.0
    
    processed_scores = processor(input_ids, scores)
    
    # Both tokens in the pattern should be banned
    if processed_scores[13] == float('-inf') and processed_scores[14] == float('-inf'):
        print("✓ PASS: 2-token pattern repetitions detected and banned")
    else:
        print("✗ FAIL: 2-token pattern repetitions not detected")
    
    print()


def test_frequency_based_detection():
    """Test that tokens appearing too frequently in the window are banned"""
    print("Test 3: Frequency-Based Detection")
    print("-" * 50)
    
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        max_token_repetition_ratio=0.4,
        max_consecutive_repetitions=5,
        enable_enhanced_detection=True
    )
    
    # Create a sequence where token 13 appears 50% of the time (exceeds 40% threshold)
    input_ids = [13, 1, 13, 2, 13, 3, 13, 4, 13, 5, 13, 6, 13, 7, 13, 8]
    
    vocab_size = 150000
    scores = torch.zeros(vocab_size)
    scores[13] = 10.0
    
    processed_scores = processor(input_ids, scores)
    
    if processed_scores[13] == float('-inf'):
        print("✓ PASS: High-frequency token detected and banned")
    else:
        print("✗ FAIL: High-frequency token not detected")
    
    print()


def test_whitelist_tokens():
    """Test that whitelisted tokens are not banned even if they repeat"""
    print("Test 4: Whitelist Token Protection")
    print("-" * 50)
    
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        max_token_repetition_ratio=0.4,
        max_consecutive_repetitions=5,
        enable_enhanced_detection=True
    )
    
    # Token 128821 is whitelisted (<td>), should not be banned even with repetitions
    input_ids = [1, 2, 3, 128821, 128821, 128821, 128821, 128821, 128821]
    
    vocab_size = 150000
    scores = torch.zeros(vocab_size)
    scores[128821] = 10.0
    
    processed_scores = processor(input_ids, scores)
    
    if processed_scores[128821] != float('-inf'):
        print("✓ PASS: Whitelisted token not banned despite repetitions")
    else:
        print("✗ FAIL: Whitelisted token was incorrectly banned")
    
    print()


def test_normal_text():
    """Test that normal text without excessive repetitions is not affected"""
    print("Test 5: Normal Text (No False Positives)")
    print("-" * 50)
    
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        max_token_repetition_ratio=0.4,
        max_consecutive_repetitions=5,
        enable_enhanced_detection=True
    )
    
    # Normal varied sequence
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    vocab_size = 150000
    scores = torch.zeros(vocab_size)
    scores[13] = 10.0
    
    processed_scores = processor(input_ids, scores)
    
    if processed_scores[13] != float('-inf'):
        print("✓ PASS: Normal text not affected by repetition detection")
    else:
        print("✗ FAIL: Normal text incorrectly flagged as repetitive")
    
    print()


def test_three_token_pattern():
    """Test that 3-token repeating patterns are detected"""
    print("Test 6: Pattern Repetitions (3-token pattern)")
    print("-" * 50)
    
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        max_token_repetition_ratio=0.4,
        max_consecutive_repetitions=5,
        enable_enhanced_detection=True
    )
    
    # Simulate a sequence with repeating pattern: [13, 14, 15] repeated 3 times
    input_ids = [1, 2, 13, 14, 15, 13, 14, 15, 13, 14, 15]
    
    vocab_size = 150000
    scores = torch.zeros(vocab_size)
    scores[13] = 10.0
    scores[14] = 10.0
    scores[15] = 10.0
    
    processed_scores = processor(input_ids, scores)
    
    # All tokens in the pattern should be banned
    if (processed_scores[13] == float('-inf') and 
        processed_scores[14] == float('-inf') and 
        processed_scores[15] == float('-inf')):
        print("✓ PASS: 3-token pattern repetitions detected and banned")
    else:
        print("✗ FAIL: 3-token pattern repetitions not detected")
    
    print()


def main():
    print("=" * 50)
    print("Testing Enhanced NoRepeatNGramLogitsProcessor")
    print("Fix for GitHub Issue #250: Infinite Dot Repetition")
    print("=" * 50)
    print()
    
    test_consecutive_repetitions()
    test_pattern_repetitions()
    test_frequency_based_detection()
    test_whitelist_tokens()
    test_normal_text()
    test_three_token_pattern()
    
    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
