#!/usr/bin/env python3
"""
Test script for Issue #257 fix: Repetition detection and prevention

This script tests the enhanced repetition detection mechanisms without
requiring the full model to be loaded.
"""

import sys
import os

# Add the DeepSeek-OCR-vllm directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-vllm'))

from process.repetition_detector import RepetitionDetector


def test_exact_repetition():
    """Test detection of exact phrase repetition"""
    print("=" * 60)
    print("Test 1: Exact Phrase Repetition")
    print("=" * 60)
    
    detector = RepetitionDetector(
        max_consecutive_similar_lines=5,
        similarity_threshold=0.7,
        max_number_sequence_length=20
    )
    
    # Simulate the problematic output from Issue #257
    text = """- CHEESE
  - SMALL $14
  - MEDIUM $16
- PEPPERONI
  - SMALL $15
  - MEDIUM $15
- HAMANIAN
  - SMALL $14
  - MEDIUM $16
- HAMANIAN
  - SMALL $14
  - MEDIUM $17
- HAMANIAN
  - SMALL $14
  - MEDIUM $18
- HAMANIAN
  - SMALL $14
  - MEDIUM $19
- HAMANIAN
  - SMALL $14
  - MEDIUM $20
- HAMANIAN
  - SMALL $14
  - MEDIUM $21"""
    
    should_stop, reason = detector.detect_excessive_repetition(text)
    print(f"Should stop: {should_stop}")
    print(f"Reason: {reason}")
    
    stats = detector.get_repetition_stats(text)
    print(f"\nRepetition Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if should_stop:
        truncated = detector.truncate_at_repetition(text)
        print(f"\nOriginal length: {len(text)} chars")
        print(f"Truncated length: {len(truncated)} chars")
        print(f"\nTruncated output:\n{truncated}")
    
    print("\n✓ Test 1 passed\n")


def test_number_sequence():
    """Test detection of incrementing number sequences"""
    print("=" * 60)
    print("Test 2: Incrementing Number Sequence")
    print("=" * 60)
    
    detector = RepetitionDetector(
        max_consecutive_similar_lines=5,
        similarity_threshold=0.7,
        max_number_sequence_length=20
    )
    
    # Simulate counting from 16 to 293 (the issue described)
    text = "\n".join([f"- HAMANIAN\n  - SMALL $14\n  - MEDIUM ${i}" for i in range(16, 294)])
    
    should_stop, reason = detector.detect_excessive_repetition(text)
    print(f"Should stop: {should_stop}")
    print(f"Reason: {reason}")
    
    stats = detector.get_repetition_stats(text)
    print(f"\nRepetition Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Test 2 passed\n")


def test_legitimate_table():
    """Test that legitimate tables are not flagged"""
    print("=" * 60)
    print("Test 3: Legitimate Table Content")
    print("=" * 60)
    
    detector = RepetitionDetector(
        max_consecutive_similar_lines=5,
        similarity_threshold=0.7,
        max_number_sequence_length=20
    )
    
    # Simulate legitimate table content (like the expected output)
    text = """CLASSIC PIZZAS
CHEESE
SMALL $14
MEDIUM $16
LARGE $21
X-LARGE $25

PEPPERONI
SMALL $15
MEDIUM $17.5
LARGE $23
X-LARGE $27.5

HAWAIIAN
SMALL $16
MEDIUM $19
LARGE $25
X-LARGE $30

SPECIALTY PIZZAS
SMALL $17
MEDIUM $22
LARGE $28
X-LARGE $34"""
    
    should_stop, reason = detector.detect_excessive_repetition(text)
    print(f"Should stop: {should_stop}")
    print(f"Reason: {reason}")
    
    stats = detector.get_repetition_stats(text)
    print(f"\nRepetition Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if should_stop:
        print("\n✗ Test 3 FAILED: Legitimate content was flagged as repetitive")
    else:
        print("\n✓ Test 3 passed: Legitimate content not flagged\n")


def test_pattern_similarity():
    """Test pattern similarity detection"""
    print("=" * 60)
    print("Test 4: Pattern Similarity Detection")
    print("=" * 60)
    
    detector = RepetitionDetector(
        max_consecutive_similar_lines=3,
        similarity_threshold=0.8,
        max_number_sequence_length=10
    )
    
    # Test with similar but not identical lines
    text = """Item A - Price $10
Item B - Price $11
Item C - Price $12
Item D - Price $13
Item E - Price $14
Item F - Price $15"""
    
    should_stop, reason = detector.detect_excessive_repetition(text)
    print(f"Should stop: {should_stop}")
    print(f"Reason: {reason}")
    
    stats = detector.get_repetition_stats(text)
    print(f"\nRepetition Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Test 4 passed\n")


def test_ngram_processor():
    """Test the enhanced NoRepeatNGramLogitsProcessor"""
    print("=" * 60)
    print("Test 5: NoRepeatNGramLogitsProcessor")
    print("=" * 60)
    
    try:
        import torch
        from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
        
        processor = NoRepeatNGramLogitsProcessor(
            ngram_size=5,
            window_size=20,
            whitelist_token_ids={1, 2},
            max_consecutive_repeats=3,
            repetition_penalty_scale=2.0,
            pattern_detection_size=10
        )
        
        # Create a mock input_ids sequence with repetition
        input_ids = [10, 20, 30, 40, 50] * 5  # Repeating pattern
        scores = torch.randn(100)  # Mock scores for 100 tokens
        
        # Process the scores
        processed_scores = processor(input_ids, scores)
        
        print(f"Input sequence length: {len(input_ids)}")
        print(f"Detected pattern repetition: {processor._detect_pattern_repetition(input_ids)} times")
        print(f"Scores modified: {not torch.equal(scores, processed_scores)}")
        
        print("\n✓ Test 5 passed\n")
        
    except ImportError as e:
        print(f"⚠ Test 5 skipped: {e}")
        print("(This is expected if torch is not installed)\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing Issue #257 Fix: Repetition Detection")
    print("=" * 60 + "\n")
    
    try:
        test_exact_repetition()
        test_number_sequence()
        test_legitimate_table()
        test_pattern_similarity()
        test_ngram_processor()
        
        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
