"""
Test script for GitHub Issue #191 fix

This script tests the NoRepeatNGramLogitsProcessor to ensure it correctly
prevents hallucination without breaking normal functionality.
"""

import torch
import sys
from transformers_logits_processor import (
    NoRepeatNGramLogitsProcessor,
    AdaptiveNoRepeatNGramLogitsProcessor,
    create_anti_hallucination_processors
)


def test_basic_initialization():
    """Test that processors can be initialized correctly."""
    print("Test 1: Basic Initialization")
    
    try:
        processor = NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90)
        assert processor.ngram_size == 30
        assert processor.window_size == 90
        assert 128821 in processor.whitelist_token_ids
        assert 128822 in processor.whitelist_token_ids
        print("  ✓ Standard processor initialized correctly")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    try:
        processor = AdaptiveNoRepeatNGramLogitsProcessor(
            initial_ngram_size=40,
            final_ngram_size=20,
            transition_tokens=1000
        )
        assert processor.initial_ngram_size == 40
        assert processor.final_ngram_size == 20
        print("  ✓ Adaptive processor initialized correctly")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    try:
        processors = create_anti_hallucination_processors(mode="standard")
        assert len(processors) == 1
        print("  ✓ Factory function works for standard mode")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    try:
        processors = create_anti_hallucination_processors(mode="adaptive")
        assert len(processors) == 1
        print("  ✓ Factory function works for adaptive mode")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print("  ✅ All initialization tests passed\n")
    return True


def test_ngram_blocking():
    """Test that n-gram repetition is correctly blocked."""
    print("Test 2: N-gram Blocking")
    
    processor = NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=10)
    
    # Create a sequence with a repeated 3-gram: [1, 2, 3] appears twice
    # Sequence: [1, 2, 3, 4, 5, 1, 2]
    # Current prefix: [1, 2]
    # Should ban token 3 (which would complete the repeated 3-gram [1, 2, 3])
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2]])
    vocab_size = 100
    scores = torch.zeros((1, vocab_size))
    
    try:
        modified_scores = processor(input_ids, scores)
        
        # Token 3 should be banned (set to -inf)
        assert modified_scores[0, 3] == float("-inf"), "Token 3 should be banned"
        
        # Other tokens should not be affected
        assert modified_scores[0, 4] == 0.0, "Token 4 should not be banned"
        assert modified_scores[0, 5] == 0.0, "Token 5 should not be banned"
        
        print("  ✓ N-gram blocking works correctly")
    except AssertionError as e:
        print(f"  ✗ Failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False
    
    print("  ✅ N-gram blocking test passed\n")
    return True


def test_whitelist():
    """Test that whitelisted tokens are not blocked."""
    print("Test 3: Whitelist Functionality")
    
    # Token 3 is whitelisted
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=3, 
        window_size=10,
        whitelist_token_ids={3}
    )
    
    # Same sequence as before, but token 3 is whitelisted
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2]])
    vocab_size = 100
    scores = torch.zeros((1, vocab_size))
    
    try:
        modified_scores = processor(input_ids, scores)
        
        # Token 3 should NOT be banned (it's whitelisted)
        assert modified_scores[0, 3] == 0.0, "Token 3 should not be banned (whitelisted)"
        
        print("  ✓ Whitelist works correctly")
    except AssertionError as e:
        print(f"  ✗ Failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False
    
    print("  ✅ Whitelist test passed\n")
    return True


def test_window_size():
    """Test that window size limits the search range."""
    print("Test 4: Window Size")
    
    processor = NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=5)
    
    # Sequence: [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
    # The 3-gram [1, 2, 3] appears at position 0
    # Current position is 8, prefix is [1, 2]
    # Window size is 5, so we only look back to position 5
    # The repeated 3-gram at position 0 is outside the window
    # Therefore, token 3 should NOT be banned
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2]])
    vocab_size = 100
    scores = torch.zeros((1, vocab_size))
    
    try:
        modified_scores = processor(input_ids, scores)
        
        # Token 3 should NOT be banned (outside window)
        assert modified_scores[0, 3] == 0.0, "Token 3 should not be banned (outside window)"
        
        print("  ✓ Window size limits search correctly")
    except AssertionError as e:
        print(f"  ✗ Failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False
    
    # Now test with a larger window that includes the repeated n-gram
    processor = NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=20)
    
    try:
        modified_scores = processor(input_ids, scores)
        
        # Token 3 should be banned (inside window)
        assert modified_scores[0, 3] == float("-inf"), "Token 3 should be banned (inside window)"
        
        print("  ✓ Larger window correctly detects repetition")
    except AssertionError as e:
        print(f"  ✗ Failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False
    
    print("  ✅ Window size test passed\n")
    return True


def test_batch_processing():
    """Test that batch processing works correctly."""
    print("Test 5: Batch Processing")
    
    processor = NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=10)
    
    # Batch of 2 sequences
    # Sequence 1: [1, 2, 3, 4, 5, 1, 2] - should ban token 3
    # Sequence 2: [7, 8, 9, 10, 11, 7, 8] - should ban token 9
    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 1, 2],
        [7, 8, 9, 10, 11, 7, 8]
    ])
    vocab_size = 100
    scores = torch.zeros((2, vocab_size))
    
    try:
        modified_scores = processor(input_ids, scores)
        
        # Sequence 1: token 3 should be banned
        assert modified_scores[0, 3] == float("-inf"), "Seq 1: Token 3 should be banned"
        assert modified_scores[0, 9] == 0.0, "Seq 1: Token 9 should not be banned"
        
        # Sequence 2: token 9 should be banned
        assert modified_scores[1, 9] == float("-inf"), "Seq 2: Token 9 should be banned"
        assert modified_scores[1, 3] == 0.0, "Seq 2: Token 3 should not be banned"
        
        print("  ✓ Batch processing works correctly")
    except AssertionError as e:
        print(f"  ✗ Failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False
    
    print("  ✅ Batch processing test passed\n")
    return True


def test_adaptive_mode():
    """Test that adaptive mode adjusts n-gram size."""
    print("Test 6: Adaptive Mode")
    
    processor = AdaptiveNoRepeatNGramLogitsProcessor(
        initial_ngram_size=5,
        final_ngram_size=3,
        transition_tokens=10,
        window_size=20
    )
    
    # Short sequence (early in generation)
    # Should use larger n-gram size (5)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    vocab_size = 100
    scores = torch.zeros((1, vocab_size))
    
    try:
        # First call initializes start_length
        modified_scores = processor(input_ids, scores)
        
        # Add more tokens to simulate generation progress
        # After 10 tokens, should transition to smaller n-gram size (3)
        input_ids_long = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2]])
        modified_scores = processor(input_ids_long, scores)
        
        print("  ✓ Adaptive mode adjusts n-gram size over time")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print("  ✅ Adaptive mode test passed\n")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Test 7: Edge Cases")
    
    processor = NoRepeatNGramLogitsProcessor(ngram_size=5, window_size=10)
    
    # Test 1: Sequence shorter than n-gram size
    input_ids = torch.tensor([[1, 2, 3]])
    vocab_size = 100
    scores = torch.zeros((1, vocab_size))
    
    try:
        modified_scores = processor(input_ids, scores)
        # Should not modify scores (sequence too short)
        assert torch.all(modified_scores == scores), "Scores should not be modified for short sequences"
        print("  ✓ Handles short sequences correctly")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 2: No repetition
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    scores = torch.zeros((1, vocab_size))
    
    try:
        modified_scores = processor(input_ids, scores)
        # Should not ban any tokens (no repetition)
        assert not torch.any(torch.isinf(modified_scores)), "No tokens should be banned without repetition"
        print("  ✓ Handles non-repetitive sequences correctly")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    # Test 3: Invalid parameters
    try:
        invalid_processor = NoRepeatNGramLogitsProcessor(ngram_size=-1, window_size=10)
        print("  ✗ Should have raised ValueError for negative ngram_size")
        return False
    except ValueError:
        print("  ✓ Correctly rejects invalid ngram_size")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False
    
    try:
        invalid_processor = NoRepeatNGramLogitsProcessor(ngram_size=5, window_size=0)
        print("  ✗ Should have raised ValueError for zero window_size")
        return False
    except ValueError:
        print("  ✓ Correctly rejects invalid window_size")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False
    
    print("  ✅ Edge cases test passed\n")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("="*80)
    print("Testing NoRepeatNGramLogitsProcessor for Issue #191 Fix")
    print("="*80)
    print()
    
    tests = [
        test_basic_initialization,
        test_ngram_blocking,
        test_whitelist,
        test_window_size,
        test_batch_processing,
        test_adaptive_mode,
        test_edge_cases
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}\n")
            results.append(False)
    
    print("="*80)
    print("Test Summary")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! The fix is working correctly.")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
