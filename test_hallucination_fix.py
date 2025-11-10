"""
Test script to verify the hallucination fix implementation.
This tests the NoRepeatNGramLogitsProcessor without requiring the full model.
"""

import sys
import os
import torch

# Add path to import the processor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-hf'))

from ngram_norepeat_hf import NoRepeatNGramLogitsProcessor, get_recommended_params


def test_processor_initialization():
    """Test that the processor can be initialized correctly."""
    print("Test 1: Processor Initialization")
    print("-" * 50)
    
    try:
        # Test basic initialization
        processor = NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90)
        print("✓ Basic initialization successful")
        
        # Test with whitelist
        processor = NoRepeatNGramLogitsProcessor(
            ngram_size=30, 
            window_size=90, 
            whitelist_token_ids={128821, 128822}
        )
        print("✓ Initialization with whitelist successful")
        
        # Test invalid parameters
        try:
            processor = NoRepeatNGramLogitsProcessor(ngram_size=-1)
            print("✗ Should have raised ValueError for negative ngram_size")
            return False
        except ValueError:
            print("✓ Correctly raises ValueError for invalid ngram_size")
        
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def test_get_recommended_params():
    """Test the recommended parameters function."""
    print("\nTest 2: Recommended Parameters")
    print("-" * 50)
    
    try:
        document_types = ["handwritten", "printed", "table", "general"]
        
        for doc_type in document_types:
            params = get_recommended_params(doc_type)
            print(f"\n{doc_type.capitalize()} document:")
            print(f"  ngram_size: {params['ngram_size']}")
            print(f"  window_size: {params['window_size']}")
            print(f"  whitelist_token_ids: {params['whitelist_token_ids']}")
            
            # Verify parameters are valid
            assert isinstance(params['ngram_size'], int) and params['ngram_size'] > 0
            assert isinstance(params['window_size'], int) and params['window_size'] > 0
            assert isinstance(params['whitelist_token_ids'], set)
        
        print("\n✓ All recommended parameters are valid")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_processor_logic():
    """Test the core logic of the processor."""
    print("\nTest 3: Processor Logic")
    print("-" * 50)
    
    try:
        processor = NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=10)
        
        # Create a mock sequence with repetition
        # Sequence: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]
        # The next token should NOT be 3 (would complete the repeated trigram [1,2,3])
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]])
        
        # Create mock scores (logits) for next token
        vocab_size = 100
        scores = torch.zeros((1, vocab_size))
        scores[0, 3] = 10.0  # Token 3 has high score
        scores[0, 50] = 5.0  # Token 50 has medium score
        
        # Apply processor
        modified_scores = processor(input_ids, scores)
        
        # Token 3 should be banned (set to -inf)
        if modified_scores[0, 3] == float('-inf'):
            print("✓ Correctly banned repeated n-gram token")
        else:
            print(f"✗ Failed to ban repeated token. Score: {modified_scores[0, 3]}")
            return False
        
        # Token 50 should not be affected
        if modified_scores[0, 50] == 5.0:
            print("✓ Non-repeated tokens unchanged")
        else:
            print(f"✗ Non-repeated token was modified. Score: {modified_scores[0, 50]}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Logic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_whitelist_functionality():
    """Test that whitelisted tokens are not banned."""
    print("\nTest 4: Whitelist Functionality")
    print("-" * 50)
    
    try:
        # Token 3 is whitelisted
        processor = NoRepeatNGramLogitsProcessor(
            ngram_size=3, 
            window_size=10,
            whitelist_token_ids={3}
        )
        
        # Same sequence as before
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2]])
        scores = torch.zeros((1, 100))
        scores[0, 3] = 10.0
        
        # Apply processor
        modified_scores = processor(input_ids, scores)
        
        # Token 3 should NOT be banned because it's whitelisted
        if modified_scores[0, 3] == 10.0:
            print("✓ Whitelisted token not banned")
            return True
        else:
            print(f"✗ Whitelisted token was banned. Score: {modified_scores[0, 3]}")
            return False
    except Exception as e:
        print(f"✗ Whitelist test failed: {e}")
        return False


def test_batch_processing():
    """Test that the processor works with batched inputs."""
    print("\nTest 5: Batch Processing")
    print("-" * 50)
    
    try:
        processor = NoRepeatNGramLogitsProcessor(ngram_size=3, window_size=10)
        
        # Batch of 2 sequences
        input_ids = torch.tensor([
            [1, 2, 3, 4, 5, 1, 2],  # Sequence 1
            [5, 6, 7, 8, 9, 5, 6]   # Sequence 2
        ])
        
        scores = torch.zeros((2, 100))
        scores[0, 3] = 10.0  # Should be banned in sequence 1
        scores[1, 7] = 10.0  # Should be banned in sequence 2
        
        modified_scores = processor(input_ids, scores)
        
        # Check both sequences
        if modified_scores[0, 3] == float('-inf') and modified_scores[1, 7] == float('-inf'):
            print("✓ Batch processing works correctly")
            return True
        else:
            print(f"✗ Batch processing failed")
            print(f"  Seq 1, token 3: {modified_scores[0, 3]}")
            print(f"  Seq 2, token 7: {modified_scores[1, 7]}")
            return False
    except Exception as e:
        print(f"✗ Batch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_short_sequence():
    """Test that short sequences (< ngram_size) are handled correctly."""
    print("\nTest 6: Short Sequence Handling")
    print("-" * 50)
    
    try:
        processor = NoRepeatNGramLogitsProcessor(ngram_size=5, window_size=10)
        
        # Sequence shorter than ngram_size
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.ones((1, 100))
        
        modified_scores = processor(input_ids, scores)
        
        # Scores should be unchanged
        if torch.equal(scores, modified_scores):
            print("✓ Short sequences handled correctly (no modification)")
            return True
        else:
            print("✗ Short sequences were incorrectly modified")
            return False
    except Exception as e:
        print(f"✗ Short sequence test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 50)
    print("Testing Hallucination Fix Implementation")
    print("=" * 50)
    
    tests = [
        test_processor_initialization,
        test_get_recommended_params,
        test_processor_logic,
        test_whitelist_functionality,
        test_batch_processing,
        test_short_sequence,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! The hallucination fix is working correctly.")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
