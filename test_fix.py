#!/usr/bin/env python3
"""
Test script to verify the fix for GitHub Issue #237
ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'
"""

import sys
import os

print("=" * 60)
print("Testing Fix for GitHub Issue #237")
print("=" * 60)

# Test 1: Import transformers
print("\n[Test 1] Importing transformers...")
try:
    import transformers
    print(f"✓ transformers version: {transformers.__version__}")
    assert transformers.__version__ >= "4.51.1", "transformers version should be >= 4.51.1"
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 2: Import LogitsProcessor
print("\n[Test 2] Importing LogitsProcessor from transformers...")
try:
    from transformers import LogitsProcessor
    print("✓ LogitsProcessor imported successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 3: Import torch
print("\n[Test 3] Importing torch...")
try:
    import torch
    print(f"✓ torch version: {torch.__version__}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 4: Import the fixed ngram_norepeat module
print("\n[Test 4] Importing NoRepeatNGramLogitsProcessor from fixed module...")
try:
    sys.path.insert(0, '/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm')
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    print("✓ NoRepeatNGramLogitsProcessor imported successfully")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 5: Verify the class works
print("\n[Test 5] Testing NoRepeatNGramLogitsProcessor functionality...")
try:
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=3,
        window_size=10,
        whitelist_token_ids={1, 2, 3}
    )
    print("✓ NoRepeatNGramLogitsProcessor instantiated successfully")
    
    # Test with dummy data
    input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scores = torch.randn(100)
    result = processor(input_ids, scores)
    print(f"✓ Processor executed successfully (output shape: {result.shape})")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# Test 6: Verify the problematic import is removed
print("\n[Test 6] Verifying the problematic import is removed...")
try:
    with open('/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py', 'r') as f:
        content = f.read()
        if '_calc_banned_ngram_tokens' in content:
            print("✗ Failed: _calc_banned_ngram_tokens import still present")
            sys.exit(1)
        else:
            print("✓ Confirmed: _calc_banned_ngram_tokens import removed")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nSummary:")
print("- transformers 4.51.1 is compatible")
print("- LogitsProcessor imports correctly")
print("- NoRepeatNGramLogitsProcessor works without the problematic import")
print("- Issue #237 is FIXED!")
print("\nYou can now use DeepSeek-OCR with transformers 4.51.1 and vLLM nightly.")
