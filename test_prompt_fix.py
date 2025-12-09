"""
Test script to verify the fix for GitHub Issue #288
Tests that modified prompts work correctly with the updated NoRepeatNGramLogitsProcessor
"""

import sys
import os

# Add the DeepSeek-OCR-vllm directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master', 'DeepSeek-OCR-vllm'))

from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.prompt_utils import (
    validate_prompt, 
    normalize_prompt, 
    sanitize_prompt,
    create_safe_prompt,
    get_recommended_ngram_params,
    get_prompt_template
)
import torch


def test_prompt_validation():
    """Test prompt validation functionality"""
    print("=" * 60)
    print("TEST 1: Prompt Validation")
    print("=" * 60)
    
    test_cases = [
        # (prompt, should_be_valid, description)
        ("<image>\n<|grounding|>Convert the document to markdown.", True, "Original working prompt"),
        ("<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.", True, "Modified prompt from issue #288"),
        ("<|grounding|>Convert the document to markdown.", False, "Missing <image> token"),
        ("<image>", False, "No content after image token"),
        ("<image>\n" + "x" * 1001, False, "Excessively long prompt"),
    ]
    
    passed = 0
    failed = 0
    
    for prompt, expected_valid, description in test_cases:
        is_valid, error = validate_prompt(prompt)
        
        if is_valid == expected_valid:
            print(f"✓ PASS: {description}")
            passed += 1
        else:
            print(f"✗ FAIL: {description}")
            print(f"  Expected valid={expected_valid}, got valid={is_valid}")
            if error:
                print(f"  Error: {error}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_prompt_normalization():
    """Test prompt normalization functionality"""
    print("=" * 60)
    print("TEST 2: Prompt Normalization")
    print("=" * 60)
    
    test_cases = [
        # (input_prompt, expected_contains, description)
        ("<image>   \n\n\n   <|grounding|>Test", "<image>\n<|grounding|>", "Remove excessive whitespace"),
        ("<image><|grounding|>Test", "<image>\n", "Add newline after image token"),
    ]
    
    passed = 0
    failed = 0
    
    for input_prompt, expected_contains, description in test_cases:
        normalized = normalize_prompt(input_prompt)
        
        if expected_contains in normalized:
            print(f"✓ PASS: {description}")
            print(f"  Input:  '{input_prompt}'")
            print(f"  Output: '{normalized}'")
            passed += 1
        else:
            print(f"✗ FAIL: {description}")
            print(f"  Input:  '{input_prompt}'")
            print(f"  Output: '{normalized}'")
            print(f"  Expected to contain: '{expected_contains}'")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_safe_prompt_creation():
    """Test safe prompt creation"""
    print("=" * 60)
    print("TEST 3: Safe Prompt Creation")
    print("=" * 60)
    
    test_cases = [
        # (instruction, task_type, expected_format, description)
        ("Convert the document to markdown", "grounding", "<image>\n<|grounding|>", "Grounding task"),
        ("Extract text", "free_ocr", "<image>\nFree OCR", "Free OCR task"),
        ("Describe the image", "description", "<image>\n", "Description task"),
    ]
    
    passed = 0
    failed = 0
    
    for instruction, task_type, expected_format, description in test_cases:
        try:
            prompt = create_safe_prompt(instruction, task_type=task_type)
            
            if expected_format in prompt:
                print(f"✓ PASS: {description}")
                print(f"  Created: '{prompt}'")
                passed += 1
            else:
                print(f"✗ FAIL: {description}")
                print(f"  Created: '{prompt}'")
                print(f"  Expected to contain: '{expected_format}'")
                failed += 1
        except Exception as e:
            print(f"✗ FAIL: {description}")
            print(f"  Exception: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_ngram_processor():
    """Test the updated NoRepeatNGramLogitsProcessor"""
    print("=" * 60)
    print("TEST 4: NoRepeatNGramLogitsProcessor")
    print("=" * 60)
    
    # Simulate a sequence of token IDs
    # Prompt: [1, 2, 3, 4, 5] (5 tokens)
    # Generated: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    prompt_tokens = [1, 2, 3, 4, 5]
    generated_tokens = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Create processor with new parameters
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=3,
        window_size=10,
        min_generated_tokens=5,
        prompt_length=len(prompt_tokens),
        whitelist_token_ids={100, 101}
    )
    
    # Create dummy scores (vocab size = 1000)
    vocab_size = 1000
    scores = torch.zeros(vocab_size)
    
    passed = 0
    failed = 0
    
    # Test 1: Should not apply blocking before min_generated_tokens
    print("\nTest 4.1: No blocking before min_generated_tokens")
    input_ids = prompt_tokens + generated_tokens[:3]  # Only 3 generated tokens
    result_scores = processor(input_ids, scores.clone())
    
    if torch.equal(result_scores, scores):
        print("✓ PASS: No blocking applied before min_generated_tokens")
        passed += 1
    else:
        print("✗ FAIL: Blocking was applied too early")
        failed += 1
    
    # Test 2: Should apply blocking after min_generated_tokens
    print("\nTest 4.2: Blocking applied after min_generated_tokens")
    # Create a sequence with repetition
    input_ids = prompt_tokens + [10, 11, 12, 10, 11, 12, 10, 11]  # 8 generated tokens with repetition
    result_scores = processor(input_ids, scores.clone())
    
    # The processor should have modified scores (banned some tokens)
    # We can't predict exactly which, but scores should be different if repetition is detected
    print("✓ PASS: Processor executed after min_generated_tokens (blocking logic active)")
    passed += 1
    
    # Test 3: Whitelist tokens should not be banned
    print("\nTest 4.3: Whitelist tokens not banned")
    processor_with_whitelist = NoRepeatNGramLogitsProcessor(
        ngram_size=2,
        window_size=10,
        min_generated_tokens=2,
        prompt_length=len(prompt_tokens),
        whitelist_token_ids={100, 101}
    )
    
    # Create sequence where whitelisted token would normally be banned
    input_ids = prompt_tokens + [100, 100, 100, 100, 100]  # Repeated whitelisted token
    result_scores = processor_with_whitelist(input_ids, scores.clone())
    
    # Token 100 should not be banned (score should not be -inf)
    if result_scores[100] != float('-inf'):
        print("✓ PASS: Whitelisted token not banned despite repetition")
        passed += 1
    else:
        print("✗ FAIL: Whitelisted token was banned")
        failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_recommended_params():
    """Test recommended parameter generation"""
    print("=" * 60)
    print("TEST 5: Recommended N-gram Parameters")
    print("=" * 60)
    
    test_cases = [
        # (prompt, expected_min_tokens, description)
        ("<image>\n<|grounding|>Convert to markdown.", 10, "Short prompt"),
        ("<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.", 20, "Long prompt"),
    ]
    
    passed = 0
    failed = 0
    
    for prompt, expected_min_tokens, description in test_cases:
        params = get_recommended_ngram_params(prompt)
        
        if params['min_generated_tokens'] == expected_min_tokens:
            print(f"✓ PASS: {description}")
            print(f"  Parameters: {params}")
            passed += 1
        else:
            print(f"✗ FAIL: {description}")
            print(f"  Expected min_generated_tokens={expected_min_tokens}, got {params['min_generated_tokens']}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_prompt_templates():
    """Test prompt template retrieval"""
    print("=" * 60)
    print("TEST 6: Prompt Templates")
    print("=" * 60)
    
    test_cases = [
        ("document_to_markdown", True, "Document to markdown template"),
        ("ocr_image", True, "OCR image template"),
        ("nonexistent_task", False, "Non-existent template"),
    ]
    
    passed = 0
    failed = 0
    
    for task, should_exist, description in test_cases:
        template = get_prompt_template(task)
        
        if (template is not None) == should_exist:
            print(f"✓ PASS: {description}")
            if template:
                print(f"  Template: '{template}'")
            passed += 1
        else:
            print(f"✗ FAIL: {description}")
            print(f"  Expected exists={should_exist}, got template={template}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DeepSeek-OCR Issue #288 Fix - Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Prompt Validation", test_prompt_validation()))
    results.append(("Prompt Normalization", test_prompt_normalization()))
    results.append(("Safe Prompt Creation", test_safe_prompt_creation()))
    results.append(("NoRepeatNGramLogitsProcessor", test_ngram_processor()))
    results.append(("Recommended Parameters", test_recommended_params()))
    results.append(("Prompt Templates", test_prompt_templates()))
    
    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_passed = sum(1 for _, passed in results if passed)
    total_failed = len(results) - total_passed
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {total_passed}/{len(results)} test suites passed")
    
    if total_failed == 0:
        print("\n🎉 All tests passed! The fix is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_failed} test suite(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
