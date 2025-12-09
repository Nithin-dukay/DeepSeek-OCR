"""
Test script to verify the prompt utilities for GitHub Issue #288
Tests prompt validation and utility functions without requiring torch
"""

import sys
import os

# Add the DeepSeek-OCR-vllm directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master', 'DeepSeek-OCR-vllm'))

from process.prompt_utils import (
    validate_prompt, 
    normalize_prompt, 
    sanitize_prompt,
    create_safe_prompt,
    get_recommended_ngram_params,
    get_prompt_template
)


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


def test_recommended_params():
    """Test recommended parameter generation"""
    print("=" * 60)
    print("TEST 4: Recommended N-gram Parameters")
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
    print("TEST 5: Prompt Templates")
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


def test_issue_288_specific():
    """Test the specific case from Issue #288"""
    print("=" * 60)
    print("TEST 6: Issue #288 Specific Case")
    print("=" * 60)
    
    # The original working prompt
    original_prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    
    # The modified prompt that caused issues
    modified_prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
    
    passed = 0
    failed = 0
    
    # Test 1: Both prompts should be valid
    print("\nTest 6.1: Both prompts should validate")
    is_valid_orig, _ = validate_prompt(original_prompt)
    is_valid_mod, _ = validate_prompt(modified_prompt)
    
    if is_valid_orig and is_valid_mod:
        print("✓ PASS: Both original and modified prompts are valid")
        passed += 1
    else:
        print("✗ FAIL: One or both prompts failed validation")
        print(f"  Original valid: {is_valid_orig}")
        print(f"  Modified valid: {is_valid_mod}")
        failed += 1
    
    # Test 2: Modified prompt should get different (more conservative) parameters
    print("\nTest 6.2: Modified prompt should get adjusted parameters")
    params_orig = get_recommended_ngram_params(original_prompt)
    params_mod = get_recommended_ngram_params(modified_prompt)
    
    if params_mod['min_generated_tokens'] >= params_orig['min_generated_tokens']:
        print("✓ PASS: Modified prompt gets more conservative parameters")
        print(f"  Original min_generated_tokens: {params_orig['min_generated_tokens']}")
        print(f"  Modified min_generated_tokens: {params_mod['min_generated_tokens']}")
        passed += 1
    else:
        print("✗ FAIL: Modified prompt should have higher min_generated_tokens")
        print(f"  Original: {params_orig}")
        print(f"  Modified: {params_mod}")
        failed += 1
    
    # Test 3: Sanitization should preserve the prompt content
    print("\nTest 6.3: Sanitization preserves content")
    sanitized = sanitize_prompt(modified_prompt)
    
    if "<|grounding|>" in sanitized and "dont add any extra space" in sanitized:
        print("✓ PASS: Sanitization preserves prompt content")
        print(f"  Sanitized: '{sanitized}'")
        passed += 1
    else:
        print("✗ FAIL: Sanitization altered prompt content")
        print(f"  Original:  '{modified_prompt}'")
        print(f"  Sanitized: '{sanitized}'")
        failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DeepSeek-OCR Issue #288 Fix - Prompt Utilities Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Prompt Validation", test_prompt_validation()))
    results.append(("Prompt Normalization", test_prompt_normalization()))
    results.append(("Safe Prompt Creation", test_safe_prompt_creation()))
    results.append(("Recommended Parameters", test_recommended_params()))
    results.append(("Prompt Templates", test_prompt_templates()))
    results.append(("Issue #288 Specific Case", test_issue_288_specific()))
    
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
        print("\n🎉 All tests passed! The prompt utilities are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_failed} test suite(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
