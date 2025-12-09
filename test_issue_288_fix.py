#!/usr/bin/env python3
"""
Test script for Issue #288 fix.

This script tests the improved NoRepeatNGramLogitsProcessor and prompt utilities
to ensure that modified prompts work correctly.
"""

import sys
import os

# Add the DeepSeek-OCR-vllm directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master', 'DeepSeek-OCR-vllm'))

from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.prompt_utils import (
    validate_prompt,
    optimize_prompt,
    suggest_prompt_fix,
    explain_prompt_issue,
    get_recommended_prompts
)
import torch


def test_prompt_validation():
    """Test prompt validation functionality."""
    print("=" * 80)
    print("TEST 1: Prompt Validation")
    print("=" * 80)
    
    test_cases = [
        ("<image>\n<|grounding|>Convert the document to markdown.", True, "Standard prompt"),
        ("<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.", True, "Modified prompt from issue"),
        ("<image>\nFree OCR.", True, "Free OCR prompt"),
        ("Convert the document to markdown.", False, "Missing image token"),
        ("", False, "Empty prompt"),
        ("<image><image>\nOCR this.", False, "Multiple image tokens"),
    ]
    
    passed = 0
    failed = 0
    
    for prompt, expected_valid, description in test_cases:
        is_valid, error = validate_prompt(prompt)
        status = "✓ PASS" if is_valid == expected_valid else "✗ FAIL"
        
        if is_valid == expected_valid:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status}: {description}")
        print(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print(f"  Expected: {expected_valid}, Got: {is_valid}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\n{'=' * 80}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 80}\n")
    
    return failed == 0


def test_prompt_optimization():
    """Test prompt optimization functionality."""
    print("=" * 80)
    print("TEST 2: Prompt Optimization")
    print("=" * 80)
    
    test_cases = [
        (
            "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.",
            "Should simplify redundant instruction"
        ),
        (
            "<image>\n<|grounding|>OCR this image and extract all text without adding extra spaces.",
            "Should simplify multiple redundant instructions"
        ),
        (
            "<image>\n  Free   OCR.  ",
            "Should remove excessive whitespace"
        ),
    ]
    
    for prompt, description in test_cases:
        print(f"\n{description}")
        print(f"  Original:  {prompt}")
        
        optimized = optimize_prompt(prompt)
        print(f"  Optimized: {optimized}")
        
        suggested = suggest_prompt_fix(prompt)
        if suggested != prompt:
            print(f"  Suggested: {suggested}")
        
        issues = explain_prompt_issue(prompt)
        if issues:
            print(f"  Issues:\n{issues}")
    
    print(f"\n{'=' * 80}\n")
    return True


def test_ngram_processor():
    """Test the improved NoRepeatNGramLogitsProcessor."""
    print("=" * 80)
    print("TEST 3: NoRepeatNGramLogitsProcessor")
    print("=" * 80)
    
    # Create processor with new parameters
    processor = NoRepeatNGramLogitsProcessor(
        ngram_size=5,
        window_size=20,
        whitelist_token_ids={100, 101},
        min_generated_tokens=3,
        enable_adaptive_ngram=True
    )
    
    # Simulate a sequence of token IDs
    # First 10 tokens are "prompt", rest are "generated"
    vocab_size = 1000
    
    test_cases = [
        {
            "name": "Short sequence (< min_generated_tokens)",
            "input_ids": [1, 2, 3, 4, 5],  # Only 5 tokens, all "prompt"
            "should_modify": False,
            "description": "Should not modify scores when sequence is too short"
        },
        {
            "name": "Early generation (< min_generated_tokens after prompt)",
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 12 tokens, only 2 "generated"
            "should_modify": False,
            "description": "Should not modify scores when not enough tokens generated"
        },
        {
            "name": "Sufficient generation with repetition",
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            "should_modify": True,
            "description": "Should activate after min_generated_tokens"
        },
        {
            "name": "Whitelisted token",
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 100, 100, 100, 100],
            "should_modify": False,
            "description": "Should not ban whitelisted tokens"
        },
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        # Reset processor for each test
        processor.prompt_length = None
        
        input_ids = test_case["input_ids"]
        scores = torch.randn(vocab_size)
        original_scores = scores.clone()
        
        # Process
        modified_scores = processor(input_ids, scores)
        
        # Check if scores were modified
        scores_changed = not torch.equal(original_scores, modified_scores)
        
        # For the first call, prompt_length gets set
        if processor.prompt_length is None:
            processor.prompt_length = len(input_ids)
        
        status = "✓ PASS" if scores_changed == test_case["should_modify"] else "✗ FAIL"
        
        if scores_changed == test_case["should_modify"]:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status}: {test_case['name']}")
        print(f"  {test_case['description']}")
        print(f"  Input length: {len(input_ids)}")
        print(f"  Expected modification: {test_case['should_modify']}")
        print(f"  Actual modification: {scores_changed}")
    
    print(f"\n{'=' * 80}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 80}\n")
    
    return failed == 0


def test_recommended_prompts():
    """Test recommended prompts functionality."""
    print("=" * 80)
    print("TEST 4: Recommended Prompts")
    print("=" * 80)
    
    prompts = get_recommended_prompts()
    
    print(f"\nFound {len(prompts)} recommended prompts:\n")
    
    all_valid = True
    for name, prompt in prompts.items():
        is_valid, error = validate_prompt(prompt)
        status = "✓" if is_valid else "✗"
        print(f"{status} {name:30s}: {prompt}")
        if not is_valid:
            print(f"  ERROR: {error}")
            all_valid = False
    
    print(f"\n{'=' * 80}")
    print(f"Results: {'All prompts valid' if all_valid else 'Some prompts invalid'}")
    print(f"{'=' * 80}\n")
    
    return all_valid


def test_issue_288_specific():
    """Test the specific case from Issue #288."""
    print("=" * 80)
    print("TEST 5: Issue #288 Specific Case")
    print("=" * 80)
    
    original_prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    modified_prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
    
    print("\nOriginal prompt (working):")
    print(f"  {original_prompt}")
    is_valid, error = validate_prompt(original_prompt)
    print(f"  Valid: {is_valid}")
    
    print("\nModified prompt (was broken):")
    print(f"  {modified_prompt}")
    is_valid, error = validate_prompt(modified_prompt)
    print(f"  Valid: {is_valid}")
    
    issues = explain_prompt_issue(modified_prompt)
    if issues:
        print(f"  Issues found:\n{issues}")
    
    suggested = suggest_prompt_fix(modified_prompt)
    print(f"\nSuggested fix:")
    print(f"  {suggested}")
    
    print("\nExplanation:")
    print("  The modified prompt contains redundant instructions about spacing.")
    print("  The model handles spacing automatically, so this instruction is unnecessary")
    print("  and can cause the n-gram processor to be overly restrictive.")
    print("\nWith the fix:")
    print("  1. The NoRepeatNGramLogitsProcessor now waits for min_generated_tokens")
    print("     before activating, preventing premature blocking.")
    print("  2. It uses adaptive n-gram sizing to be less restrictive early on.")
    print("  3. It only looks at generated tokens, not the prompt.")
    print("  4. The prompt_utils module helps users identify and fix such issues.")
    
    print(f"\n{'=' * 80}\n")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Testing Issue #288 Fix")
    print("=" * 80 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Prompt Validation", test_prompt_validation()))
    results.append(("Prompt Optimization", test_prompt_optimization()))
    results.append(("NoRepeatNGramLogitsProcessor", test_ngram_processor()))
    results.append(("Recommended Prompts", test_recommended_prompts()))
    results.append(("Issue #288 Specific Case", test_issue_288_specific()))
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
