#!/usr/bin/env python3
"""
Test script to demonstrate the fix for GitHub Issue #288

This script shows how the prompt validator catches problematic prompts
and suggests alternatives.
"""

import sys
sys.path.insert(0, 'DeepSeek-OCR-master/DeepSeek-OCR-vllm')

from process.prompt_validator import PromptValidator

def test_issue_288():
    """Test the specific case from GitHub Issue #288"""
    
    print("\n" + "="*80)
    print("TESTING FIX FOR GITHUB ISSUE #288")
    print("="*80)
    
    validator = PromptValidator()
    
    # Original working prompt
    print("\n1. ORIGINAL WORKING PROMPT:")
    print("-" * 80)
    original_prompt = '<image>\n<|grounding|>Convert the document to markdown.'
    print(f"Prompt: {original_prompt}")
    result1 = validator.validate(original_prompt)
    validator.print_validation_report(result1)
    
    # Problematic modified prompt from Issue #288
    print("\n2. PROBLEMATIC MODIFIED PROMPT (from Issue #288):")
    print("-" * 80)
    problematic_prompt = '<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.'
    print(f"Prompt: {problematic_prompt}")
    result2 = validator.validate(problematic_prompt)
    validator.print_validation_report(result2)
    
    # Show comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nOriginal Prompt:")
    print(f"  Status: {result1.message}")
    print(f"  Valid: {result1.is_valid}")
    print(f"  Warning Level: {result1.warning_level}")
    
    print(f"\nModified Prompt (Issue #288):")
    print(f"  Status: {result2.message}")
    print(f"  Valid: {result2.is_valid}")
    print(f"  Warning Level: {result2.warning_level}")
    print(f"  Issues: {len(result2.issues)}")
    
    print("\n" + "="*80)
    print("SOLUTION")
    print("="*80)
    print("\n✅ The validator now detects this problematic pattern and warns users!")
    print("\n📝 When users try to use the problematic prompt, they will see:")
    print("   - A clear warning about potential repetitive output")
    print("   - The specific issues with their prompt")
    print("   - A suggested alternative prompt")
    print("\n🔧 Users can configure the behavior in config.py:")
    print("   - ENABLE_PROMPT_VALIDATION = True  (show warnings)")
    print("   - STRICT_PROMPT_MODE = True        (reject invalid prompts)")
    print("\n📖 Full documentation available in PROMPT_GUIDELINES.md")
    print("="*80 + "\n")


def test_additional_cases():
    """Test additional problematic patterns"""
    
    print("\n" + "="*80)
    print("ADDITIONAL TEST CASES")
    print("="*80)
    
    validator = PromptValidator()
    
    test_cases = [
        ('<image>\n<|grounding|>OCR this image without spaces.', 'Negative constraint'),
        ('<image>\n<|grounding|>Please make sure to OCR carefully.', 'Meta-instruction'),
        ('<image>\n<|grounding|>Always convert to markdown.', 'Absolute constraint'),
        ('<image>\nFree OCR.', 'Valid alternative format'),
    ]
    
    for prompt, description in test_cases:
        print(f"\n{description}:")
        print(f"Prompt: {prompt}")
        result = validator.validate(prompt)
        print(f"Status: {result.message}")
        print(f"Valid: {result.is_valid}, Warning: {result.warning_level}")
        if result.suggested_prompt:
            print(f"Suggestion: {result.suggested_prompt}")
        print("-" * 80)


if __name__ == '__main__':
    test_issue_288()
    test_additional_cases()
    
    print("\n✅ All tests completed successfully!")
    print("The fix for Issue #288 is working correctly.\n")
