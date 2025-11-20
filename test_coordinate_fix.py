#!/usr/bin/env python3
"""
Test script to verify the coordinate sanitization fix for GitHub Issue #278
"""
import re


def sanitize_coordinate_string(coord_str):
    """
    Sanitize malformed coordinate strings before parsing.
    Handles cases like '[[550, s 331, 652, 345]]' where 's 331' should be '331'.
    
    Args:
        coord_str: The coordinate string to sanitize
        
    Returns:
        Sanitized coordinate string
    """
    # Remove patterns like 's <number>' and replace with just '<number>'
    # This handles the case where model generates 's 331' instead of '331'
    sanitized = re.sub(r'\bs\s+(\d+)', r'\1', coord_str)
    
    # Remove any other stray single letters followed by spaces before numbers
    sanitized = re.sub(r'\b[a-zA-Z]\s+(\d+)', r'\1', sanitized)
    
    # Clean up any extra spaces within brackets
    sanitized = re.sub(r'\[\s+', '[', sanitized)
    sanitized = re.sub(r'\s+\]', ']', sanitized)
    sanitized = re.sub(r',\s+', ', ', sanitized)
    
    return sanitized


def test_coordinate_sanitization():
    """Test various coordinate string scenarios"""
    
    test_cases = [
        {
            "name": "GitHub Issue #278 - 's 331' pattern",
            "input": "[[550, s 331, 652, 345]]",
            "expected": "[[550, 331, 652, 345]]",
        },
        {
            "name": "Valid coordinates - should not change",
            "input": "[[100, 200, 300, 400]]",
            "expected": "[[100, 200, 300, 400]]",
        },
        {
            "name": "Multiple coordinates with 's' pattern",
            "input": "[[550, s 331, 652, 345], [100, s 200, 300, 400]]",
            "expected": "[[550, 331, 652, 345], [100, 200, 300, 400]]",
        },
        {
            "name": "Other single letter patterns",
            "input": "[[550, a 331, 652, b 345]]",
            "expected": "[[550, 331, 652, 345]]",
        },
        {
            "name": "Extra spaces in brackets",
            "input": "[[  550,  331,  652,  345  ]]",
            "expected": "[[550, 331, 652, 345]]",
        },
        {
            "name": "Mixed valid and malformed",
            "input": "[[100, 200, 300, 400], [550, s 331, 652, 345]]",
            "expected": "[[100, 200, 300, 400], [550, 331, 652, 345]]",
        },
    ]
    
    print("=" * 80)
    print("Testing Coordinate Sanitization Fix for GitHub Issue #278")
    print("=" * 80)
    print()
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  Input:    {test_case['input']}")
        
        result = sanitize_coordinate_string(test_case['input'])
        print(f"  Output:   {result}")
        print(f"  Expected: {test_case['expected']}")
        
        # Test if the result can be evaluated
        try:
            parsed = eval(result)
            print(f"  Parsed:   {parsed}")
            can_parse = True
        except Exception as e:
            print(f"  Parse Error: {e}")
            can_parse = False
        
        # Check if output matches expected
        passed = result == test_case['expected'] and can_parse
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Status:   {status}")
        print()
        
        if not passed:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = test_coordinate_sanitization()
    exit(0 if success else 1)
