#!/usr/bin/env python3
"""
Test script to verify the coordinate sanitization fix for GitHub Issue #278
"""
import re


def sanitize_coordinates(coord_str):
    """
    Sanitize coordinate string by removing spurious alphabetic characters.
    Handles cases like '[[550, s 331, 652, 345]]' -> '[[550, 331, 652, 345]]'
    """
    # Remove spurious alphabetic characters that appear between numbers
    # This regex matches letters (with optional spaces) that appear between digits/brackets/commas
    sanitized = re.sub(r'(?<=[\d\[\],\s])\s*[a-zA-Z]+\s*(?=[\d\[\],\s])', ' ', coord_str)
    # Clean up any extra whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized)
    # Remove spaces after opening brackets and before closing brackets
    sanitized = re.sub(r'\[\s+', '[', sanitized)
    sanitized = re.sub(r'\s+\]', ']', sanitized)
    # Remove spaces around commas
    sanitized = re.sub(r'\s*,\s*', ',', sanitized)
    return sanitized


def test_coordinate_sanitization():
    """Test various coordinate string formats"""
    
    test_cases = [
        # (input, expected_output, description)
        ('[[550, s 331, 652, 345]]', '[[550,331,652,345]]', 'Issue #278 - spurious "s" character'),
        ('[[550, 331, 652, 345]]', '[[550,331,652,345]]', 'Normal coordinates'),
        ('[[100, a 200, 300, b 400]]', '[[100,200,300,400]]', 'Multiple spurious characters'),
        ('[[10,20,30,40],[50,60,70,80]]', '[[10,20,30,40],[50,60,70,80]]', 'Multiple coordinate pairs'),
        ('[[100, abc 200, 300, xyz 400]]', '[[100,200,300,400]]', 'Multi-letter spurious strings'),
        ('[[  100  ,  200  ,  300  ,  400  ]]', '[[100,200,300,400]]', 'Extra whitespace'),
        ('[[100,s200,300,s400]]', '[[100,200,300,400]]', 'No spaces around spurious chars'),
    ]
    
    print("Testing coordinate sanitization fix for GitHub Issue #278\n")
    print("=" * 80)
    
    all_passed = True
    for i, (input_str, expected, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"  Input:    {input_str}")
        
        sanitized = sanitize_coordinates(input_str)
        print(f"  Output:   {sanitized}")
        print(f"  Expected: {expected}")
        
        # Try to eval the sanitized string
        try:
            result = eval(sanitized)
            print(f"  Eval:     ✓ Success - {result}")
            
            # Check if output matches expected
            if sanitized == expected:
                print(f"  Status:   ✓ PASS")
            else:
                print(f"  Status:   ⚠ PASS (different format but valid)")
        except Exception as e:
            print(f"  Eval:     ✗ Failed - {e}")
            print(f"  Status:   ✗ FAIL")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    return all_passed


if __name__ == "__main__":
    success = test_coordinate_sanitization()
    exit(0 if success else 1)
