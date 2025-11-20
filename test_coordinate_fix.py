#!/usr/bin/env python3
"""
Test script to verify the coordinate sanitization fix for GitHub Issue #278
"""
import re

def sanitize_coordinates(coord_str):
    """
    Sanitize coordinate strings by removing invalid patterns.
    
    Handles cases like:
    - "[[550, s 331, 652, 345]]" -> "[[550, 331, 652, 345]]"
    - Removes letter + space patterns before numbers
    
    Args:
        coord_str: The coordinate string to sanitize
        
    Returns:
        Sanitized coordinate string
    """
    # Remove patterns like "s 331" -> "331" (letter followed by space and number)
    # This handles tokenization artifacts from the model
    sanitized = re.sub(r'\b[a-zA-Z]\s+(\d+)', r'\1', coord_str)
    return sanitized


def test_malformed_coordinates():
    """Test the example from GitHub Issue #278"""
    print("=" * 60)
    print("Test 1: Malformed coordinates from Issue #278")
    print("=" * 60)
    
    # The problematic coordinate string from the issue
    malformed = "[[550, s 331, 652, 345]]"
    print(f"Original:  {malformed}")
    
    sanitized = sanitize_coordinates(malformed)
    print(f"Sanitized: {sanitized}")
    
    try:
        result = eval(sanitized)
        print(f"Parsed:    {result}")
        print("✓ SUCCESS: Coordinates parsed correctly!")
        assert result == [[550, 331, 652, 345]], "Parsed coordinates don't match expected values"
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_valid_coordinates():
    """Test that valid coordinates still work"""
    print("\n" + "=" * 60)
    print("Test 2: Valid coordinates (should remain unchanged)")
    print("=" * 60)
    
    test_cases = [
        "[[100, 200, 300, 400]]",
        "[[10, 20, 30, 40], [50, 60, 70, 80]]",
        "[[0, 0, 999, 999]]",
    ]
    
    all_passed = True
    for coord_str in test_cases:
        print(f"\nOriginal:  {coord_str}")
        sanitized = sanitize_coordinates(coord_str)
        print(f"Sanitized: {sanitized}")
        
        try:
            original_result = eval(coord_str)
            sanitized_result = eval(sanitized)
            
            if original_result == sanitized_result:
                print("✓ SUCCESS: Coordinates unchanged and valid")
            else:
                print(f"✗ FAILED: Coordinates changed unexpectedly")
                print(f"  Original result:  {original_result}")
                print(f"  Sanitized result: {sanitized_result}")
                all_passed = False
        except Exception as e:
            print(f"✗ FAILED: {e}")
            all_passed = False
    
    return all_passed


def test_multiple_malformed_patterns():
    """Test various malformed patterns"""
    print("\n" + "=" * 60)
    print("Test 3: Multiple malformed patterns")
    print("=" * 60)
    
    test_cases = [
        ("[[a 100, b 200, c 300, d 400]]", [[100, 200, 300, 400]]),
        ("[[x 50, y 60, 70, 80]]", [[50, 60, 70, 80]]),
        ("[[550, s 331, 652, t 345]]", [[550, 331, 652, 345]]),
    ]
    
    all_passed = True
    for coord_str, expected in test_cases:
        print(f"\nOriginal:  {coord_str}")
        sanitized = sanitize_coordinates(coord_str)
        print(f"Sanitized: {sanitized}")
        
        try:
            result = eval(sanitized)
            print(f"Parsed:    {result}")
            
            if result == expected:
                print("✓ SUCCESS: Coordinates parsed correctly!")
            else:
                print(f"✗ FAILED: Expected {expected}, got {result}")
                all_passed = False
        except Exception as e:
            print(f"✗ FAILED: {e}")
            all_passed = False
    
    return all_passed


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 60)
    print("Test 4: Edge cases")
    print("=" * 60)
    
    test_cases = [
        ("[[s 0, 0, 100, 100]]", [[0, 0, 100, 100]]),  # Zero coordinate
        ("[[999, 999, s 999, 999]]", [[999, 999, 999, 999]]),  # Max coordinate
        ("[[s 123]]", [[123]]),  # Single coordinate
    ]
    
    all_passed = True
    for coord_str, expected in test_cases:
        print(f"\nOriginal:  {coord_str}")
        sanitized = sanitize_coordinates(coord_str)
        print(f"Sanitized: {sanitized}")
        
        try:
            result = eval(sanitized)
            print(f"Parsed:    {result}")
            
            if result == expected:
                print("✓ SUCCESS: Edge case handled correctly!")
            else:
                print(f"✗ FAILED: Expected {expected}, got {result}")
                all_passed = False
        except Exception as e:
            print(f"✗ FAILED: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing Coordinate Sanitization Fix for Issue #278")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Malformed coordinates", test_malformed_coordinates()))
    results.append(("Valid coordinates", test_valid_coordinates()))
    results.append(("Multiple malformed patterns", test_multiple_malformed_patterns()))
    results.append(("Edge cases", test_edge_cases()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
