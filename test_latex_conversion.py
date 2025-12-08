#!/usr/bin/env python3
"""
Test script for LaTeX formula conversion functionality
Tests GitHub Issue #219 fix
"""

import sys
sys.path.insert(0, '/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm')

from process.latex_utils import convert_latex_to_markdown, process_model_output


def test_inline_math_markdown():
    """Test inline math conversion to markdown format"""
    print("Test 1: Inline math to markdown")
    input_text = r"\(\mathrm{A} > 50\%\)"
    expected = r"$\mathrm{A} > 50\%$"
    result = convert_latex_to_markdown(input_text, 'markdown')
    print(f"  Input:    {repr(input_text)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Result:   {repr(result)}")
    print(f"  Status:   {'✓ PASS' if result == expected else '✗ FAIL'}")
    print()
    return result == expected


def test_display_math_markdown():
    """Test display math conversion to markdown format"""
    print("Test 2: Display math to markdown")
    input_text = r"\[E = mc^2\]"
    expected = r"$$E = mc^2$$"
    result = convert_latex_to_markdown(input_text, 'markdown')
    print(f"  Input:    {repr(input_text)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Result:   {repr(result)}")
    print(f"  Status:   {'✓ PASS' if result == expected else '✗ FAIL'}")
    print()
    return result == expected


def test_inline_math_tags():
    """Test inline math conversion to special tags format"""
    print("Test 3: Inline math to special tags")
    input_text = r"\(x^2 + y^2 = z^2\)"
    expected = r"<|math|>x^2 + y^2 = z^2<|/math|>"
    result = convert_latex_to_markdown(input_text, 'tags')
    print(f"  Input:    {repr(input_text)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Result:   {repr(result)}")
    print(f"  Status:   {'✓ PASS' if result == expected else '✗ FAIL'}")
    print()
    return result == expected


def test_display_math_tags():
    """Test display math conversion to special tags format"""
    print("Test 4: Display math to special tags")
    input_text = r"\[\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}\]"
    expected = r"<|math|>\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}<|/math|>"
    result = convert_latex_to_markdown(input_text, 'tags')
    print(f"  Input:    {repr(input_text)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Result:   {repr(result)}")
    print(f"  Status:   {'✓ PASS' if result == expected else '✗ FAIL'}")
    print()
    return result == expected


def test_mixed_content():
    """Test text with both inline and display math"""
    print("Test 5: Mixed content with inline and display math")
    input_text = r"The equation \(a^2 + b^2 = c^2\) is known as the Pythagorean theorem. The general form is: \[\sum_{i=1}^n a_i^2 = c^2\]"
    expected = r"The equation $a^2 + b^2 = c^2$ is known as the Pythagorean theorem. The general form is: $$\sum_{i=1}^n a_i^2 = c^2$$"
    result = convert_latex_to_markdown(input_text, 'markdown')
    print(f"  Input:    {repr(input_text)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Result:   {repr(result)}")
    print(f"  Status:   {'✓ PASS' if result == expected else '✗ FAIL'}")
    print()
    return result == expected


def test_multiple_inline_math():
    """Test multiple inline math expressions"""
    print("Test 6: Multiple inline math expressions")
    input_text = r"We have \(x = 5\) and \(y = 10\), so \(x + y = 15\)."
    expected = r"We have $x = 5$ and $y = 10$, so $x + y = 15$."
    result = convert_latex_to_markdown(input_text, 'markdown')
    print(f"  Input:    {repr(input_text)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Result:   {repr(result)}")
    print(f"  Status:   {'✓ PASS' if result == expected else '✗ FAIL'}")
    print()
    return result == expected


def test_complex_formula():
    """Test complex formula with special characters"""
    print("Test 7: Complex formula with special characters")
    input_text = r"\[\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u\]"
    expected = r"$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$$"
    result = convert_latex_to_markdown(input_text, 'markdown')
    print(f"  Input:    {repr(input_text)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Result:   {repr(result)}")
    print(f"  Status:   {'✓ PASS' if result == expected else '✗ FAIL'}")
    print()
    return result == expected


def test_no_escape_sequence_error():
    """Test that the conversion prevents Python escape sequence errors"""
    print("Test 8: No Python escape sequence errors")
    input_text = r"\(\mathrm{A} > 50\%\)"
    try:
        result = convert_latex_to_markdown(input_text, 'markdown')
        # Try to use the result in a string context
        test_string = f"Result: {result}"
        print(f"  Input:    {repr(input_text)}")
        print(f"  Result:   {repr(result)}")
        print(f"  Test string: {test_string}")
        print(f"  Status:   ✓ PASS (No escape sequence errors)")
        print()
        return True
    except Exception as e:
        print(f"  Input:    {repr(input_text)}")
        print(f"  Error:    {e}")
        print(f"  Status:   ✗ FAIL (Escape sequence error)")
        print()
        return False


def test_process_model_output():
    """Test the main process_model_output function"""
    print("Test 9: process_model_output function")
    input_text = r"The value is \(\mathrm{A} > 50\%\) and the formula is \[E = mc^2\]"
    expected = r"The value is $\mathrm{A} > 50\%$ and the formula is $$E = mc^2$$"
    result = process_model_output(input_text, latex_format='markdown', clean_formulas=False)
    print(f"  Input:    {repr(input_text)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Result:   {repr(result)}")
    print(f"  Status:   {'✓ PASS' if result == expected else '✗ FAIL'}")
    print()
    return result == expected


def test_markdown_rendering():
    """Test that the output is valid markdown"""
    print("Test 10: Markdown rendering validation")
    input_text = r"# Math Example\n\nInline: \(x^2\)\n\nDisplay:\n\[y = mx + b\]"
    result = convert_latex_to_markdown(input_text, 'markdown')
    print(f"  Input (raw):\n{input_text}")
    print(f"\n  Result (markdown):\n{result}")
    
    # Check if result contains proper markdown math delimiters
    has_inline = '$x^2$' in result
    has_display = '$$y = mx + b$$' in result
    success = has_inline and has_display
    
    print(f"\n  Has inline math ($...$): {has_inline}")
    print(f"  Has display math ($$...$$): {has_display}")
    print(f"  Status:   {'✓ PASS' if success else '✗ FAIL'}")
    print()
    return success


def main():
    """Run all tests"""
    print("=" * 70)
    print("LaTeX Formula Conversion Tests - GitHub Issue #219")
    print("=" * 70)
    print()
    
    tests = [
        test_inline_math_markdown,
        test_display_math_markdown,
        test_inline_math_tags,
        test_display_math_tags,
        test_mixed_content,
        test_multiple_inline_math,
        test_complex_formula,
        test_no_escape_sequence_error,
        test_process_model_output,
        test_markdown_rendering,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Status:   ✗ FAIL")
            print()
            results.append(False)
    
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
