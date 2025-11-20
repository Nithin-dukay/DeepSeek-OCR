# Fix for GitHub Issue #278: Malformed Coordinates with 's' Character

## Problem Description

When parsing OCR output, the model sometimes generates malformed coordinate strings like:
```
<|ref|>text<|/ref|><|det|>[[550, s 331, 652, 345]]<|/det|>
```

The issue is the `s 331` pattern where `s` appears before the number `331`. This causes the coordinate parsing to fail because `eval()` cannot parse invalid Python syntax.

## Root Cause

The `extract_coordinates_and_label()` function in both `run_dpsk_ocr_image.py` and `run_dpsk_ocr_pdf.py` was using `eval()` directly on the coordinate string without any sanitization or validation. When the model generates malformed output like `s 331`, the `eval()` call fails with a syntax error.

## Solution

Added a `sanitize_coordinate_string()` function that cleans malformed coordinate strings before parsing. The function:

1. **Removes 's <number>' patterns**: Replaces `s 331` with `331`
2. **Handles other single-letter patterns**: Removes any stray single letters before numbers (e.g., `a 200`, `b 345`)
3. **Cleans up extra spaces**: Normalizes spacing within brackets and around commas
4. **Preserves valid coordinates**: Does not modify correctly formatted coordinate strings

### Implementation Details

```python
def sanitize_coordinate_string(coord_str):
    """
    Sanitize malformed coordinate strings before parsing.
    Handles cases like '[[550, s 331, 652, 345]]' where 's 331' should be '331'.
    """
    # Remove patterns like 's <number>' and replace with just '<number>'
    sanitized = re.sub(r'\bs\s+(\d+)', r'\1', coord_str)
    
    # Remove any other stray single letters followed by spaces before numbers
    sanitized = re.sub(r'\b[a-zA-Z]\s+(\d+)', r'\1', sanitized)
    
    # Clean up any extra spaces within brackets
    sanitized = re.sub(r'\[\s+', '[', sanitized)
    sanitized = re.sub(r'\s+\]', ']', sanitized)
    sanitized = re.sub(r',\s+', ', ', sanitized)
    
    return sanitized
```

### Enhanced Error Handling

The updated `extract_coordinates_and_label()` function now provides better error messages:

```python
def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        coord_str = ref_text[2]
        
        # Sanitize the coordinate string before parsing
        sanitized_coord_str = sanitize_coordinate_string(coord_str)
        
        # Parse the sanitized coordinate string
        cor_list = eval(sanitized_coord_str)
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        print(f"Original coordinate string: {ref_text[2] if len(ref_text) > 2 else 'N/A'}")
        if 'sanitized_coord_str' in locals():
            print(f"Sanitized coordinate string: {sanitized_coord_str}")
        return None

    return (label_type, cor_list)
```

## Files Modified

1. **DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py**
   - Added `sanitize_coordinate_string()` function
   - Updated `extract_coordinates_and_label()` to use sanitization
   - Enhanced error logging

2. **DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py**
   - Added `sanitize_coordinate_string()` function
   - Updated `extract_coordinates_and_label()` to use sanitization
   - Enhanced error logging

## Testing

Created comprehensive test suite (`test_coordinate_fix.py`) that validates:

### Test Cases

1. **GitHub Issue #278 pattern**: `[[550, s 331, 652, 345]]` → `[[550, 331, 652, 345]]` ✓
2. **Valid coordinates**: `[[100, 200, 300, 400]]` → `[[100, 200, 300, 400]]` ✓
3. **Multiple coordinates with 's' pattern**: Multiple malformed coordinates ✓
4. **Other single letter patterns**: `a 331`, `b 345` patterns ✓
5. **Extra spaces**: Handles excessive whitespace ✓
6. **Mixed valid and malformed**: Combination of valid and malformed coordinates ✓

All tests passed successfully! ✓

## Usage

The fix is transparent to users. When the model generates malformed coordinates, they will now be automatically sanitized and parsed correctly. If parsing still fails after sanitization, detailed error messages will help with debugging.

### Example

**Before Fix:**
```
Error: invalid syntax
```

**After Fix:**
```
# Coordinates are automatically sanitized and parsed
# If parsing fails, you get detailed error messages:
Error parsing coordinates: <error details>
Original coordinate string: [[550, s 331, 652, 345]]
Sanitized coordinate string: [[550, 331, 652, 345]]
```

## Backward Compatibility

The fix maintains full backward compatibility:
- Valid coordinate strings are not modified
- The function signature remains unchanged
- Existing functionality is preserved

## Future Improvements

Potential enhancements for consideration:
1. Add more robust validation of coordinate values (e.g., range checks)
2. Consider using a proper parser instead of `eval()` for better security
3. Add logging/metrics to track frequency of malformed coordinates
4. Investigate root cause in model to reduce malformed output generation

## Related Issues

- GitHub Issue #278: "What is the meaning of the `s` in the coordinates?"

## Conclusion

This fix resolves the coordinate parsing issue by sanitizing malformed strings before evaluation. The solution is robust, well-tested, and maintains backward compatibility while providing better error handling and debugging information.
