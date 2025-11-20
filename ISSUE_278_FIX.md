# Fix for GitHub Issue #278: Malformed Coordinates with 's' Character

## Problem Description

When parsing OCR output, the model sometimes generates malformed coordinate strings like:
```
<|ref|>text<|/ref|><|det|>[[550, s 331, 652, 345]]<|/det|>
```

The `s 331` pattern (letter followed by space and number) is invalid Python syntax and causes the `eval()` function to fail when parsing coordinates.

## Root Cause

The `s` character (and potentially other letters) appears to be a tokenization or generation artifact from the model. This creates invalid coordinate strings that cannot be parsed by Python's `eval()` function.

## Solution

Added a `sanitize_coordinates()` function that cleans coordinate strings before parsing by:
1. Detecting patterns like `s 331` (letter + space + number)
2. Removing the letter and space, keeping only the number
3. Using regex pattern: `r'\b[a-zA-Z]\s+(\d+)'` → `r'\1'`

## Changes Made

### Files Modified

1. **DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py**
2. **DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py**

### Implementation Details

#### Added `sanitize_coordinates()` function:
```python
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
```

#### Updated `extract_coordinates_and_label()` function:
```python
def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        # Sanitize the coordinate string before eval
        coord_str = sanitize_coordinates(ref_text[2])
        cor_list = eval(coord_str)
    except Exception as e:
        print(f"Warning: Failed to parse coordinates. Original: {ref_text[2]}, Error: {e}")
        return None

    return (label_type, cor_list)
```

## Testing

Comprehensive tests were created and all passed successfully:

### Test Cases

1. **Malformed coordinates from Issue #278**
   - Input: `[[550, s 331, 652, 345]]`
   - Output: `[[550, 331, 652, 345]]`
   - ✓ PASSED

2. **Valid coordinates (unchanged)**
   - Input: `[[100, 200, 300, 400]]`
   - Output: `[[100, 200, 300, 400]]`
   - ✓ PASSED

3. **Multiple malformed patterns**
   - Input: `[[a 100, b 200, c 300, d 400]]`
   - Output: `[[100, 200, 300, 400]]`
   - ✓ PASSED

4. **Edge cases**
   - Zero coordinates: `[[s 0, 0, 100, 100]]` → `[[0, 0, 100, 100]]`
   - Max coordinates: `[[999, 999, s 999, 999]]` → `[[999, 999, 999, 999]]`
   - Single coordinate: `[[s 123]]` → `[[123]]`
   - ✓ ALL PASSED

## Benefits

1. **Robust parsing**: Handles malformed model outputs gracefully
2. **Backward compatible**: Valid coordinates remain unchanged
3. **Better error reporting**: Improved error messages for debugging
4. **Comprehensive**: Handles various malformed patterns (a, b, c, s, t, x, y, etc.)

## Usage

No changes required to existing code. The fix is transparent to users:

```python
# Before: Would fail with "s 331" pattern
# After: Automatically sanitizes and parses correctly

result = extract_coordinates_and_label(ref_text, image_width, image_height)
# Now handles both valid and malformed coordinates
```

## Future Considerations

If other malformed patterns are discovered, the `sanitize_coordinates()` function can be easily extended with additional regex patterns or cleaning logic.

## Related Issue

- GitHub Issue #278: "What is the meaning of the `s` in the coordinates?"
- Answer: The `s` is a tokenization artifact that should not be present. This fix removes it automatically.
