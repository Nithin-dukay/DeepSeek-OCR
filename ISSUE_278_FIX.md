# Fix for GitHub Issue #278: Spurious 's' Character in Coordinates

## Issue Description

When parsing OCR output, the model occasionally generates coordinates with spurious alphabetic characters, such as:

```
<|ref|>text<|/ref|><|det|>[[550, s 331, 652, 345]]<|/det|>
```

The `s` character (or other letters) appearing between coordinate numbers causes the `eval()` function to fail with a syntax error, preventing proper coordinate parsing.

## Root Cause

The `extract_coordinates_and_label()` function in both `run_dpsk_ocr_image.py` and `run_dpsk_ocr_pdf.py` was using `eval()` directly on the coordinate string without any sanitization. When the model outputs invalid characters, the parsing fails.

## Solution

Added a `sanitize_coordinates()` function that:

1. **Removes spurious alphabetic characters** that appear between numbers, brackets, and commas
2. **Cleans up whitespace** to ensure proper formatting
3. **Preserves valid coordinate structure** like `[[x1, y1, x2, y2], ...]`

### Implementation Details

The sanitization function uses regex patterns to:
- Remove letters that appear between valid coordinate characters (digits, brackets, commas)
- Normalize whitespace
- Clean up formatting around brackets and commas

### Code Changes

**Files Modified:**
1. `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`
2. `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py`

**Changes Made:**
- Added `sanitize_coordinates()` function before `extract_coordinates_and_label()`
- Modified `extract_coordinates_and_label()` to sanitize coordinate strings before calling `eval()`

### Example Transformations

| Input | Output | Status |
|-------|--------|--------|
| `[[550, s 331, 652, 345]]` | `[[550,331,652,345]]` | ✓ Fixed |
| `[[100, a 200, 300, b 400]]` | `[[100,200,300,400]]` | ✓ Fixed |
| `[[550, 331, 652, 345]]` | `[[550,331,652,345]]` | ✓ Works |
| `[[10,20,30,40],[50,60,70,80]]` | `[[10,20,30,40],[50,60,70,80]]` | ✓ Works |

## Testing

A comprehensive test suite (`test_coordinate_fix.py`) was created to verify:
- The reported issue case: `[[550, s 331, 652, 345]]`
- Normal coordinates without spurious characters
- Multiple spurious characters in one coordinate set
- Multiple coordinate pairs
- Multi-letter spurious strings
- Extra whitespace handling
- Spurious characters without spaces

**All tests pass successfully.**

## Impact

This fix ensures that:
- Coordinate parsing is more robust against model output variations
- The system gracefully handles spurious characters in coordinate strings
- Normal coordinate parsing continues to work as expected
- Error handling remains in place for truly malformed coordinates

## Backward Compatibility

The fix is fully backward compatible:
- Normal coordinate strings continue to work without any changes
- The sanitization only removes invalid characters
- The existing try-except error handling is preserved
- No changes to the API or function signatures

## Usage

No changes are required for users. The fix is transparent and automatically handles both clean and malformed coordinate strings.

## Related Files

- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py` - Image processing script
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py` - PDF processing script
- `test_coordinate_fix.py` - Test suite for the fix
