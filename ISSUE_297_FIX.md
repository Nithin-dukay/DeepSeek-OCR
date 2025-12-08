# Fix for GitHub Issue #297: Partial Content Recognition Problem

## Issue Description

When using DeepSeek-OCR to recognize certain images, only partial content is recognized. For example, in the reported case:
- **Recognized**: "xxx单位" (left side text)
- **Missing**: "2024年10月22日 印发" (right side date)

## Root Cause Analysis

The issue is caused by the **N-gram repetition filter** (`NoRepeatNGramLogitsProcessor`) being too aggressive in certain scenarios:

1. **Aggressive Filtering**: The default n-gram filter (ngram_size=30, window_size=90) may block legitimate tokens that appear similar to previously generated content
2. **Sparse Document Problem**: Documents with minimal text spread across the page are particularly susceptible to this issue
3. **Early Stopping**: The model may stop generating tokens prematurely when the filter blocks too many candidates

## Solution

We've implemented **configurable n-gram filtering parameters** that allow users to adjust or disable the filter based on their document type.

### Changes Made

#### 1. **config.py** - New Configuration Parameters

Added three new parameters:

```python
# N-gram repetition filter settings
NGRAM_SIZE = 30              # Size of n-gram to check for repetition
NGRAM_WINDOW_SIZE = 90       # Window size to search for repeated n-grams
DISABLE_NGRAM_FILTER = False # Set to True to disable filtering entirely
```

#### 2. **Updated All Inference Scripts**

Modified the following files to use configurable parameters:
- `run_dpsk_ocr_image.py`
- `run_dpsk_ocr_pdf.py`
- `run_dpsk_ocr_eval_batch.py`

Each script now:
- Imports the new configuration parameters
- Conditionally creates the logits processor based on settings
- Prints the active configuration for transparency

#### 3. **Test Script**

Created `test_issue_297.py` for easy testing and validation with command-line options.

## Usage Guide

### For Sparse Documents (Recommended for Issue #297)

**Option 1: Disable N-gram Filtering** (Most effective)

Edit `config.py`:
```python
DISABLE_NGRAM_FILTER = True
```

Or use the test script:
```bash
python test_issue_297.py --image test_issue_297.png --disable-ngram
```

**Option 2: Adjust N-gram Parameters**

For less aggressive filtering:
```python
NGRAM_SIZE = 20              # Smaller = less restrictive
NGRAM_WINDOW_SIZE = 150      # Larger = more context
```

Or use the test script:
```bash
python test_issue_297.py --image test_issue_297.png --ngram-size 20 --window-size 150
```

### For Dense Documents with Repetition Risk

Keep default settings or use stricter filtering:
```python
NGRAM_SIZE = 40              # Larger = more restrictive
NGRAM_WINDOW_SIZE = 90       # Standard window
DISABLE_NGRAM_FILTER = False
```

## Testing the Fix

### Using the Test Script

```bash
# Navigate to the vllm directory
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm

# Test with disabled n-gram filtering (recommended for sparse documents)
python test_issue_297.py --image ../../test_issue_297.png --disable-ngram

# Test with adjusted parameters
python test_issue_297.py --image ../../test_issue_297.png --ngram-size 20 --window-size 150

# Save output to file
python test_issue_297.py --image ../../test_issue_297.png --disable-ngram --output result.txt
```

### Using Existing Scripts

1. Edit `config.py` to set your desired parameters
2. Update `INPUT_PATH` and `OUTPUT_PATH` in `config.py`
3. Run the appropriate script:

```bash
# For single images
python run_dpsk_ocr_image.py

# For PDFs
python run_dpsk_ocr_pdf.py

# For batch evaluation
python run_dpsk_ocr_eval_batch.py
```

## When to Adjust N-gram Parameters

### Disable Filtering (`DISABLE_NGRAM_FILTER = True`)

Use when:
- ✓ Document has sparse text (like Issue #297)
- ✓ Content is unlikely to have repetitive patterns
- ✓ Missing content in OCR output
- ✓ Short documents with minimal text

### Increase Window Size (`NGRAM_WINDOW_SIZE = 150-200`)

Use when:
- ✓ Document has varied content across pages
- ✓ Need to allow more context for repetition detection
- ✓ Experiencing partial recognition issues

### Decrease N-gram Size (`NGRAM_SIZE = 20-25`)

Use when:
- ✓ Filter is too restrictive
- ✓ Legitimate content is being blocked
- ✓ Document has natural repetition (e.g., headers, footers)

### Keep Default or Increase N-gram Size (`NGRAM_SIZE = 40+`)

Use when:
- ✓ Document has high risk of repetitive output
- ✓ Tables with many similar cells
- ✓ Forms with repeated patterns
- ✓ Need stricter filtering

## Technical Details

### N-gram Repetition Filter

The `NoRepeatNGramLogitsProcessor` prevents the model from generating repetitive sequences by:

1. **Tracking n-grams**: Monitors sequences of N tokens
2. **Window-based search**: Looks for repetitions within a sliding window
3. **Token blocking**: Sets probability to -inf for tokens that would create repetitions
4. **Whitelist**: Allows certain tokens (like `<td>`, `</td>`) to repeat

### Default Settings by Script

- **run_dpsk_ocr_image.py**: Uses config values directly
- **run_dpsk_ocr_pdf.py**: Uses smaller values (20/50) for faster processing unless overridden
- **run_dpsk_ocr_eval_batch.py**: Uses larger ngram_size (40) for stricter filtering unless overridden

## Verification

To verify the fix works for Issue #297:

1. The date "2024年10月22日 印发" should now be recognized
2. Both left and right side content should appear in output
3. No premature stopping should occur

Expected output should include:
```
■xxx单位 ■                    2024年10月22日 印发
```

## Backward Compatibility

- **Default behavior unchanged**: If you don't modify `config.py`, the system behaves exactly as before
- **Opt-in changes**: All new features require explicit configuration
- **No breaking changes**: Existing scripts and workflows continue to work

## Performance Considerations

- **Disabling filter**: Slightly faster inference, but risk of repetitive output
- **Larger window**: Slightly slower due to more comparisons
- **Smaller n-gram**: Faster processing, less restrictive

## Troubleshooting

### Still Missing Content?

1. Try disabling n-gram filter completely
2. Check if prompt is appropriate for document type
3. Verify image quality and resolution
4. Try different prompts:
   - `"<image>\nFree OCR."` - For plain text
   - `"<image>\n<|grounding|>Convert the document to markdown."` - For structured documents
   - `"<image>\n<|grounding|>OCR this image."` - For general images

### Too Much Repetition?

1. Enable n-gram filter if disabled
2. Increase `NGRAM_SIZE` (e.g., 40-50)
3. Decrease `NGRAM_WINDOW_SIZE` (e.g., 50-70)

### Performance Issues?

1. Use smaller window sizes for faster processing
2. Consider using PDF-specific settings (smaller values)
3. Adjust `MAX_CONCURRENCY` in config.py

## Contributing

If you encounter similar issues or have suggestions for improvement:

1. Test with the provided test script
2. Document your findings (image type, settings used, results)
3. Open an issue on GitHub with details
4. Include sample images if possible (ensure no sensitive data)

## References

- **GitHub Issue**: #297
- **Related Files**:
  - `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`
  - `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`
  - `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py`
  - `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py`
  - `DeepSeek-OCR-master/DeepSeek-OCR-vllm/test_issue_297.py`
  - `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py`

## License

This fix is part of the DeepSeek-OCR project and follows the same license terms.
