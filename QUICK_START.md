# Quick Start Guide: Fix for Issue #295

## Problem
DeepSeek-OCR fails to detect the last two columns of wide tables (e.g., Table 3 on page 11 of the paper).

## Solution
We've implemented a fix with optimized settings for wide tables.

## Quick Fix (3 Steps)

### Step 1: Enable Wide Table Configuration

Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

```python
# Change these two lines:
USE_WIDE_TABLE_CONFIG = True  # Changed from False
DOCUMENT_TYPE = 'wide_table'   # Changed from 'default'
```

### Step 2: Set Your Input/Output Paths

In the same `config.py` file:

```python
INPUT_PATH = 'path/to/your/image.png'  # Your test image
OUTPUT_PATH = './output'                # Where to save results
```

### Step 3: Run the OCR

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

## What Changed?

The fix automatically:
- ✅ Reduces n-gram size from 30 to 20 (more lenient)
- ✅ Increases token limit from 8,192 to 12,288 (50% more)
- ✅ Expands whitelist to include all table tokens
- ✅ Increases MAX_CROPS from 6 to 9 (better resolution)
- ✅ Monitors token usage and warns if approaching limits

## Verify the Fix

Run the test script:

```bash
python test_issue_295.py
```

You should see: ✅ All fix components are in place!

## Configuration Options

| Document Type | When to Use | ngram_size | max_tokens |
|---------------|-------------|------------|------------|
| `wide_table` | Wide tables (10+ columns) | 20 | 12,288 |
| `table_heavy` | Many tables in document | 25 | 10,240 |
| `pdf` | Multi-page PDFs | 15 | 12,288 |
| `default` | Original settings | 30 | 8,192 |

## Example: Test with Page 11 of the Paper

```bash
# Extract page 11
python3 -c "import fitz; doc = fitz.open('DeepSeek_OCR_paper.pdf'); page = doc[10]; pix = page.get_pixmap(dpi=150); pix.save('page11.png')"

# Edit config.py
# USE_WIDE_TABLE_CONFIG = True
# DOCUMENT_TYPE = 'wide_table'
# INPUT_PATH = 'page11.png'
# OUTPUT_PATH = './output'

# Run OCR
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py

# Check results
cat ./output/result.mmd
```

## Expected Output

You should now see **all columns** including the last two (Chinese `table` and `order` columns) that were previously missing.

## Troubleshooting

### Still missing columns?
Try increasing the token limit in `table_token_config.py`:

```python
WIDE_TABLE = {
    'ngram_size': 20,
    'window_size': 70,
    'max_tokens': 16384,  # Increase from 12288
    'max_model_len': 16384,
    'whitelist_token_ids': TABLE_TOKEN_WHITELIST,
}
```

### Token limit warnings?
The system will warn you:
```
⚠️  WARNING: Output approaching token limit!
   Current: 11500 / 12288 tokens (93.6%)
```

Solution: Increase `max_tokens` as shown above.

### Want to revert to original settings?
Simply set in `config.py`:
```python
USE_WIDE_TABLE_CONFIG = False
```

## More Information

- **Full Documentation**: See `ISSUE_295_FIX.md`
- **Solution Summary**: See `SOLUTION_SUMMARY.md`
- **Test Script**: Run `python test_issue_295.py`

## Support

If you encounter issues:
1. Check that all files are present: `python test_issue_295.py`
2. Verify configuration: Check `config.py` settings
3. Monitor token usage: Look for warnings during processing
4. Review documentation: See `ISSUE_295_FIX.md` for details
