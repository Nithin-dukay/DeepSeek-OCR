# Quick Fix Guide for Issue #248

## TL;DR - What Changed?

The token limit has been increased from **8192** to **16384** to prevent truncation errors when processing complex documents.

## If You're Still Getting the Error

Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` and increase these values:

```python
MAX_MODEL_LEN = 32768  # Increase this
MAX_TOKENS = 32768     # Increase this
```

⚠️ **Note**: Higher values need more GPU memory. If you get OOM errors, reduce these values or lower `MAX_CONCURRENCY`.

## What the Fix Does

### Before (Hardcoded):
```python
max_model_len=8192  # Fixed value
max_tokens=8192     # Fixed value
```

### After (Configurable):
```python
max_model_len=MAX_MODEL_LEN  # From config.py (default: 16384)
max_tokens=MAX_TOKENS        # From config.py (default: 16384)
```

## New Features

1. **Configurable Limits**: Adjust token limits in one place (`config.py`)
2. **Warning Messages**: Get notified when outputs are truncated
3. **Higher Defaults**: 16384 tokens (2x the original limit)
4. **Better Error Handling**: Clear guidance when limits are reached

## Files Modified

- ✅ `config.py` - Added MAX_MODEL_LEN and MAX_TOKENS
- ✅ `run_dpsk_ocr_pdf.py` - Uses config values + error handling
- ✅ `run_dpsk_ocr_image.py` - Uses config values + error handling  
- ✅ `run_dpsk_ocr_eval_batch.py` - Uses config values + error handling

## Example Warning Message

When a document exceeds the token limit, you'll now see:

```
⚠️  Warning: Page 5 output was truncated due to token limit.
⚠️  Consider increasing MAX_TOKENS in config.py (current: 16384)
⚠️  Token usage - Prompt: 494, Generated: 15890
```

## Recommended Settings

| Your GPU Memory | MAX_TOKENS | MAX_MODEL_LEN |
|-----------------|------------|---------------|
| 16GB            | 8192       | 8192          |
| 24GB            | 16384      | 16384         |
| 40GB            | 32768      | 32768         |
| 80GB+           | 65536      | 65536         |

## Testing Your Changes

1. Process a PDF that previously failed
2. Check for warning messages
3. Verify output files are complete
4. Monitor GPU memory usage

## Need Help?

- Check `FIX_ISSUE_248.md` for detailed documentation
- Adjust `gpu_memory_utilization` in the scripts if needed
- Lower `MAX_CONCURRENCY` if you have memory issues
