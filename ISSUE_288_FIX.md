# Fix for GitHub Issue #288: Model Not Working When Prompt Modified

## Issue Summary

**Problem:** When users modified the standard prompt from:
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```

to:
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
```

The model would produce repetitive output (numbers repeating endlessly) instead of performing OCR correctly.

## Root Cause

The DeepSeek-OCR model is highly sensitive to prompt format. The model was trained on specific prompt templates, and deviations from these templates can disrupt the generation pattern, causing:

1. **Repetitive token generation** - The model gets stuck in loops
2. **Bypassing n-gram protection** - Even with `NoRepeatNGramLogitsProcessor` active
3. **Incorrect output** - Numbers or text that don't exist in the image

Specific problematic patterns:
- **Negative instructions**: "don't add", "without", "never"
- **Meta-instructions**: "make sure", "please", "always"
- **Complex constraints**: Multiple sentences with detailed requirements

## Solution Implemented

### 1. Prompt Validator (`process/prompt_validator.py`)

A comprehensive validation system that:
- ✅ Detects problematic prompt patterns
- ✅ Provides clear warnings about potential issues
- ✅ Suggests safe alternative prompts
- ✅ Explains why certain patterns are problematic

**Example output:**
```
======================================================================
PROMPT VALIDATION REPORT
======================================================================

Status: ⚠️  WARNING: This prompt may cause repetitive or incorrect output!
Warning Level: HIGH

Issues Detected (1):
  1. Negative instructions (e.g., "dont add") can confuse the model

💡 Suggested Alternative:
   <image>\n<|grounding|>Convert the document to markdown.

⚠️  RECOMMENDATION: Use one of the standard prompt formats
   to avoid repetitive or incorrect outputs.
======================================================================
```

### 2. Updated Configuration (`config.py`)

Added:
- Clear documentation of recommended prompts
- Warnings about unsafe modifications
- Configuration options for validation behavior

```python
# Enable/disable validation
ENABLE_PROMPT_VALIDATION = True

# Strict mode - reject invalid prompts
STRICT_PROMPT_MODE = False
```

### 3. Integration with Runner Scripts

Both `run_dpsk_ocr_image.py` and `run_dpsk_ocr_pdf.py` now:
- Validate prompts before processing
- Display warnings for problematic prompts
- Suggest alternatives when issues are detected
- Can optionally reject invalid prompts in strict mode

### 4. Comprehensive Documentation (`PROMPT_GUIDELINES.md`)

A complete guide covering:
- ✅ Recommended prompt formats
- ✅ Safe vs. unsafe modifications
- ✅ Troubleshooting repetitive output
- ✅ Best practices for prompt customization
- ✅ Technical explanation of why prompts matter

## Usage

### For Users Experiencing Issue #288

**Quick Fix:**
```python
# In config.py, change from:
PROMPT = '<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.'

# To:
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
```

The model handles spacing correctly with the standard prompt.

### Validation in Action

When you run the scripts with a problematic prompt:

```bash
python run_dpsk_ocr_image.py
```

You'll see:
```
======================================================================
PROMPT VALIDATION REPORT
======================================================================

Status: ⚠️  WARNING: This prompt may cause repetitive or incorrect output!
Warning Level: HIGH

Issues Detected (1):
  1. Negative instructions (e.g., "dont add") can confuse the model

💡 Suggested Alternative:
   <image>\n<|grounding|>Convert the document to markdown.
======================================================================

⚠️  Proceeding with potentially problematic prompt...
   If you experience repetitive output, use the suggested prompt instead.
```

### Configuration Options

**Option 1: Warnings Only (Default)**
```python
ENABLE_PROMPT_VALIDATION = True
STRICT_PROMPT_MODE = False
```
- Shows warnings but allows execution
- Good for experimentation

**Option 2: Strict Mode**
```python
ENABLE_PROMPT_VALIDATION = True
STRICT_PROMPT_MODE = True
```
- Rejects invalid prompts
- Forces use of recommended formats
- Prevents accidental issues

**Option 3: Disabled**
```python
ENABLE_PROMPT_VALIDATION = False
```
- No validation (not recommended)
- For advanced users only

## Testing

Run the test script to verify the fix:

```bash
python3 test_prompt_validation.py
```

This demonstrates:
- ✅ Original prompt validates correctly
- ✅ Problematic prompt is detected
- ✅ Clear warnings are displayed
- ✅ Suggestions are provided

## Files Modified/Created

### Created:
1. `DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/prompt_validator.py` - Validation logic
2. `PROMPT_GUIDELINES.md` - Comprehensive documentation
3. `test_prompt_validation.py` - Test script
4. `ISSUE_288_FIX.md` - This document

### Modified:
1. `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` - Added validation settings and documentation
2. `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py` - Integrated validator
3. `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py` - Integrated validator

## Recommended Prompts

These prompts are tested and work reliably:

| Use Case | Prompt |
|----------|--------|
| Document to Markdown | `<image>\n<|grounding|>Convert the document to markdown.` |
| General OCR | `<image>\n<|grounding|>OCR this image.` |
| Free OCR (no layout) | `<image>\nFree OCR.` |
| Figure/Chart parsing | `<image>\nParse the figure.` |
| Image description | `<image>\nDescribe this image in detail.` |
| Text location | `<image>\nLocate <|ref|>TEXT<|/ref|> in the image.` |

## Safe Modifications

If you need to customize prompts:

✅ **SAFE** - Append format specifications:
```python
"<image>\n<|grounding|>Convert the document to markdown in JSON format."
"<image>\n<|grounding|>Convert the document to markdown, focus on tables."
```

❌ **UNSAFE** - Negative instructions:
```python
"<image>\n<|grounding|>Convert the document to markdown, dont add spaces."
"<image>\n<|grounding|>OCR without line breaks."
```

## Benefits

1. **Prevents Issue #288** - Users are warned before encountering repetitive output
2. **Educational** - Users learn why certain prompts don't work
3. **Configurable** - Can be strict or permissive based on needs
4. **Non-breaking** - Existing valid prompts continue to work
5. **Helpful** - Provides actionable suggestions

## Future Improvements

Potential enhancements:
- Add more sophisticated pattern detection
- Machine learning-based prompt quality scoring
- Automatic prompt correction (not just suggestion)
- Integration with model fine-tuning for better prompt robustness

## Conclusion

This fix addresses GitHub Issue #288 by:
1. ✅ Detecting problematic prompt modifications
2. ✅ Warning users before issues occur
3. ✅ Providing clear guidance and alternatives
4. ✅ Documenting best practices comprehensively
5. ✅ Maintaining backward compatibility

Users who follow the recommended prompts or heed the warnings will no longer experience the repetitive output issue described in Issue #288.
