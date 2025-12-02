# DeepSeek-OCR Prompt Guidelines

## Overview

The DeepSeek-OCR model is highly sensitive to prompt format. Using non-standard prompts can lead to:
- **Repetitive output** (same text/numbers repeated endlessly)
- **Incorrect OCR results**
- **Model confusion and poor performance**

This guide explains how to use prompts correctly and avoid common pitfalls.

---

## ⚠️ Issue #288: Prompt Modification Problem

**Problem:** Users reported that modifying the standard prompt causes repetitive output.

**Example of problematic modification:**
```python
# ❌ PROBLEMATIC - Causes repetitive output
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
```

**Why it fails:**
- Negative instructions ("don't add") confuse the model
- Additional constraints after the main instruction disrupt generation patterns
- The model is trained on specific prompt formats and deviates poorly from them

**Solution:** Use recommended prompts without modifications, or make only minimal, positive additions.

---

## ✅ Recommended Prompts

These prompts are tested and work reliably:

### 1. Document to Markdown (Most Common)
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```
**Use for:** Documents, PDFs, scanned pages with layout preservation

### 2. General OCR
```python
prompt = "<image>\n<|grounding|>OCR this image."
```
**Use for:** General images with text, photos, screenshots

### 3. Free OCR (No Layout)
```python
prompt = "<image>\nFree OCR."
```
**Use for:** Simple text extraction without layout/structure preservation

### 4. Figure/Chart Parsing
```python
prompt = "<image>\nParse the figure."
```
**Use for:** Charts, graphs, diagrams, scientific figures

### 5. General Image Description
```python
prompt = "<image>\nDescribe this image in detail."
```
**Use for:** Understanding image content beyond just text

### 6. Text Location (Grounding)
```python
prompt = "<image>\nLocate <|ref|>specific text<|/ref|> in the image."
```
**Use for:** Finding specific text and getting bounding box coordinates

---

## 🔧 Safe Prompt Modifications

If you need to customize prompts, follow these guidelines:

### ✅ SAFE Modifications (Append to base prompt)

**Output Format Specifications:**
```python
# Good - Specifies output format positively
prompt = "<image>\n<|grounding|>Convert the document to markdown in JSON format."
prompt = "<image>\n<|grounding|>Convert the document to markdown as plain text."
```

**Focus Area Specifications:**
```python
# Good - Focuses on specific content
prompt = "<image>\n<|grounding|>Convert the document to markdown, focus on tables."
prompt = "<image>\n<|grounding|>OCR this image, extract text only."
```

**Language Specifications:**
```python
# Good - Specifies language handling
prompt = "<image>\n<|grounding|>Convert the document to markdown in English."
prompt = "<image>\n<|grounding|>OCR this image, preserve original language."
```

### ❌ UNSAFE Modifications (Avoid These)

**Negative Instructions:**
```python
# Bad - Negative constraints confuse the model
prompt = "<image>\n<|grounding|>Convert the document to markdown, don't add extra spaces."
prompt = "<image>\n<|grounding|>OCR without line breaks."
prompt = "<image>\n<|grounding|>Convert but never include images."
```

**Meta-Instructions:**
```python
# Bad - Instructions about how to perform the task
prompt = "<image>\n<|grounding|>Convert the document to markdown, make sure it's accurate."
prompt = "<image>\n<|grounding|>OCR this image, always check for errors."
prompt = "<image>\n<|grounding|>Please be careful when converting."
```

**Multiple Complex Sentences:**
```python
# Bad - Too many instructions cause confusion
prompt = "<image>\n<|grounding|>Convert the document to markdown. Make sure to preserve formatting. Don't add extra spaces. Keep tables intact."
```

**Absolute Constraints:**
```python
# Bad - Absolute terms may cause repetitive behavior
prompt = "<image>\n<|grounding|>Always convert to markdown."
prompt = "<image>\n<|grounding|>Never skip any text."
```

---

## 🛠️ Troubleshooting

### Problem: Repetitive Output (Numbers/Text Repeating)

**Symptoms:**
- Same numbers or text repeated endlessly
- Output doesn't match image content
- Generation doesn't stop properly

**Solutions:**

1. **Use a recommended prompt format** (most important)
   ```python
   # Replace your custom prompt with:
   prompt = "<image>\n<|grounding|>Convert the document to markdown."
   ```

2. **Check for problematic patterns:**
   - Remove negative instructions ("don't", "without", "never")
   - Remove meta-instructions ("make sure", "always", "please")
   - Simplify to a single clear instruction

3. **Verify n-gram processor is active:**
   ```python
   from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
   
   logits_processors = [
       NoRepeatNGramLogitsProcessor(
           ngram_size=30, 
           window_size=90, 
           whitelist_token_ids={128821, 128822}
       )
   ]
   ```

4. **Enable prompt validation:**
   ```python
   # In config.py
   ENABLE_PROMPT_VALIDATION = True
   STRICT_PROMPT_MODE = False  # Set to True to reject invalid prompts
   ```

### Problem: Poor OCR Quality

**Solutions:**
- Try different resolution modes (Tiny/Small/Base/Large/Gundam)
- Use the appropriate prompt for your content type
- Ensure image quality is sufficient (not too blurry/low-res)

### Problem: Missing Layout/Structure

**Solutions:**
- Use `<|grounding|>` prompts for layout preservation
- Use "Convert the document to markdown" instead of "Free OCR"
- Check that CROP_MODE is set appropriately

---

## 📋 Validation System

The codebase now includes automatic prompt validation:

### How It Works

When you run the OCR scripts, your prompt is automatically validated:

```
======================================================================
PROMPT VALIDATION REPORT
======================================================================

Status: ⚠️  WARNING: This prompt may cause repetitive or incorrect output!
Warning Level: HIGH

Issues Detected (2):
  1. Negative instructions (e.g., "dont add") can confuse the model
  2. Spacing instructions are not reliably followed

💡 Suggested Alternative:
   <image>\n<|grounding|>Convert the document to markdown.

⚠️  RECOMMENDATION: Use one of the standard prompt formats
   to avoid repetitive or incorrect outputs.

   See PROMPT_GUIDELINES.md for more information.
======================================================================
```

### Configuration Options

In `config.py`:

```python
# Enable/disable validation warnings
ENABLE_PROMPT_VALIDATION = True

# Strict mode - reject invalid prompts entirely
STRICT_PROMPT_MODE = False  # Set to True to enforce valid prompts only
```

### Manual Validation

You can validate prompts programmatically:

```python
from process.prompt_validator import validate_prompt

prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add spaces."
result = validate_prompt(prompt, verbose=True)

if not result.is_valid:
    print(f"Use this instead: {result.suggested_prompt}")
```

---

## 🎯 Best Practices

1. **Start with a recommended prompt** - Don't modify unless necessary
2. **Make minimal changes** - Add only what's essential
3. **Use positive instructions** - Say what you want, not what you don't want
4. **Test incrementally** - If modifying, test each change
5. **Enable validation** - Let the system warn you about problematic prompts
6. **Keep it simple** - One clear instruction works best

---

## 📚 Examples

### Example 1: Document OCR (Correct)
```python
# ✅ Recommended approach
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# Result: Clean markdown with proper layout preservation
```

### Example 2: Document OCR (Incorrect)
```python
# ❌ Problematic approach
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

# Result: Repetitive output, numbers repeating endlessly
```

### Example 3: Custom Format (Correct)
```python
# ✅ Safe customization
prompt = "<image>\n<|grounding|>Convert the document to markdown in JSON format."

# Result: Markdown content wrapped in JSON structure
```

### Example 4: Table Extraction (Correct)
```python
# ✅ Focused extraction
prompt = "<image>\n<|grounding|>Convert the document to markdown, focus on tables."

# Result: Emphasis on table extraction with proper formatting
```

---

## 🔍 Technical Details

### Why Prompts Matter

The DeepSeek-OCR model uses:
1. **Vision encoder** - Processes the image
2. **Language model** - Generates text based on prompt
3. **Special tokens** - `<|grounding|>` triggers layout-aware processing

The prompt directly affects:
- Token generation patterns
- Attention mechanisms
- Output structure and format

### N-gram Repetition Prevention

The model uses `NoRepeatNGramLogitsProcessor` to prevent repetition:
- Tracks recent n-grams (sequences of tokens)
- Bans tokens that would create repeated patterns
- Whitelist allows specific tokens (like table tags)

**However**, this only works if the generation pattern is reasonable. Bad prompts can create patterns that bypass this protection.

---

## 📞 Support

If you encounter issues:

1. **Check this guide** - Most problems are covered here
2. **Validate your prompt** - Use the built-in validator
3. **Try a recommended prompt** - Start with a known-good format
4. **Report persistent issues** - Open a GitHub issue with:
   - Your exact prompt
   - Sample image (if possible)
   - Output showing the problem
   - Configuration settings

---

## 📝 Summary

**Key Takeaways:**
- ✅ Use recommended prompt formats
- ❌ Avoid negative instructions and meta-instructions
- 🔧 Make only minimal, positive modifications
- 🛡️ Enable prompt validation to catch issues early
- 📖 Refer to this guide when customizing prompts

**Quick Fix for Issue #288:**
```python
# Replace this:
prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

# With this:
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```

The model will handle spacing correctly with the standard prompt.
