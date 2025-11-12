# Table Recognition Quick Reference

## 🚀 Quick Start

### Prompt
```python
prompt = "<image>\n<|grounding|>Convert the table to HTML."
```

### Essential Configuration
```python
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from vllm import SamplingParams

logits_processors = [NoRepeatNGramLogitsProcessor(
    ngram_size=30, 
    window_size=90, 
    whitelist_token_ids={128821, 128822}  # <td>, </td>
)]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)
```

## 📝 Prompt Variations

| Task | Prompt |
|------|--------|
| Table to HTML | `<image>\n<|grounding|>Convert the table to HTML.` |
| Table to Markdown | `<image>\n<|grounding|>Convert the table to markdown.` |
| Multiple tables | `<image>\n<|grounding|>Extract and convert all tables to HTML.` |
| Document with tables | `<image>\n<|grounding|>Convert the document to markdown.` |

## 🔑 Token IDs

| Token | ID | Purpose |
|-------|-----|---------|
| `<td>` | 128821 | Table cell (must whitelist) |
| `</td>` | 128822 | Table cell close (must whitelist) |

## ⚙️ Recommended Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ngram_size` | 30 | N-gram repetition check size |
| `window_size` | 90 | Sliding window size |
| `temperature` | 0.0 | Deterministic output |
| `max_tokens` | 8192 | Maximum output length |
| `skip_special_tokens` | False | Keep HTML tags |
| `crop_mode` | True | Better for large images |
| `base_size` | 1024 | Standard resolution |
| `image_size` | 640 | Crop tile size |

## 💡 Key Points

✅ **DO**:
- Use `<|grounding|>` tag for structured output
- Whitelist tokens 128821 and 128822
- Enable `crop_mode` for large images
- Set `skip_special_tokens=False`

❌ **DON'T**:
- Forget to whitelist `<td>` and `</td>` tokens
- Use high temperature (causes inconsistent output)
- Skip the `<|grounding|>` tag for tables

## 🛠️ Command Line

```bash
# Basic usage
python table_to_html_example.py --image table.jpg --output result.html

# With transformers backend
python table_to_html_example.py --image table.jpg --backend transformers
```

## 📚 Full Documentation

- Detailed guide: [TABLE_RECOGNITION.md](TABLE_RECOGNITION.md)
- Main README: [README.md](README.md)
- Example script: [table_to_html_example.py](table_to_html_example.py)
