# Architecture: Issue #288 Fix

## Problem Flow (Before Fix)

```
┌─────────────────────────────────────────────────────────────────┐
│ User Input                                                      │
│ "<image>\n<|grounding|>Convert to markdown, dont add space"    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Tokenizer                                                       │
│ Converts prompt to token IDs                                   │
│ [1, 2, 3, ..., 50] (50 tokens)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Model Generation Starts                                         │
│ Tokens: [1, 2, 3, ..., 50, 51, 52, ...]                       │
│         └─ Prompt ──┘  └─ Generated ─┘                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ OLD NoRepeatNGramLogitsProcessor                               │
│                                                                 │
│ ❌ Problem 1: Searches ENTIRE sequence (prompt + generated)    │
│    - Prompt content interferes with generation                 │
│                                                                 │
│ ❌ Problem 2: Blocks immediately on first match                │
│    - Too aggressive, blocks legitimate tokens                  │
│                                                                 │
│ ❌ Problem 3: No warm-up period                                │
│    - Starts blocking before model establishes patterns         │
│                                                                 │
│ ❌ Problem 4: Longer prompts = more false positives            │
│    - More prompt tokens = more interference                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Result: Model outputs repeated numbers                         │
│ "123 123 123 123 123 ..."                                      │
│ ❌ OCR FAILED                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Solution Flow (After Fix)

```
┌─────────────────────────────────────────────────────────────────┐
│ User Input                                                      │
│ "<image>\n<|grounding|>Convert to markdown, dont add space"    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ NEW: Prompt Validation (Optional)                              │
│                                                                 │
│ validate_and_optimize_prompt()                                 │
│ ✓ Checks for problematic patterns                             │
│ ✓ Provides warnings and suggestions                           │
│ ✓ Optimizes formatting                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Tokenizer                                                       │
│ Converts prompt to token IDs                                   │
│ [1, 2, 3, ..., 50] (50 tokens)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Model Generation Starts                                         │
│ Tokens: [1, 2, 3, ..., 50, 51, 52, 53, ...]                   │
│         └─ Prompt ──┘  └─ Generated ─┘                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ NEW NoRepeatNGramLogitsProcessor                               │
│                                                                 │
│ Step 1: Track Prompt Length                                    │
│ ┌─────────────────────────────────────────────────────────┐   │
│ │ First call: prompt_length = 50                          │   │
│ │ Separates prompt from generation                        │   │
│ └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│ Step 2: Check Generation Progress                              │
│ ┌─────────────────────────────────────────────────────────┐   │
│ │ generated_length = len(input_ids) - prompt_length      │   │
│ │ if generated_length < min_generated_tokens (10):       │   │
│ │     return scores  # Don't block yet                   │   │
│ └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│ Step 3: Search Only Generated Tokens                           │
│ ┌─────────────────────────────────────────────────────────┐   │
│ │ search_start = prompt_length                            │   │
│ │ search_end = len(input_ids) - ngram_size + 1          │   │
│ │ # Only looks at generated tokens, ignores prompt       │   │
│ └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│ Step 4: Adaptive Blocking                                      │
│ ┌─────────────────────────────────────────────────────────┐   │
│ │ if enable_adaptive:                                     │   │
│ │     # Only ban if token repeats MULTIPLE times         │   │
│ │     if repetition_count[token] > 1:                    │   │
│ │         banned_tokens.add(token)                       │   │
│ └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│ ✅ Benefit 1: Prompt doesn't interfere                         │
│ ✅ Benefit 2: Smarter blocking (adaptive)                      │
│ ✅ Benefit 3: Warm-up period (min_generated_tokens)            │
│ ✅ Benefit 4: Works with any prompt length                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Result: Model performs OCR correctly                           │
│ "# Document Title\n\nThis is the content..."                  │
│ ✅ OCR SUCCESS                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DeepSeek-OCR System                         │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
│   Prompt     │  │  Image           │  │   Model      │
│  Validation  │  │  Processing      │  │  Inference   │
│   (NEW)      │  │                  │  │              │
└──────┬───────┘  └────────┬─────────┘  └──────┬───────┘
       │                   │                    │
       │                   │                    │
       └───────────────────┼────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Logits Processing     │
              │  (ENHANCED)            │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Token Generation      │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  OCR Output            │
              └────────────────────────┘
```

## Key Improvements

### 1. Prompt-Aware Processing

**Before:**
```
[Prompt Tokens | Generated Tokens]
 └────────── All searched ────────┘
         ❌ Interference
```

**After:**
```
[Prompt Tokens | Generated Tokens]
 └─ Ignored ─┘  └─ Searched ──┘
         ✅ No Interference
```

### 2. Adaptive Blocking

**Before:**
```
Pattern found once → BAN immediately
❌ Too aggressive
```

**After:**
```
Pattern found once → Track
Pattern found twice → BAN
✅ Smarter blocking
```

### 3. Warm-up Period

**Before:**
```
Token 1: Check and block
Token 2: Check and block
Token 3: Check and block
❌ Blocks too early
```

**After:**
```
Token 1-10: Don't block (warm-up)
Token 11+: Check and block if needed
✅ Allows model to establish patterns
```

## Data Flow Example

### Scenario: Issue #288 Prompt

```
Input Prompt:
"<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

┌─────────────────────────────────────────────────────────────┐
│ Step 1: Tokenization                                        │
├─────────────────────────────────────────────────────────────┤
│ Token IDs: [1, 2, 3, ..., 65]  (65 tokens)                │
│ prompt_length = 65                                          │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Generation Begins                                   │
├─────────────────────────────────────────────────────────────┤
│ Token 66: generated_length = 1 < 10 → Don't block          │
│ Token 67: generated_length = 2 < 10 → Don't block          │
│ ...                                                         │
│ Token 75: generated_length = 10 → Start checking           │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: N-gram Checking (Token 76+)                        │
├─────────────────────────────────────────────────────────────┤
│ Search range: [66, 76] (only generated tokens)             │
│ Current prefix: tokens[75:76]                               │
│ Check for repetitions in generated tokens only             │
│ If found multiple times → Block                            │
│ If found once → Allow (adaptive mode)                      │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Output                                              │
├─────────────────────────────────────────────────────────────┤
│ "# Document Title\n\nThis is the content of the document." │
│ ✅ Correct OCR output                                       │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Matrix

| Use Case | ngram_size | window_size | min_generated | adaptive |
|----------|------------|-------------|---------------|----------|
| Document OCR | 40 | 90 | 15 | True |
| Table Extraction | 20 | 50 | 5 | True |
| General OCR | 30 | 90 | 10 | True |
| Strict (No Repeat) | 30 | 90 | 10 | False |

## Testing Strategy

```
┌─────────────────────────────────────────────────────────────┐
│ Test Suite                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Unit Tests                                               │
│    ✓ NoRepeatNGramLogitsProcessor logic                    │
│    ✓ Prompt validation functions                           │
│    ✓ Optimization functions                                │
│                                                             │
│ 2. Integration Tests                                        │
│    ✓ Original prompt (should work)                         │
│    ✓ Modified prompt (Issue #288)                          │
│    ✓ Various custom prompts                                │
│                                                             │
│ 3. Regression Tests                                         │
│    ✓ Backward compatibility                                │
│    ✓ Existing code still works                             │
│                                                             │
│ 4. Performance Tests                                        │
│    ✓ No significant overhead                               │
│    ✓ Generation quality improved                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Summary

The fix addresses Issue #288 through a multi-layered approach:

1. **Enhanced Processor:** Smarter n-gram blocking that doesn't interfere with prompts
2. **Validation Tools:** Helps users create better prompts
3. **Documentation:** Clear guides and examples
4. **Testing:** Comprehensive test suite

Result: ✅ Modified prompts now work correctly while maintaining repetition prevention.
