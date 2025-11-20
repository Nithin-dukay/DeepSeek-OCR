# Fix for GitHub Issue #250: Model Emits Weird Output on Series of Dots

## Problem Description

The DeepSeek-OCR model was entering an infinite loop when processing certain document pages, continuously emitting a series of dots like:
```
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
```

This issue occurred when the model encountered specific patterns in documents that triggered repetitive token generation.

## Root Cause Analysis

The original `NoRepeatNGramLogitsProcessor` only checked for exact n-gram matches to prevent repetitions. However, this approach had limitations:

1. **Pattern Tokenization**: Dots with spaces (`. . . .`) could be tokenized as alternating tokens (e.g., `[dot_token, space_token, dot_token, space_token, ...]`), which created repeating 2-token or 3-token patterns that weren't caught by the n-gram detector.

2. **Insufficient Detection**: The n-gram approach required exact prefix matches, which didn't catch all types of repetitive patterns.

3. **No Frequency Analysis**: There was no mechanism to detect when a token appeared too frequently in the output window, even if it wasn't part of an exact n-gram match.

## Solution Implemented

### Enhanced NoRepeatNGramLogitsProcessor

The fix enhances the `NoRepeatNGramLogitsProcessor` with multiple detection mechanisms:

#### 1. **Consecutive Repetition Detection**
- Detects when the same token appears multiple times consecutively
- Default threshold: 5 consecutive identical tokens
- Example: `[13, 13, 13, 13, 13]` → token 13 is banned

#### 2. **Pattern Repetition Detection**
- Detects repeating 2-token and 3-token patterns
- Checks if a pattern repeats 3 or more times
- Example: `[13, 14, 13, 14, 13, 14]` → tokens 13 and 14 are banned
- This specifically catches the `. <space> . <space>` pattern from Issue #250

#### 3. **Frequency-Based Detection**
- Analyzes token frequency within the sliding window
- Bans tokens that exceed a configurable ratio (default: 40%)
- Example: If token 13 appears in 50% of the last 90 tokens → token 13 is banned

#### 4. **Whitelist Protection**
- Maintains backward compatibility with whitelisted tokens
- Table-related tokens (`<td>`, `</td>`) remain whitelisted
- Whitelisted tokens are never banned, even if they repeat

### Configuration Parameters

New parameters added to `config.py`:

```python
# Enhanced repetition detection parameters (Fix for Issue #250)
ENABLE_ENHANCED_REPETITION_DETECTION = True  # Enable enhanced pattern detection
MAX_TOKEN_REPETITION_RATIO = 0.4  # Max ratio of same token in window (0.4 = 40%)
MAX_CONSECUTIVE_REPETITIONS = 5  # Max consecutive identical tokens allowed
```

### Files Modified

1. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py`**
   - Enhanced the `NoRepeatNGramLogitsProcessor` class
   - Added new detection mechanisms
   - Added configurable parameters

2. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`**
   - Added configuration parameters for enhanced detection

3. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`**
   - Updated to use enhanced processor with new parameters

4. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_pdf.py`**
   - Updated to use enhanced processor with new parameters

5. **`DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch.py`**
   - Updated to use enhanced processor with new parameters

## Testing

### Test Results

All detection mechanisms have been tested and verified:

```
✓ Test 1: Consecutive Repetitions - PASS
✓ Test 2: 2-Token Pattern Repetitions - PASS
✓ Test 3: 3-Token Pattern Repetitions - PASS
✓ Test 4: Frequency-Based Detection - PASS
✓ Test 5: Normal Text (No False Positives) - PASS
✓ Test 6: Edge Case - Threshold Boundary - PASS
✓ Test 7: Simulating GitHub Issue #250 - PASS
```

The test specifically simulating the infinite dot pattern from Issue #250 confirms that the enhanced detection would catch and prevent this issue.

## Usage

### Default Behavior

The fix is enabled by default. No changes are required to existing code:

```python
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Enhanced detection is enabled by default
logits_processors = [NoRepeatNGramLogitsProcessor(
    ngram_size=30, 
    window_size=90, 
    whitelist_token_ids={128821, 128822}
)]
```

### Custom Configuration

You can customize the detection parameters:

```python
logits_processors = [NoRepeatNGramLogitsProcessor(
    ngram_size=30,
    window_size=90,
    whitelist_token_ids={128821, 128822},
    max_token_repetition_ratio=0.3,  # More aggressive (30% threshold)
    max_consecutive_repetitions=3,    # Stricter consecutive limit
    enable_enhanced_detection=True
)]
```

### Disabling Enhanced Detection

If needed, you can disable the enhanced detection to revert to original behavior:

```python
logits_processors = [NoRepeatNGramLogitsProcessor(
    ngram_size=30,
    window_size=90,
    whitelist_token_ids={128821, 128822},
    enable_enhanced_detection=False  # Disable enhanced detection
)]
```

## Backward Compatibility

- ✅ All existing functionality is preserved
- ✅ Whitelisted tokens continue to work as expected
- ✅ Original n-gram detection still operates
- ✅ Can be disabled if needed via configuration
- ✅ No breaking changes to API

## Performance Impact

The enhanced detection adds minimal computational overhead:
- Pattern checks: O(k) where k is pattern length (2 or 3)
- Frequency analysis: O(w) where w is window size
- Overall impact: Negligible compared to model inference time

## Recommendations

1. **Keep Default Settings**: The default parameters (40% ratio, 5 consecutive) are well-balanced for most use cases.

2. **Monitor Output**: If you notice legitimate repeated content being cut off, consider:
   - Adding those tokens to the whitelist
   - Increasing `MAX_TOKEN_REPETITION_RATIO`
   - Increasing `MAX_CONSECUTIVE_REPETITIONS`

3. **Table Processing**: The whitelist for `<td>` and `</td>` tokens ensures table processing remains unaffected.

## Future Improvements

Potential enhancements for future versions:
- Dynamic threshold adjustment based on content type
- Machine learning-based repetition detection
- Context-aware whitelisting
- Per-token type repetition limits

## Conclusion

This fix addresses GitHub Issue #250 by implementing comprehensive repetition detection that catches various patterns of infinite loops, including the specific dot repetition issue reported. The solution is backward compatible, configurable, and thoroughly tested.
