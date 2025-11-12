# Solution for GitHub Issue #249: How to Get Streamed Tokens from model.infer()

## Problem Statement

The `model.infer()` method in the HuggingFace transformers implementation:
- Displays output to console in real-time (via `TextStreamer`)
- Returns `None` by default
- Only returns text when `eval_mode=True` (but without streaming)
- Users need to stream tokens to clients instead of just printing to console

## Root Cause

The `infer` method (located in `modeling_deepseekocr.py` on HuggingFace Hub) has the following behavior:

```python
# Default mode: streams to console, returns None
streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
output_ids = self.generate(..., streamer=streamer, ...)
# No return statement -> returns None

# Eval mode: no streaming, returns full text
if eval_mode:
    outputs = tokenizer.decode(output_ids[0, ...])
    return outputs  # Returns the full generated text
```

## Solutions

We provide **three solutions** with increasing complexity and capability:

### Solution 1: Use `eval_mode=True` (Simple, No Streaming)

**Best for**: Simple use cases where you can wait for full generation

```python
result = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path,
    base_size=1024, 
    image_size=640, 
    crop_mode=True,
    eval_mode=True  # Returns the full output instead of None
)

# Now you can send 'result' to your client
print(f"Generated text: {result}")
```

**Pros**: 
- Simple, works immediately
- No code changes needed
- Returns clean text output

**Cons**: 
- No streaming (client waits for full generation)
- Not suitable for long documents

---

### Solution 2: Custom Streamer with Queue (Token Streaming)

**Best for**: Real-time streaming to clients with HuggingFace transformers

This solution creates a custom streamer that captures tokens in a queue instead of printing to console.

See: `examples/solution2_custom_streamer.py`

**Pros**: 
- True token-by-token streaming
- Works with existing HuggingFace model
- Can integrate with web frameworks

**Cons**: 
- Requires monkey-patching or model code modification
- More complex implementation

---

### Solution 3: Use vLLM with Async Streaming (Production Ready)

**Best for**: Production deployments, high performance, web APIs

The repository already includes a vLLM implementation with native async streaming support.

See: 
- `examples/solution3_vllm_fastapi_server.py` - FastAPI server with streaming
- `examples/solution3_vllm_client.py` - Client example

**Pros**: 
- Native async streaming support
- High performance (2500+ tokens/s on A100)
- Production-ready
- Easy integration with FastAPI/Flask
- Supports Server-Sent Events (SSE)

**Cons**: 
- Requires vLLM installation
- Different API than HuggingFace transformers

---

## Comparison Table

| Feature | Solution 1 (eval_mode) | Solution 2 (Custom Streamer) | Solution 3 (vLLM) |
|---------|----------------------|----------------------------|-------------------|
| Streaming | ❌ No | ✅ Yes | ✅ Yes |
| Easy to implement | ✅ Very easy | ⚠️ Moderate | ✅ Easy |
| Performance | Standard | Standard | ⚠️ High (2500+ tok/s) |
| Production ready | ⚠️ Basic | ⚠️ Moderate | ✅ Yes |
| Async support | ❌ No | ⚠️ With threading | ✅ Native |
| Code changes | None | Moderate | Minimal |

## Recommended Approach

**For quick prototyping**: Use Solution 1 (eval_mode=True)

**For production with streaming**: Use Solution 3 (vLLM with FastAPI)

**For HuggingFace-only environments**: Use Solution 2 (Custom Streamer)

## Installation Requirements

### Solution 1 & 2 (HuggingFace)
```bash
pip install transformers torch torchvision Pillow
```

### Solution 3 (vLLM)
```bash
pip install vllm fastapi uvicorn
# Or use the existing vLLM setup in DeepSeek-OCR-vllm/
```

## Additional Resources

- Original vLLM streaming example: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`
- HuggingFace example: `DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py`
- Configuration: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`

## Testing

Each solution includes example code that can be tested:

```bash
# Solution 1
python examples/solution1_eval_mode.py

# Solution 2
python examples/solution2_custom_streamer.py

# Solution 3 (Server)
python examples/solution3_vllm_fastapi_server.py
# Then in another terminal:
python examples/solution3_vllm_client.py
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- Discord: https://discord.gg/Tc7c45Zzu5
