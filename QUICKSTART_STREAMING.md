# Quick Start: Streaming Tokens from DeepSeek-OCR

This guide helps you quickly solve **GitHub Issue #249**: Getting streamed tokens from `model.infer()`.

## TL;DR - Choose Your Solution

```python
# ❌ PROBLEM: This returns None
result = model.infer(tokenizer, prompt=prompt, image_file=image_file)
print(result)  # None

# ✅ SOLUTION 1: Simple fix - add eval_mode=True
result = model.infer(tokenizer, prompt=prompt, image_file=image_file, eval_mode=True)
print(result)  # Returns the generated text!

# ✅ SOLUTION 2: Custom streamer for token-by-token streaming
# See: examples/solution2_custom_streamer.py

# ✅ SOLUTION 3: Production API with vLLM (RECOMMENDED)
# See: examples/solution3_vllm_fastapi_server.py
```

---

## Solution 1: Immediate Fix (30 seconds)

**Just add `eval_mode=True`** to make `model.infer()` return the output:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)

# Add eval_mode=True
result = model.infer(
    tokenizer,
    prompt="<image>\n<|grounding|>Convert the document to markdown.",
    image_file='your_image.jpg',
    output_path='output',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    eval_mode=True  # ← ADD THIS LINE
)

# Now result contains the generated text
print(result)
```

**When to use**: Quick prototyping, simple applications, no streaming needed.

---

## Solution 2: Token Streaming (5 minutes)

For real-time token-by-token streaming, use a custom streamer:

```python
from transformers import TextStreamer
from queue import Queue

class CustomTokenStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=False):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.token_queue = Queue()
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        eos_text = self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False)
        text = text.replace(eos_text, "\n")
        self.token_queue.put(text)
        if stream_end:
            self.token_queue.put(None)
    
    def get_tokens(self):
        while True:
            token = self.token_queue.get()
            if token is None:
                break
            yield token

# Use the streamer
streamer = CustomTokenStreamer(tokenizer)
# ... integrate with model.generate() ...
for token in streamer.get_tokens():
    print(token, end='', flush=True)
```

**Full implementation**: See `examples/solution2_custom_streamer.py`

**When to use**: Need real-time streaming, must use HuggingFace transformers.

---

## Solution 3: Production API (10 minutes) ⭐ RECOMMENDED

For production deployments with high performance and easy client integration:

### Step 1: Start the Server

```bash
# Install dependencies (if not already installed)
pip install vllm fastapi uvicorn

# Start the streaming API server
python examples/solution3_vllm_fastapi_server.py
```

Server starts on `http://localhost:8000` with:
- Interactive docs at `/docs`
- Health check at `/health`
- Multiple streaming endpoints

### Step 2: Use the API

**Python Client (SSE Streaming)**:
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/ocr/stream",
    json={
        "image_path": "your_image.jpg",
        "prompt": "<image>\n<|grounding|>Convert the document to markdown."
    },
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        data = json.loads(line[6:])
        if 'token' in data:
            print(data['token'], end='', flush=True)
```

**JavaScript Client (Browser)**:
```javascript
fetch('http://localhost:8000/ocr/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        image_path: 'your_image.jpg',
        prompt: '<image>\n<|grounding|>Convert the document to markdown.'
    })
})
.then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    function readStream() {
        reader.read().then(({ done, value }) => {
            if (done) return;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            lines.forEach(line => {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.substring(6));
                    if (data.token) {
                        document.getElementById('output').textContent += data.token;
                    }
                }
            });
            
            readStream();
        });
    }
    
    readStream();
});
```

**File Upload**:
```bash
curl -X POST "http://localhost:8000/ocr/upload/stream" \
  -F "file=@your_image.jpg" \
  -F "prompt=<image>\nFree OCR."
```

**When to use**: Production deployments, web APIs, high performance needed (2500+ tokens/s).

---

## Comparison

| Feature | Solution 1 | Solution 2 | Solution 3 |
|---------|-----------|-----------|-----------|
| **Complexity** | ⭐ Very Easy | ⭐⭐ Moderate | ⭐ Easy |
| **Streaming** | ❌ No | ✅ Yes | ✅ Yes |
| **Performance** | Standard | Standard | ⚡ High (2500+ tok/s) |
| **Setup Time** | 30 seconds | 5 minutes | 10 minutes |
| **Production Ready** | ⚠️ Basic | ⚠️ Moderate | ✅ Yes |
| **Client Support** | Any | Any | Python, JS, cURL |
| **API Docs** | N/A | N/A | ✅ Interactive |

---

## Testing Your Solution

### Test Solution 1
```bash
python examples/solution1_eval_mode.py
```

### Test Solution 2
```bash
python examples/solution2_custom_streamer.py
```

### Test Solution 3
```bash
# Terminal 1: Start server
python examples/solution3_vllm_fastapi_server.py

# Terminal 2: Test client
python examples/solution3_vllm_client.py

# Or use browser
open http://localhost:8000/docs
```

---

## Common Issues

### Issue: `model.infer()` still returns None
**Solution**: Make sure you added `eval_mode=True`:
```python
result = model.infer(..., eval_mode=True)  # ← Don't forget this!
```

### Issue: Server won't start (Solution 3)
**Check**:
1. vLLM installed: `pip install vllm`
2. CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
3. GPU memory: `nvidia-smi`

### Issue: Connection refused
**Solution**: Make sure server is running:
```bash
curl http://localhost:8000/health
```

---

## Next Steps

1. **Read Full Documentation**: [`ISSUE_249_SOLUTION.md`](ISSUE_249_SOLUTION.md)
2. **Explore Examples**: [`examples/README.md`](examples/README.md)
3. **API Documentation**: http://localhost:8000/docs (when server is running)
4. **Original Code**: 
   - HuggingFace: `DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py`
   - vLLM: `DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py`

---

## Support

- **GitHub Issues**: https://github.com/deepseek-ai/DeepSeek-OCR/issues/249
- **Discord**: https://discord.gg/Tc7c45Zzu5
- **Documentation**: See `ISSUE_249_SOLUTION.md` for detailed explanations

---

## Summary

**Quick Fix**: Add `eval_mode=True` to `model.infer()`

**Production**: Use Solution 3 (vLLM + FastAPI) for best performance and streaming

**Custom Needs**: See Solution 2 for custom streaming implementations

Choose the solution that best fits your needs and follow the examples provided!
