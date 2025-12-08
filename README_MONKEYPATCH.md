# DeepSeek-OCR vLLM MonkeyPatch Solution

This solution provides a workaround for running vLLM online serving with custom modes and resolutions for DeepSeek-OCR.

## 🎯 Problem

The DeepSeek-OCR model supports multiple resolution modes, but vLLM's online serving doesn't provide an easy way to switch between them. This solution allows you to:

- Switch between predefined modes (Tiny, Small, Base, Large, Gundam)
- Use custom resolutions
- Modify vLLM package files safely with automatic backups

## 📋 Supported Modes

| Mode   | Base Size | Image Size | Crop Mode | Vision Tokens | Use Case |
|--------|-----------|------------|-----------|---------------|----------|
| Tiny   | 512×512   | 512×512    | False     | 64            | Fast processing, low memory |
| Small  | 640×640   | 640×640    | False     | 100           | Balanced speed/quality |
| Base   | 1024×1024 | 1024×1024  | False     | 256           | Default mode |
| Large  | 1280×1280 | 1280×1280  | False     | 400           | High quality OCR |
| Gundam | 1024×1024 | 640×640    | True      | Dynamic       | Multi-crop for large docs |

## 🚀 Quick Start

### Option 1: Local Installation (Recommended)

```bash
# 1. Apply monkeypatch for Large mode
python monkeypatch_vllm.py --mode large

# 2. Start vLLM server
./start_vllm_server.sh large

# Or use environment variables for customization
PORT=8002 GPU_MEMORY_UTIL=0.8 ./start_vllm_server.sh large
```

### Option 2: Docker (Original Issue Approach)

```bash
# Start Docker container with Large mode
./docker_vllm_server.sh large 0 8002

# Arguments: [mode] [gpu_device] [port]
# - mode: tiny, small, base, large, gundam
# - gpu_device: GPU device ID (e.g., 0, 1, or "0,1" for multiple)
# - port: Host port to expose (default: 8002)
```

## 📖 Detailed Usage

### 1. MonkeyPatch Script (`monkeypatch_vllm.py`)

This Python script modifies vLLM package files to support custom modes.

#### Using Predefined Modes

```bash
# Apply Large mode (1280×1280)
python monkeypatch_vllm.py --mode large

# Apply Gundam mode (dynamic cropping)
python monkeypatch_vllm.py --mode gundam

# Apply Tiny mode (512×512)
python monkeypatch_vllm.py --mode tiny
```

#### Using Custom Configuration

```bash
# Custom resolution: 1536×1536 without cropping
python monkeypatch_vllm.py --base-size 1536 --image-size 1536 --crop-mode false

# Custom with cropping enabled
python monkeypatch_vllm.py --base-size 1024 --image-size 768 --crop-mode true
```

#### Restoring Original Files

```bash
# Restore from automatic backups
python monkeypatch_vllm.py --restore
```

#### Advanced Options

```bash
# Specify vLLM installation path manually
python monkeypatch_vllm.py --mode large --vllm-path /custom/path/to/vllm

# Get help
python monkeypatch_vllm.py --help
```

### 2. Server Launcher Script (`start_vllm_server.sh`)

Wrapper script that applies monkeypatch and starts vLLM server.

```bash
# Make script executable
chmod +x start_vllm_server.sh

# Start with Large mode
./start_vllm_server.sh large

# Start with custom vLLM arguments
./start_vllm_server.sh base --gpu-memory-utilization 0.8 --tensor-parallel-size 2

# Use environment variables
MODEL_NAME="deepseek-ai/DeepSeek-OCR" \
PORT=8000 \
API_KEY="my-secret-key" \
GPU_MEMORY_UTIL=0.9 \
./start_vllm_server.sh large
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `deepseek-ai/DeepSeek-OCR` | Model name or path |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `API_KEY` | `123` | API authentication key |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `GPU_MEMORY_UTIL` | `0.9` | GPU memory utilization (0.0-1.0) |
| `MAX_MODEL_LEN` | `8192` | Maximum sequence length |
| `MAX_NUM_SEQS` | `100` | Maximum number of sequences |
| `MAX_NUM_BATCHED_TOKENS` | `1280` | Maximum batched tokens |

### 3. Docker Launcher Script (`docker_vllm_server.sh`)

Docker wrapper that applies patches inside the container.

```bash
# Make script executable
chmod +x docker_vllm_server.sh

# Start with Large mode on GPU 0, port 8002
./docker_vllm_server.sh large 0 8002

# Start with Base mode on GPU 1, port 8000
./docker_vllm_server.sh base 1 8000

# Start with Gundam mode on multiple GPUs
./docker_vllm_server.sh gundam "0,1" 8003

# Custom container name
CONTAINER_NAME="my_ocr_server" ./docker_vllm_server.sh large 0 8002
```

#### Docker Management

```bash
# View logs
docker logs -f vllm_ocr_large

# Stop server
docker stop vllm_ocr_large

# Remove container
docker rm vllm_ocr_large

# List running containers
docker ps | grep vllm_ocr
```

## 🔧 How It Works

### MonkeyPatch Process

1. **Locate vLLM Package**: Automatically finds your vLLM installation
2. **Create Backups**: Creates `.backup` files before any modifications
3. **Patch Processor File**: Modifies `BASE_SIZE`, `IMAGE_SIZE`, `CROP_MODE` constants
4. **Patch Model File**: Updates model to use `BASE_SIZE` instead of `vision_config.image_size`
5. **Verify Changes**: Confirms all patches were applied correctly

### Files Modified

- **Processor**: `vllm/transformers_utils/processors/deepseek_ocr.py`
  - `BASE_SIZE`: Global view resolution
  - `IMAGE_SIZE`: Local crop resolution
  - `CROP_MODE`: Enable/disable dynamic cropping

- **Model** (if needed): `vllm/model_executor/models/deepseek_ocr.py`
  - Changes `base_size = self.vision_config.image_size` to `base_size = BASE_SIZE`

## 📝 Example API Usage

After starting the server, you can use the OpenAI-compatible API:

```python
import requests
from PIL import Image
import base64
from io import BytesIO

# Load and encode image
image = Image.open("document.jpg")
buffered = BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# API request
response = requests.post(
    "http://localhost:8002/v1/chat/completions",
    headers={
        "Authorization": "Bearer 123",
        "Content-Type": "application/json"
    },
    json={
        "model": "ocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<image>\n<|grounding|>Convert the document to markdown."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.0
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Using OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="123"
)

response = client.chat.completions.create(
    model="ocr",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>\nFree OCR."},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ],
    max_tokens=8192,
    temperature=0.0
)

print(response.choices[0].message.content)
```

## 🐛 Troubleshooting

### Issue: "vLLM package not found"

```bash
# Install vLLM first
pip install vllm

# Or specify path manually
python monkeypatch_vllm.py --mode large --vllm-path /path/to/vllm
```

### Issue: "Could not find DeepSeek OCR processor file"

This means vLLM doesn't have DeepSeek-OCR support yet. Install a version that supports it:

```bash
# Install vLLM with DeepSeek-OCR support
pip install vllm>=0.8.5
```

### Issue: Patches not applied

```bash
# Check vLLM installation
python -c "import vllm; print(vllm.__file__)"

# Restore and reapply
python monkeypatch_vllm.py --restore
python monkeypatch_vllm.py --mode large
```

### Issue: Docker container fails to start

```bash
# Check Docker logs
docker logs vllm_ocr_large

# Ensure GPU is available
nvidia-smi

# Check if port is already in use
lsof -i :8002
```

### Issue: Out of memory errors

Try reducing memory usage:

```bash
# Reduce GPU memory utilization
GPU_MEMORY_UTIL=0.7 ./start_vllm_server.sh large

# Use smaller mode
./start_vllm_server.sh base

# Reduce batch size
MAX_NUM_SEQS=50 ./start_vllm_server.sh large
```

## ⚠️ Important Notes

1. **Backup Safety**: The script automatically creates `.backup` files before modifying anything
2. **Version Compatibility**: Tested with vLLM 0.8.5+. May need adjustments for other versions
3. **Persistence**: Patches persist until you restore backups or reinstall vLLM
4. **Docker Isolation**: Docker patches are applied inside the container and don't affect your host system
5. **Multiple Modes**: To switch modes, simply run the monkeypatch script again with a different mode

## 🔄 Switching Between Modes

```bash
# Switch to Large mode
python monkeypatch_vllm.py --mode large
./start_vllm_server.sh large

# Later, switch to Gundam mode
python monkeypatch_vllm.py --mode gundam
./start_vllm_server.sh gundam

# Restore original settings
python monkeypatch_vllm.py --restore
```

## 📚 Additional Resources

- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM Documentation](https://docs.vllm.ai/)
- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)

## 🤝 Contributing

If you encounter issues or have improvements:

1. Check existing GitHub issues
2. Create a new issue with:
   - Your vLLM version (`pip show vllm`)
   - Error messages
   - Steps to reproduce

## 📄 License

This solution is provided as-is for the DeepSeek-OCR project. Follow the same license as the main project.

---

**Note**: This is a workaround solution for GitHub Issue #293. For production use, consider requesting official support for mode switching in vLLM upstream.
