# Quick Start Guide - DeepSeek-OCR vLLM MonkeyPatch

## 🚀 TL;DR - Get Started in 30 Seconds

### Local Installation
```bash
# 1. Apply Large mode patch
python monkeypatch_vllm.py --mode large

# 2. Start server
./start_vllm_server.sh large
```

### Docker
```bash
# Start Docker container with Large mode on GPU 0, port 8002
./docker_vllm_server.sh large 0 8002
```

## 📋 All Available Modes

```bash
# Tiny mode (512×512, 64 tokens) - Fastest, lowest memory
python monkeypatch_vllm.py --mode tiny
./start_vllm_server.sh tiny

# Small mode (640×640, 100 tokens) - Balanced
python monkeypatch_vllm.py --mode small
./start_vllm_server.sh small

# Base mode (1024×1024, 256 tokens) - Default
python monkeypatch_vllm.py --mode base
./start_vllm_server.sh base

# Large mode (1280×1280, 400 tokens) - Highest quality
python monkeypatch_vllm.py --mode large
./start_vllm_server.sh large

# Gundam mode (1024 + 640 crops) - Dynamic cropping for large documents
python monkeypatch_vllm.py --mode gundam
./start_vllm_server.sh gundam
```

## 🐳 Docker Quick Commands

```bash
# Start server
./docker_vllm_server.sh [mode] [gpu] [port]

# Examples:
./docker_vllm_server.sh large 0 8002      # Large mode, GPU 0, port 8002
./docker_vllm_server.sh base 1 8000       # Base mode, GPU 1, port 8000
./docker_vllm_server.sh gundam "0,1" 8003 # Gundam mode, GPUs 0&1, port 8003

# View logs
docker logs -f vllm_ocr_large

# Stop server
docker stop vllm_ocr_large
```

## 🔧 Common Customizations

### Change Port
```bash
PORT=8080 ./start_vllm_server.sh large
```

### Reduce Memory Usage
```bash
GPU_MEMORY_UTIL=0.7 ./start_vllm_server.sh large
```

### Multi-GPU
```bash
TENSOR_PARALLEL_SIZE=2 ./start_vllm_server.sh large
```

### Custom API Key
```bash
API_KEY="my-secret-key" ./start_vllm_server.sh large
```

## 📡 Test Your Server

```bash
# Simple health check
curl http://localhost:8000/health

# Test with OpenAI client
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='123')
response = client.chat.completions.create(
    model='ocr',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=100
)
print(response.choices[0].message.content)
"
```

## 🔄 Switch Modes

```bash
# Switch from Base to Large
python monkeypatch_vllm.py --mode large
# Restart server
./start_vllm_server.sh large

# Switch back to Base
python monkeypatch_vllm.py --mode base
./start_vllm_server.sh base
```

## 🔙 Restore Original Settings

```bash
python monkeypatch_vllm.py --restore
```

## ❓ Troubleshooting

### Server won't start
```bash
# Check if vLLM is installed
pip show vllm

# Check GPU availability
nvidia-smi

# Check if port is in use
lsof -i :8000
```

### Out of memory
```bash
# Use smaller mode
./start_vllm_server.sh small

# Or reduce memory usage
GPU_MEMORY_UTIL=0.6 ./start_vllm_server.sh base
```

### Docker issues
```bash
# Check Docker logs
docker logs vllm_ocr_large

# Remove old containers
docker rm -f $(docker ps -a | grep vllm_ocr | awk '{print $1}')
```

## 📚 Full Documentation

See [README_MONKEYPATCH.md](README_MONKEYPATCH.md) for complete documentation.

## 🎯 Mode Selection Guide

| Mode   | When to Use |
|--------|-------------|
| Tiny   | Quick tests, low-res images, limited GPU memory |
| Small  | General purpose, balanced speed/quality |
| Base   | Default mode, good for most documents |
| Large  | High-quality OCR, detailed documents, sufficient GPU memory |
| Gundam | Very large documents, multi-page PDFs, need dynamic cropping |

## 💡 Pro Tips

1. **Start with Base mode** - It's the default and works well for most cases
2. **Use Large mode for quality** - When accuracy is more important than speed
3. **Use Gundam for large docs** - Automatically crops and processes large documents
4. **Monitor GPU memory** - Use `nvidia-smi` to check memory usage
5. **Adjust batch size** - If OOM, reduce `MAX_NUM_SEQS` environment variable

---

**Need help?** Check the full [README_MONKEYPATCH.md](README_MONKEYPATCH.md) or open an issue on GitHub.
