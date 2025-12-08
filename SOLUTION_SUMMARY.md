# Solution Summary - GitHub Issue #293

## 📌 Issue
**[MonkeyPatch] Workaround for vLLM online serving with custom modes / resolution**

Users need a way to easily switch between different DeepSeek-OCR modes (Tiny, Small, Base, Large, Gundam) when using vLLM online serving, without manually editing package files.

## ✅ Solution Provided

This solution provides three complementary approaches:

### 1. **Python MonkeyPatch Script** (`monkeypatch_vllm.py`)
- Automatically locates and patches vLLM package files
- Supports all 5 modes: Tiny, Small, Base, Large, Gundam
- Creates automatic backups before modifications
- Verifies patches were applied correctly
- Can restore original files

**Key Features:**
- ✅ Safe: Creates backups automatically
- ✅ Flexible: Supports predefined modes and custom configurations
- ✅ Verifiable: Confirms all changes were applied
- ✅ Reversible: Easy restoration from backups

### 2. **Local Server Launcher** (`start_vllm_server.sh`)
- Bash wrapper that applies patches and starts vLLM server
- Supports environment variables for customization
- Provides clear status messages
- Easy mode switching

**Key Features:**
- ✅ One-command startup
- ✅ Environment variable configuration
- ✅ Automatic patch application
- ✅ Clear error messages

### 3. **Docker Launcher** (`docker_vllm_server.sh`)
- Docker-based solution (matches original issue approach)
- Applies patches inside container (no host system changes)
- Supports multi-GPU configurations
- Easy container management

**Key Features:**
- ✅ Isolated environment
- ✅ No host system modifications
- ✅ Multi-GPU support
- ✅ Easy cleanup

## 📁 Files Created

| File | Purpose |
|------|---------|
| `monkeypatch_vllm.py` | Core patching script |
| `start_vllm_server.sh` | Local server launcher |
| `docker_vllm_server.sh` | Docker server launcher |
| `README_MONKEYPATCH.md` | Complete documentation |
| `QUICK_START.md` | Quick reference guide |
| `example_client.py` | Example API client |
| `SOLUTION_SUMMARY.md` | This file |

## 🚀 Usage Examples

### Quick Start (Local)
```bash
# Apply Large mode and start server
python monkeypatch_vllm.py --mode large
./start_vllm_server.sh large
```

### Quick Start (Docker)
```bash
# Start Docker container with Large mode
./docker_vllm_server.sh large 0 8002
```

### All Modes
```bash
# Tiny: 512×512 (64 tokens)
python monkeypatch_vllm.py --mode tiny

# Small: 640×640 (100 tokens)
python monkeypatch_vllm.py --mode small

# Base: 1024×1024 (256 tokens)
python monkeypatch_vllm.py --mode base

# Large: 1280×1280 (400 tokens)
python monkeypatch_vllm.py --mode large

# Gundam: Dynamic cropping
python monkeypatch_vllm.py --mode gundam
```

### Custom Configuration
```bash
# Custom resolution
python monkeypatch_vllm.py --base-size 1536 --image-size 1536 --crop-mode false
```

### Restore Original
```bash
python monkeypatch_vllm.py --restore
```

## 🔧 How It Works

### Patching Process

1. **Locate vLLM Package**
   - Automatically finds vLLM installation
   - Searches for DeepSeek-OCR processor and model files

2. **Create Backups**
   - Creates `.backup` files before any modifications
   - Ensures safe rollback capability

3. **Apply Patches**
   - **Processor file**: Modifies `BASE_SIZE`, `IMAGE_SIZE`, `CROP_MODE`
   - **Model file**: Changes `base_size = self.vision_config.image_size` to `base_size = BASE_SIZE`

4. **Verify Changes**
   - Confirms all patches were applied correctly
   - Reports any issues

### Files Modified

```
vllm/
├── transformers_utils/
│   └── processors/
│       └── deepseek_ocr.py          # Patched: BASE_SIZE, IMAGE_SIZE, CROP_MODE
└── model_executor/
    └── models/
        └── deepseek_ocr.py          # Patched: base_size assignment
```

## 📊 Mode Comparison

| Mode   | Base Size | Image Size | Crop | Tokens | Memory | Speed | Quality |
|--------|-----------|------------|------|--------|--------|-------|---------|
| Tiny   | 512       | 512        | No   | 64     | Low    | Fast  | Basic   |
| Small  | 640       | 640        | No   | 100    | Low    | Fast  | Good    |
| Base   | 1024      | 1024       | No   | 256    | Medium | Med   | Good    |
| Large  | 1280      | 1280       | No   | 400    | High   | Slow  | Best    |
| Gundam | 1024      | 640        | Yes  | Var    | High   | Slow  | Best    |

## 🎯 Use Cases

### Tiny Mode
- Quick tests and prototyping
- Low-resolution images
- Limited GPU memory
- Real-time processing needs

### Small Mode
- General purpose OCR
- Balanced speed/quality
- Mobile or edge deployment
- Batch processing

### Base Mode
- Default mode for most documents
- Standard quality requirements
- Typical GPU memory
- Production use

### Large Mode
- High-quality OCR requirements
- Detailed documents
- Sufficient GPU memory
- Accuracy over speed

### Gundam Mode
- Very large documents
- Multi-page PDFs
- Complex layouts
- Dynamic cropping needed

## 🔍 Technical Details

### Processor Modifications
```python
# Before
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True

# After (Large mode)
BASE_SIZE = 1280
IMAGE_SIZE = 1280
CROP_MODE = False
```

### Model Modifications
```python
# Before
base_size = self.vision_config.image_size

# After
base_size = BASE_SIZE
```

## 🧪 Testing

### Verify Installation
```bash
# Check vLLM is installed
pip show vllm

# Test monkeypatch script
python monkeypatch_vllm.py --help

# Test server launcher
./start_vllm_server.sh --help
```

### Test API
```bash
# Health check
curl http://localhost:8000/health

# Test with example client
python example_client.py --image test.jpg
```

## 🐛 Troubleshooting

### Common Issues

1. **vLLM not found**
   ```bash
   pip install vllm>=0.8.5
   ```

2. **Processor file not found**
   - Ensure vLLM has DeepSeek-OCR support
   - Check vLLM version: `pip show vllm`

3. **Out of memory**
   ```bash
   # Use smaller mode
   python monkeypatch_vllm.py --mode small
   
   # Or reduce memory usage
   GPU_MEMORY_UTIL=0.7 ./start_vllm_server.sh base
   ```

4. **Docker issues**
   ```bash
   # Check logs
   docker logs vllm_ocr_large
   
   # Ensure GPU is available
   nvidia-smi
   ```

## 📈 Performance Tips

1. **Start with Base mode** - Good balance for most cases
2. **Use Large for quality** - When accuracy matters most
3. **Use Gundam for large docs** - Automatic cropping helps
4. **Monitor GPU memory** - Use `nvidia-smi` to check usage
5. **Adjust batch size** - Reduce `MAX_NUM_SEQS` if OOM

## 🔄 Switching Modes

```bash
# Switch from Base to Large
python monkeypatch_vllm.py --mode large
# Restart server (Ctrl+C and rerun)
./start_vllm_server.sh large

# Switch to Gundam
python monkeypatch_vllm.py --mode gundam
./start_vllm_server.sh gundam

# Restore original
python monkeypatch_vllm.py --restore
```

## 🔐 Security Notes

1. **Backups**: Always created before modifications
2. **Reversible**: Easy restoration with `--restore`
3. **Docker isolation**: Container patches don't affect host
4. **API keys**: Change default key in production

## 📚 Documentation

- **Quick Start**: See `QUICK_START.md`
- **Full Documentation**: See `README_MONKEYPATCH.md`
- **Example Client**: See `example_client.py`

## 🤝 Contributing

This solution addresses GitHub Issue #293. For improvements:

1. Test with your vLLM version
2. Report issues with version info
3. Suggest enhancements
4. Share your use cases

## 📄 License

Follows the same license as the DeepSeek-OCR project.

## 🎉 Summary

This solution provides a **safe, flexible, and easy-to-use** workaround for switching between DeepSeek-OCR modes in vLLM online serving. It supports:

✅ All 5 modes (Tiny, Small, Base, Large, Gundam)  
✅ Custom configurations  
✅ Automatic backups  
✅ Easy restoration  
✅ Local and Docker deployment  
✅ Comprehensive documentation  
✅ Example client code  

**Ready to use in production or development environments!**

---

**Questions?** Check the documentation or open an issue on GitHub.
