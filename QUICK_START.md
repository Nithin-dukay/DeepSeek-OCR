# DeepSeek-OCR Quick Start Guide

## 🚀 30-Second Start

```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
bash setup_vllm_local.sh
python check_environment.py
```

## 📋 Choose Your Path

### Path 1: Local Machine (Recommended)
```bash
bash setup_vllm_local.sh
```
- ✅ Most stable
- ✅ Tested configuration
- ✅ Works with vLLM 0.8.5

### Path 2: Latest Features
```bash
bash setup_vllm_upstream.sh
```
- ✅ Latest vLLM features
- ✅ Officially supported
- ✅ Auto-updates dependencies

### Path 3: Kaggle/Colab
Copy code from `kaggle_notebook_example.py`
- ✅ Step-by-step cells
- ✅ Platform-optimized
- ✅ Includes examples

## 🔧 Fix Existing Error

Got `ImportError: cannot import name 'GenerationMixin'`?

**Quick Fix:**
```bash
python check_environment.py  # Diagnose
pip uninstall transformers -y
pip install transformers==4.46.3  # For vLLM 0.8.5
```

## 📖 Need More Help?

| Question | Answer |
|----------|--------|
| Detailed installation? | See [INSTALLATION.md](INSTALLATION.md) |
| Environment issues? | Run `python check_environment.py` |
| Kaggle/Colab setup? | Use `kaggle_notebook_example.py` |
| Troubleshooting? | Check [INSTALLATION.md](INSTALLATION.md) |

## ⚡ Quick Commands

```bash
# Check environment
python check_environment.py

# Install vLLM 0.8.5
bash setup_vllm_local.sh

# Install vLLM nightly
bash setup_vllm_upstream.sh

# Run inference (vLLM)
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py

# Run inference (Transformers)
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

## 🎯 Version Cheat Sheet

| Setup | transformers | vLLM | PyTorch |
|-------|--------------|------|---------|
| Local (0.8.5) | 4.46.3 | 0.8.5 | 2.6.0 |
| Nightly | ≥4.51.1 | nightly | ≥2.4.0 |

## ⚠️ Common Mistakes

❌ **DON'T:**
- Install transformers before vLLM
- Mix vLLM 0.8.5 with transformers 4.51.1
- Skip the environment check

✅ **DO:**
- Follow installation order
- Use automated scripts
- Verify with `check_environment.py`

## 💡 Pro Tips

1. **Always check first**: `python check_environment.py`
2. **Use automation**: Let scripts handle installation
3. **Read errors**: They tell you what's wrong
4. **Platform matters**: Kaggle/Colab need different steps

## 🆘 Still Stuck?

1. Read [INSTALLATION.md](INSTALLATION.md)
2. Check [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
3. Create issue with `check_environment.py` output

---

**Remember**: Installation order matters! Follow the guides and use the tools. 🎉
