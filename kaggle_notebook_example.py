"""
DeepSeek-OCR Kaggle/Colab Notebook Example

This script demonstrates the CORRECT installation order for DeepSeek-OCR
in Kaggle or Google Colab environments to avoid the ImportError with GenerationMixin.

IMPORTANT: Run each section in separate cells in your notebook!

Author: DeepSeek-OCR Team
Issue: Fixes GitHub Issue #237 - ImportError: cannot import name 'GenerationMixin'
"""

# ============================================================================
# CELL 1: Check GPU and Environment
# ============================================================================
print("=" * 60)
print("CELL 1: Checking GPU and Environment")
print("=" * 60)

import subprocess
import sys

# Check GPU
print("\n📊 GPU Information:")
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])

print(f"\n🐍 Python Version: {sys.version}")
print(f"📍 Python Executable: {sys.executable}")


# ============================================================================
# CELL 2: Install PyTorch with CUDA 11.8
# ============================================================================
print("\n" + "=" * 60)
print("CELL 2: Installing PyTorch 2.6.0 with CUDA 11.8")
print("=" * 60)
print("\n⚠️  CRITICAL: Install PyTorch BEFORE vLLM!")
print("This ensures correct CUDA version compatibility.\n")

# Install PyTorch with CUDA 11.8
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
    "--index-url", "https://download.pytorch.org/whl/cu118",
    "-q"
], check=True)

print("✅ PyTorch installed successfully!")

# Verify PyTorch installation
import torch
print(f"\n📦 PyTorch Version: {torch.__version__}")
print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 CUDA Version: {torch.version.cuda}")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# CELL 3: Install vLLM Nightly
# ============================================================================
print("\n" + "=" * 60)
print("CELL 3: Installing vLLM Nightly")
print("=" * 60)
print("\n⚠️  IMPORTANT: Do NOT install transformers separately!")
print("vLLM will install the correct transformers version automatically.\n")

# Install vLLM nightly
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "--pre", "vllm",
    "--extra-index-url", "https://wheels.vllm.ai/nightly",
    "-q"
], check=True)

print("✅ vLLM installed successfully!")

# Verify vLLM installation
import vllm
import transformers
print(f"\n📦 vLLM Version: {vllm.__version__}")
print(f"📦 Transformers Version: {transformers.__version__}")


# ============================================================================
# CELL 4: Install Additional Dependencies
# ============================================================================
print("\n" + "=" * 60)
print("CELL 4: Installing Additional Dependencies")
print("=" * 60)

# Install other required packages
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "Pillow", "PyMuPDF", "img2pdf", "einops", "easydict", "addict", "numpy",
    "-q"
], check=True)

print("✅ Additional dependencies installed successfully!")


# ============================================================================
# CELL 5: Install Flash Attention (Optional)
# ============================================================================
print("\n" + "=" * 60)
print("CELL 5: Installing Flash Attention (Optional)")
print("=" * 60)
print("\n⚠️  This step is optional and may take several minutes.")
print("If it fails, you can skip it - the model will work without it.\n")

try:
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "flash-attn",
        "--no-build-isolation",
        "-q"
    ], check=True, timeout=600)  # 10 minute timeout
    print("✅ Flash Attention installed successfully!")
except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
    print("⚠️  Flash Attention installation failed or timed out (this is OK)")
    print("The model will use standard attention instead.")


# ============================================================================
# CELL 6: Verify Installation
# ============================================================================
print("\n" + "=" * 60)
print("CELL 6: Verifying Installation")
print("=" * 60)

def verify_installation():
    """Verify all packages are installed correctly"""
    checks_passed = True
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - will run on CPU (very slow)")
    except ImportError:
        print("❌ PyTorch not installed")
        checks_passed = False
    
    # Check vLLM
    try:
        import vllm
        print(f"✅ vLLM {vllm.__version__}")
    except ImportError:
        print("❌ vLLM not installed")
        checks_passed = False
    
    # Check transformers
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        checks_passed = False
    
    # Check other packages
    required_packages = ['PIL', 'einops', 'easydict', 'addict', 'numpy']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            checks_passed = False
    
    # Check flash-attn (optional)
    try:
        import flash_attn
        print(f"✅ flash-attn {flash_attn.__version__} (optional)")
    except ImportError:
        print("⚠️  flash-attn not installed (optional)")
    
    return checks_passed

if verify_installation():
    print("\n" + "=" * 60)
    print("🎉 Installation completed successfully!")
    print("=" * 60)
else:
    print("\n" + "=" * 60)
    print("❌ Installation incomplete - please check errors above")
    print("=" * 60)


# ============================================================================
# CELL 7: Example Usage
# ============================================================================
print("\n" + "=" * 60)
print("CELL 7: Example Usage")
print("=" * 60)

# Example code for using DeepSeek-OCR
example_code = '''
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

# Load model (this may take 2-5 minutes on first run)
print("Loading DeepSeek-OCR model...")
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor],
    max_model_len=8192,  # Adjust based on your GPU memory
    trust_remote_code=True,
    dtype="bfloat16",    # Kaggle T4 supports bfloat16
)
print("✅ Model loaded successfully!")

# Load your image
image = Image.open("/kaggle/input/your-dataset/test_image.jpg").convert("RGB")

# Set prompt based on your task
prompt = "<image>\\n<|grounding|>Convert the document to markdown."
# Other prompt options:
# prompt = "<image>\\nFree OCR."
# prompt = "<image>\\n<|grounding|>OCR this image."
# prompt = "<image>\\nDescribe this image in detail."

# Prepare input
model_input = [{
    "prompt": prompt,
    "multi_modal_data": {"image": image}
}]

# Configure sampling parameters
sampling_params = SamplingParams(
    temperature=0.0,        # Deterministic output
    max_tokens=8192,        # Maximum output length
    extra_args=dict(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},  # <td>, </td> tokens
    ),
    skip_special_tokens=False,
)

# Generate OCR output
print("Generating OCR output...")
outputs = llm.generate(model_input, sampling_params)

# Print result
print("\\n" + "="*50)
print("OCR OUTPUT:")
print("="*50)
print(outputs[0].outputs[0].text)

# Save result
with open("ocr_output.txt", "w", encoding="utf-8") as f:
    f.write(outputs[0].outputs[0].text)
print("\\n✅ Output saved to ocr_output.txt")
'''

print("\n📝 Copy and paste this code into a new cell to use DeepSeek-OCR:\n")
print(example_code)


# ============================================================================
# CELL 8: Troubleshooting Tips
# ============================================================================
print("\n" + "=" * 60)
print("CELL 8: Troubleshooting Tips")
print("=" * 60)

troubleshooting_tips = """
🔧 Common Issues and Solutions:

1. ImportError: cannot import name 'GenerationMixin'
   ❌ Cause: Wrong transformers version or installation order
   ✅ Fix: Restart kernel and follow the installation order above
   
2. CUDA Out of Memory
   ❌ Cause: GPU memory exhausted
   ✅ Fix: Reduce max_model_len to 4096 or use smaller images
   
3. Model download is slow
   ❌ Cause: Network speed
   ✅ Fix: Be patient, first download takes 5-10 minutes
   
4. vLLM import error
   ❌ Cause: Incompatible package versions
   ✅ Fix: Restart kernel and reinstall following the order above

5. Flash attention installation fails
   ❌ Cause: Compilation issues
   ✅ Fix: Skip it - it's optional, model works without it

📚 For more help:
- See INSTALLATION.md in the repository
- Check GitHub Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues
- Run check_environment.py to diagnose issues

💡 Pro Tips:
- Always install PyTorch BEFORE vLLM
- Never install transformers separately when using vLLM nightly
- Use bfloat16 dtype on Kaggle T4 GPUs for better performance
- Adjust max_model_len based on your GPU memory (4096-8192)
"""

print(troubleshooting_tips)


# ============================================================================
# CELL 9: Performance Tips
# ============================================================================
print("\n" + "=" * 60)
print("CELL 9: Performance Tips for Kaggle/Colab")
print("=" * 60)

performance_tips = """
⚡ Performance Optimization:

1. Image Size:
   - Tiny (512x512): ~64 tokens, fastest
   - Small (640x640): ~100 tokens, good balance
   - Base (1024x1024): ~256 tokens, high quality
   - Gundam (dynamic): Adaptive, best for documents

2. GPU Memory Management:
   - Kaggle T4: 16GB VRAM
   - Recommended max_model_len: 8192
   - If OOM, reduce to 4096

3. Batch Processing:
   - Process multiple images in one call
   - Use list of inputs for better throughput

4. Model Loading:
   - First load: 2-5 minutes (downloads model)
   - Subsequent loads: 30-60 seconds (cached)

5. Output Length:
   - Adjust max_tokens based on expected output
   - Longer outputs = more time and memory

Example for batch processing:
```python
model_input = [
    {"prompt": prompt, "multi_modal_data": {"image": image1}},
    {"prompt": prompt, "multi_modal_data": {"image": image2}},
    {"prompt": prompt, "multi_modal_data": {"image": image3}},
]
outputs = llm.generate(model_input, sampling_params)
```
"""

print(performance_tips)

print("\n" + "=" * 60)
print("🎉 Setup Complete! You're ready to use DeepSeek-OCR!")
print("=" * 60)
print("\n📖 Next steps:")
print("1. Upload your images to Kaggle/Colab")
print("2. Copy the example code from CELL 7")
print("3. Modify the image path and prompt")
print("4. Run and enjoy! 🚀")
