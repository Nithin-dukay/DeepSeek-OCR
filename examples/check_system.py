"""
System Check Script for DeepSeek-OCR on Windows 11
Run this script to verify your system configuration
"""

import sys
import platform

print("=" * 60)
print("DeepSeek-OCR System Check")
print("=" * 60)

# Python version
print(f"\nPython Version: {sys.version}")
print(f"Python Executable: {sys.executable}")

# Operating System
print(f"\nOperating System: {platform.system()} {platform.release()}")
print(f"Platform: {platform.platform()}")

# Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch installed: {torch.__version__}")
    
    # CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
        
        # GPU information
        num_gpus = torch.cuda.device_count()
        print(f"\nNumber of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Check if compute capability supports flash-attention
            compute_capability = props.major + props.minor / 10
            if compute_capability >= 7.5:
                print(f"  ✓ Flash-Attention 2.x supported (compute capability >= 7.5)")
            else:
                print(f"  ✗ Flash-Attention 2.x NOT supported (compute capability < 7.5)")
                print(f"    Recommendation: Use 'sdpa' or 'eager' attention mode")
        
        # Test CUDA operations
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            print(f"\n✓ CUDA operations working correctly")
        except Exception as e:
            print(f"\n✗ CUDA operations failed: {e}")
    else:
        print(f"✗ CUDA not available")
        print(f"  Recommendation: Use CPU mode or install CUDA toolkit")
    
    # Check bfloat16 support
    if torch.cuda.is_available():
        try:
            x = torch.randn(2, 2).cuda().to(torch.bfloat16)
            print(f"✓ bfloat16 supported on GPU")
        except:
            print(f"✗ bfloat16 NOT supported on GPU")
            print(f"  Recommendation: Use torch.float16 instead")
    
except ImportError:
    print("\n✗ PyTorch not installed")
    print("  Install with: pip install torch torchvision")

# Check Transformers
try:
    import transformers
    print(f"\n✓ Transformers installed: {transformers.__version__}")
except ImportError:
    print("\n✗ Transformers not installed")
    print("  Install with: pip install transformers")

# Check other dependencies
dependencies = [
    ('einops', 'einops'),
    ('PIL', 'Pillow'),
    ('numpy', 'numpy'),
    ('easydict', 'easydict'),
    ('addict', 'addict'),
]

print("\nOther Dependencies:")
for module_name, package_name in dependencies:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name}: {version}")
    except ImportError:
        print(f"  ✗ {package_name} not installed")

# Check flash-attn (optional)
try:
    import flash_attn
    print(f"\n✓ flash-attn installed: {flash_attn.__version__}")
    print("  Note: May not work on Windows or older GPUs")
except ImportError:
    print("\n✗ flash-attn not installed (this is OK for Windows)")
    print("  Recommendation: Use 'sdpa' or 'eager' attention mode instead")

# Recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if 'torch' in sys.modules:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        compute_capability = props.major + props.minor / 10
        vram_gb = props.total_memory / 1024**3
        
        print(f"\nFor your GPU ({torch.cuda.get_device_name(0)}):")
        
        if compute_capability < 7.5:
            print("  • Use attn_implementation='sdpa' (recommended)")
            print("  • Or use attn_implementation='eager' (most compatible)")
            print("  • Do NOT use flash_attention_2")
        else:
            print("  • You can use flash_attention_2 (if installed)")
            print("  • Or use attn_implementation='sdpa' (easier to install)")
        
        if vram_gb < 8:
            print(f"  • Limited VRAM ({vram_gb:.1f} GB detected)")
            print("  • Use Tiny mode: base_size=512, image_size=512")
            print("  • Or Small mode: base_size=640, image_size=640")
            print("  • Set crop_mode=False to save memory")
            print("  • Use torch.float16 instead of torch.bfloat16")
        elif vram_gb < 12:
            print(f"  • Moderate VRAM ({vram_gb:.1f} GB detected)")
            print("  • Use Small or Base mode")
        else:
            print(f"  • Good VRAM ({vram_gb:.1f} GB detected)")
            print("  • You can use Base or Large mode")
    else:
        print("\nNo GPU detected:")
        print("  • Use CPU mode (will be slow)")
        print("  • Use Tiny mode: base_size=512, image_size=512")
        print("  • Use torch.float32 for CPU")
        print("  • Consider using a GPU for better performance")

print("\n" + "=" * 60)
print("Example command to run:")
print("=" * 60)

if 'torch' in sys.modules and torch.cuda.is_available():
    print("\npython examples/run_windows_gpu_sdpa.py")
else:
    print("\npython examples/run_windows_cpu.py")

print("\n")
