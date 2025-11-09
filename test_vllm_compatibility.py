#!/usr/bin/env python3
"""
Test script to verify vLLM compatibility and diagnose Issue #110

This script checks your environment and vLLM installation to ensure
compatibility with DeepSeek-OCR.

Usage:
    python test_vllm_compatibility.py
"""

import sys
import os


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_status(check_name, status, message=""):
    """Print a status line"""
    status_symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{status_symbol} {check_name:.<50} {status_text}{reset}")
    if message:
        print(f"  → {message}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    is_compatible = version.major == 3 and version.minor >= 8
    message = f"Python {version.major}.{version.minor}.{version.micro}"
    return is_compatible, message


def check_torch():
    """Check PyTorch installation"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            message = f"v{version}, CUDA {cuda_version}, {gpu_name} ({gpu_memory:.1f}GB)"
        else:
            message = f"v{version}, CUDA NOT AVAILABLE"
        
        return cuda_available, message
    except ImportError:
        return False, "PyTorch not installed"


def check_vllm():
    """Check vLLM installation and version"""
    try:
        import vllm
        from packaging import version
        
        vllm_version = vllm.__version__
        parsed_version = version.parse(vllm_version.split('+')[0])
        
        # Check if it's a recommended version
        recommended = version.parse("0.8.5")
        
        if parsed_version == recommended:
            status = True
            message = f"v{vllm_version} (RECOMMENDED)"
        elif parsed_version < version.parse("0.6.0"):
            status = True
            message = f"v{vllm_version} (Compatible)"
        elif parsed_version < version.parse("0.9.0"):
            status = True
            message = f"v{vllm_version} (May have issues, see warnings)"
        else:
            status = False
            message = f"v{vllm_version} (NOT RECOMMENDED - use 0.8.5)"
        
        return status, message
    except ImportError:
        return False, "vLLM not installed"


def check_transformers():
    """Check Transformers installation"""
    try:
        import transformers
        version = transformers.__version__
        return True, f"v{version}"
    except ImportError:
        return False, "Transformers not installed"


def check_flash_attention():
    """Check Flash Attention installation"""
    try:
        import flash_attn
        return True, "Installed"
    except ImportError:
        return False, "Not installed (optional but recommended)"


def check_other_dependencies():
    """Check other required dependencies"""
    dependencies = {
        'PIL': 'Pillow',
        'einops': 'einops',
        'addict': 'addict',
        'fitz': 'PyMuPDF',
    }
    
    all_installed = True
    missing = []
    
    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            all_installed = False
            missing.append(package)
    
    if all_installed:
        return True, "All dependencies installed"
    else:
        return False, f"Missing: {', '.join(missing)}"


def check_vllm_environment():
    """Check vLLM environment variables"""
    vllm_use_v1 = os.environ.get('VLLM_USE_V1', 'not set')
    
    if vllm_use_v1 == '0':
        return True, "VLLM_USE_V1=0 (v0 architecture - recommended)"
    elif vllm_use_v1 == '1':
        return False, "VLLM_USE_V1=1 (v1 architecture - may cause issues)"
    else:
        return True, "VLLM_USE_V1 not set (will be auto-configured)"


def test_vllm_import():
    """Test if vLLM can be imported without errors"""
    try:
        from vllm import LLM, SamplingParams
        return True, "vLLM imports successfully"
    except Exception as e:
        return False, f"Import error: {str(e)[:50]}"


def provide_recommendations(results):
    """Provide recommendations based on test results"""
    print_header("RECOMMENDATIONS")
    
    has_issues = not all(results.values())
    
    if not has_issues:
        print("\n✓ Your environment looks good! You should be able to run DeepSeek-OCR.")
        print("\nTo get started:")
        print("  1. Update config.py with your image/PDF paths")
        print("  2. Run: cd DeepSeek-OCR-master/DeepSeek-OCR-vllm")
        print("  3. Run: python run_dpsk_ocr_image_fixed.py")
    else:
        print("\n⚠ Issues detected. Please follow these recommendations:\n")
        
        if not results.get('torch', True):
            print("1. Install PyTorch with CUDA support:")
            print("   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118")
            print()
        
        if not results.get('vllm', True):
            print("2. Install vLLM 0.8.5 (recommended):")
            print("   Download from: https://github.com/vllm-project/vllm/releases/tag/v0.8.5")
            print("   pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl")
            print()
        
        if not results.get('transformers', True):
            print("3. Install Transformers:")
            print("   pip install transformers==4.46.3")
            print()
        
        if not results.get('dependencies', True):
            print("4. Install other dependencies:")
            print("   pip install -r requirements.txt")
            print()
        
        if not results.get('flash_attn', True):
            print("5. Install Flash Attention (optional but recommended):")
            print("   pip install flash-attn==2.7.3 --no-build-isolation")
            print()
    
    print("\nFor more help, see:")
    print("  - TROUBLESHOOTING.md")
    print("  - ISSUE_110_FIX.md")
    print("  - GitHub Issues: https://github.com/deepseek-ai/DeepSeek-OCR/issues")


def main():
    """Main test function"""
    print_header("DeepSeek-OCR Compatibility Test")
    print("This script checks if your environment is properly configured")
    print("to run DeepSeek-OCR and avoid Issue #110 (Engine core crash)")
    
    print_header("SYSTEM CHECKS")
    
    results = {}
    
    # Python version
    status, message = check_python_version()
    print_status("Python Version", status, message)
    results['python'] = status
    
    # PyTorch
    status, message = check_torch()
    print_status("PyTorch + CUDA", status, message)
    results['torch'] = status
    
    # vLLM
    status, message = check_vllm()
    print_status("vLLM Version", status, message)
    results['vllm'] = status
    
    # Transformers
    status, message = check_transformers()
    print_status("Transformers", status, message)
    results['transformers'] = status
    
    # Flash Attention
    status, message = check_flash_attention()
    print_status("Flash Attention", status, message)
    results['flash_attn'] = status
    
    # Other dependencies
    status, message = check_other_dependencies()
    print_status("Other Dependencies", status, message)
    results['dependencies'] = status
    
    print_header("VLLM CONFIGURATION")
    
    # Environment variables
    status, message = check_vllm_environment()
    print_status("Environment Variables", status, message)
    results['env'] = status
    
    # vLLM import test
    status, message = test_vllm_import()
    print_status("vLLM Import Test", status, message)
    results['vllm_import'] = status
    
    # Recommendations
    provide_recommendations(results)
    
    print("\n" + "=" * 70 + "\n")
    
    # Exit code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
