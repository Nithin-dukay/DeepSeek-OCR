#!/usr/bin/env python3
"""
DeepSeek-OCR Installation Verification Script
This script checks if vLLM and all required dependencies are properly installed.
"""

import sys
import importlib
import subprocess
from typing import List, Tuple

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def print_success(msg: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def check_module(module_name: str) -> Tuple[bool, str]:
    """Check if a Python module can be imported."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def check_cuda_availability():
    """Check if CUDA is available through PyTorch."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            return True, {
                'cuda_version': cuda_version,
                'device_count': device_count,
                'device_name': device_name
            }
        else:
            return False, "CUDA not available"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("DeepSeek-OCR Installation Verification")
    print("=" * 60)
    print()

    # Core dependencies to check
    core_modules = [
        'torch',
        'torchvision',
        'xformers',
        'vllm',
        'flash_attn',
        'transformers',
        'pydantic',
    ]

    # Additional dependencies
    additional_modules = [
        'cachetools',
        'cloudpickle',
        'psutil',
        'zmq',
        'msgspec',
        'blake3',
        'hf_transfer',
        'prometheus_client',
        'PIL',  # Pillow
        'numpy',
        'einops',
    ]

    all_modules = core_modules + additional_modules
    
    print_info("Checking core dependencies...")
    print()
    
    missing_modules = []
    installed_modules = []
    
    for module in core_modules:
        success, info = check_module(module)
        if success:
            print_success(f"✓ {module:20s} version: {info}")
            installed_modules.append(module)
        else:
            print_error(f"✗ {module:20s} NOT FOUND")
            missing_modules.append(module)
    
    print()
    print_info("Checking additional dependencies...")
    print()
    
    for module in additional_modules:
        success, info = check_module(module)
        if success:
            print_success(f"✓ {module:20s} version: {info}")
            installed_modules.append(module)
        else:
            print_warning(f"✗ {module:20s} NOT FOUND")
            missing_modules.append(module)
    
    print()
    print_info("Checking CUDA availability...")
    print()
    
    cuda_success, cuda_info = check_cuda_availability()
    if cuda_success:
        print_success(f"✓ CUDA is available")
        print(f"  - CUDA Version: {cuda_info['cuda_version']}")
        print(f"  - Device Count: {cuda_info['device_count']}")
        print(f"  - Device Name: {cuda_info['device_name']}")
    else:
        print_error(f"✗ CUDA not available: {cuda_info}")
    
    print()
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print()
    
    print(f"Installed modules: {len(installed_modules)}/{len(all_modules)}")
    print(f"Missing modules: {len(missing_modules)}")
    
    if missing_modules:
        print()
        print_warning("Missing modules detected:")
        for module in missing_modules:
            print(f"  - {module}")
        print()
        print_info("To install missing modules, run:")
        for module in missing_modules:
            # Handle special cases
            if module == 'PIL':
                print(f"  pip install Pillow")
            elif module == 'zmq':
                print(f"  pip install pyzmq")
            else:
                print(f"  pip install {module}")
    
    print()
    
    # Try to import and test vLLM
    print_info("Testing vLLM import...")
    try:
        import vllm
        print_success(f"✓ vLLM successfully imported (version: {vllm.__version__})")
        
        # Try to check vLLM's internal dependencies
        print()
        print_info("Checking vLLM internal components...")
        try:
            from vllm import LLM, SamplingParams
            print_success("✓ vLLM core components accessible")
        except Exception as e:
            print_error(f"✗ vLLM core components error: {e}")
        
        try:
            from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
            print_success("✓ DeepSeek-OCR model components accessible")
        except Exception as e:
            print_warning(f"⚠ DeepSeek-OCR model components not accessible: {e}")
            print_info("  This is normal if you haven't downloaded the model yet")
        
    except ImportError as e:
        print_error(f"✗ vLLM import failed: {e}")
        print()
        print_info("This usually means:")
        print("  1. vLLM is not installed")
        print("  2. Some dependencies are missing")
        print("  3. There's a version conflict")
        print()
        print_info("Try running the setup script again:")
        print("  bash scripts/setup_rtx5090_cuda128.sh")
    
    print()
    print("=" * 60)
    
    if not missing_modules and cuda_success:
        print_success("✓ All checks passed! Your installation looks good.")
        print()
        print_info("Next steps:")
        print("  1. Configure paths in: DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py")
        print("  2. Run DeepSeek-OCR:")
        print("     cd DeepSeek-OCR-master/DeepSeek-OCR-vllm")
        print("     python run_dpsk_ocr_image.py")
        return 0
    else:
        print_warning("⚠ Some issues detected. Please review the output above.")
        print()
        print_info("For troubleshooting, see: docs/INSTALL_RTX5090_CUDA128.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
