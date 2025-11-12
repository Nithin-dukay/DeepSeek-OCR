#!/usr/bin/env python3
"""
DeepSeek-OCR Installation Verification Script

This script verifies that all required packages for DeepSeek-OCR with vLLM
are correctly installed and compatible with each other.

Usage: python scripts/verify_installation.py
"""

import sys
import importlib
from typing import Dict, List, Tuple, Optional

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'


def print_header():
    """Print script header"""
    print(f"\n{Colors.BLUE}{'=' * 70}{Colors.NC}")
    print(f"{Colors.BOLD}DeepSeek-OCR Installation Verification{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 70}{Colors.NC}\n")


def print_section(title: str):
    """Print section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.NC}")
    print(f"{Colors.CYAN}{'-' * len(title)}{Colors.NC}")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.NC} {message}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.NC} {message}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.NC} {message}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.NC} {message}")


def check_python_version() -> bool:
    """Check Python version"""
    print_section("Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_info(f"Python version: {version_str}")
    
    if version.major == 3 and version.minor == 12:
        print_success("Python 3.12 detected (recommended)")
        return True
    elif version.major == 3 and version.minor >= 10:
        print_warning(f"Python {version_str} detected. Python 3.12 is recommended.")
        return True
    else:
        print_error(f"Python {version_str} detected. Python 3.10+ is required.")
        return False


def check_package(package_name: str, import_name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Check if a package is installed and return its version
    
    Args:
        package_name: Name of the package to check
        import_name: Name to use for import (if different from package_name)
    
    Returns:
        Tuple of (success, version)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def check_core_packages() -> Dict[str, bool]:
    """Check core ML packages"""
    print_section("Core ML Packages")
    
    results = {}
    
    # PyTorch
    success, version = check_package('torch')
    if success:
        print_success(f"PyTorch: {version}")
        results['torch'] = True
        
        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
                print_info(f"  CUDA available: Yes (version {cuda_version})")
                print_info(f"  GPU devices: {device_count}")
                if device_count > 0:
                    print_info(f"  Device 0: {device_name}")
            else:
                print_warning("  CUDA not available")
        except Exception as e:
            print_warning(f"  Could not check CUDA: {e}")
    else:
        print_error(f"PyTorch: Not installed ({version})")
        results['torch'] = False
    
    # Torchvision
    success, version = check_package('torchvision')
    if success:
        print_success(f"Torchvision: {version}")
        results['torchvision'] = True
    else:
        print_error(f"Torchvision: Not installed ({version})")
        results['torchvision'] = False
    
    # xformers
    success, version = check_package('xformers')
    if success:
        print_success(f"xformers: {version}")
        results['xformers'] = True
    else:
        print_error(f"xformers: Not installed ({version})")
        results['xformers'] = False
    
    # flash_attn
    success, version = check_package('flash_attn')
    if success:
        print_success(f"flash_attn: {version}")
        results['flash_attn'] = True
    else:
        print_warning(f"flash_attn: Not installed ({version})")
        results['flash_attn'] = False
    
    return results


def check_vllm() -> bool:
    """Check vLLM installation"""
    print_section("vLLM")
    
    success, version = check_package('vllm')
    if success:
        print_success(f"vLLM: {version}")
        
        # Try to import key vLLM components
        try:
            from vllm import LLM, SamplingParams
            print_success("  Core components importable")
            
            from vllm.engine.arg_utils import AsyncEngineArgs
            print_success("  Engine components importable")
            
            from vllm.model_executor.models.registry import ModelRegistry
            print_success("  Model registry importable")
            
            return True
        except ImportError as e:
            print_error(f"  Failed to import vLLM components: {e}")
            return False
    else:
        print_error(f"vLLM: Not installed ({version})")
        return False


def check_transformers() -> bool:
    """Check transformers installation"""
    print_section("Transformers")
    
    success, version = check_package('transformers')
    if success:
        print_success(f"Transformers: {version}")
        
        # Try to import key components
        try:
            from transformers import AutoTokenizer, AutoModel
            print_success("  Core components importable")
            return True
        except ImportError as e:
            print_error(f"  Failed to import transformers components: {e}")
            return False
    else:
        print_error(f"Transformers: Not installed ({version})")
        return False


def check_dependencies() -> Dict[str, bool]:
    """Check other required dependencies"""
    print_section("Other Dependencies")
    
    dependencies = [
        'pydantic',
        'PIL',  # Pillow
        'numpy',
        'einops',
        'easydict',
        'addict',
        'fitz',  # PyMuPDF
        'img2pdf',
        'tqdm',
        'requests',
    ]
    
    results = {}
    for dep in dependencies:
        success, version = check_package(dep)
        if success:
            print_success(f"{dep}: {version}")
            results[dep] = True
        else:
            print_warning(f"{dep}: Not installed")
            results[dep] = False
    
    return results


def check_cuda_compatibility() -> bool:
    """Check CUDA compatibility between packages"""
    print_section("CUDA Compatibility Check")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print_warning("CUDA not available in PyTorch")
            return False
        
        torch_cuda = torch.version.cuda
        print_info(f"PyTorch CUDA version: {torch_cuda}")
        
        # Check if versions match expected CUDA 12.8
        if torch_cuda and '12.8' in torch_cuda:
            print_success("CUDA 12.8 detected (expected)")
        elif torch_cuda:
            print_warning(f"CUDA {torch_cuda} detected. CUDA 12.8 is recommended for RTX 5090.")
        
        # Try a simple CUDA operation
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = x * 2
            print_success("CUDA operations working")
            return True
        except Exception as e:
            print_error(f"CUDA operation failed: {e}")
            return False
            
    except Exception as e:
        print_error(f"Could not check CUDA compatibility: {e}")
        return False


def check_deepseek_ocr() -> bool:
    """Check if DeepSeek-OCR specific files are present"""
    print_section("DeepSeek-OCR Files")
    
    import os
    
    files_to_check = [
        'requirements.txt',
        'DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py',
        'DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py',
        'DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py',
    ]
    
    all_present = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print_success(f"{file_path}")
        else:
            print_warning(f"{file_path} not found")
            all_present = False
    
    return all_present


def print_summary(results: Dict[str, bool]):
    """Print summary of verification results"""
    print_section("Summary")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"\nTotal checks: {total}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.NC}")
    print(f"{Colors.RED}Failed: {failed}{Colors.NC}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All checks passed! Installation is complete.{Colors.NC}")
        print(f"\n{Colors.CYAN}Next steps:{Colors.NC}")
        print("  1. Configure paths in DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py")
        print("  2. Run: cd DeepSeek-OCR-master/DeepSeek-OCR-vllm")
        print("  3. Test: python run_dpsk_ocr_image.py")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Some checks failed. Please review the errors above.{Colors.NC}")
        print(f"\n{Colors.CYAN}Troubleshooting:{Colors.NC}")
        print("  - See docs/INSTALL_CUDA_12.8.md for detailed installation instructions")
        print("  - Run: bash scripts/install_cuda128_rtx5090.sh for automated installation")
        return False


def main():
    """Main verification function"""
    print_header()
    
    results = {}
    
    # Check Python version
    results['python'] = check_python_version()
    
    # Check core packages
    core_results = check_core_packages()
    results.update(core_results)
    
    # Check vLLM
    results['vllm'] = check_vllm()
    
    # Check transformers
    results['transformers'] = check_transformers()
    
    # Check dependencies
    dep_results = check_dependencies()
    # Don't fail on missing optional dependencies
    
    # Check CUDA compatibility
    results['cuda_compat'] = check_cuda_compatibility()
    
    # Check DeepSeek-OCR files
    deepseek_present = check_deepseek_ocr()
    
    # Print summary
    success = print_summary(results)
    
    print(f"\n{Colors.BLUE}{'=' * 70}{Colors.NC}\n")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
