#!/usr/bin/env python3
"""
DeepSeek-OCR Environment Checker

This script validates your Python environment for DeepSeek-OCR compatibility.
It checks package versions, CUDA availability, and identifies common issues.

Usage:
    python check_environment.py
"""

import sys
import importlib.util
from typing import Dict, List, Tuple, Optional
import subprocess


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.NC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.NC}")


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and 8 <= version.minor <= 12:
        return True, version_str
    else:
        return False, version_str


def check_package_version(package_name: str, required_version: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
    """
    Check if a package is installed and optionally verify its version
    
    Returns:
        (is_installed, installed_version, message)
    """
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if required_version:
            if version == required_version:
                return True, version, f"{package_name} {version} (matches required {required_version})"
            else:
                return False, version, f"{package_name} {version} (required: {required_version})"
        else:
            return True, version, f"{package_name} {version}"
    except ImportError:
        return False, None, f"{package_name} not installed"


def check_cuda() -> Tuple[bool, Dict[str, str]]:
    """Check CUDA availability and version"""
    info = {}
    
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = str(torch.cuda.is_available())
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda or 'unknown'
            info['cudnn_version'] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else 'N/A'
            info['gpu_count'] = str(torch.cuda.device_count())
            info['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'
            return True, info
        else:
            return False, info
    except ImportError:
        return False, {'error': 'PyTorch not installed'}


def check_vllm_compatibility() -> Tuple[str, List[str]]:
    """
    Determine which vLLM installation method is being used and check compatibility
    
    Returns:
        (installation_type, issues)
    """
    issues = []
    
    try:
        import vllm
        import transformers
        
        vllm_version = vllm.__version__
        transformers_version = transformers.__version__
        
        # Check if using vLLM 0.8.5
        if '0.8.5' in vllm_version:
            installation_type = "vLLM 0.8.5 (Local)"
            
            # For vLLM 0.8.5, transformers should be 4.46.3
            if transformers_version != '4.46.3':
                issues.append(
                    f"Transformers version mismatch: {transformers_version} (expected 4.46.3 for vLLM 0.8.5)\n"
                    f"  Fix: pip uninstall transformers -y && pip install transformers==4.46.3"
                )
        else:
            installation_type = "vLLM Nightly/Other"
            
            # For vLLM nightly, transformers should be >= 4.51.1
            from packaging import version
            if version.parse(transformers_version) < version.parse('4.51.1'):
                issues.append(
                    f"Transformers version may be too old: {transformers_version} (vLLM nightly typically requires >= 4.51.1)\n"
                    f"  Fix: pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly --force-reinstall"
                )
        
        return installation_type, issues
    except ImportError as e:
        return "Unknown", [f"Cannot determine vLLM compatibility: {str(e)}"]


def check_flash_attention() -> Tuple[bool, str]:
    """Check if flash-attn is installed"""
    try:
        import flash_attn
        version = getattr(flash_attn, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, "Not installed (optional)"


def main():
    """Main function to run all checks"""
    print_header("DeepSeek-OCR Environment Checker")
    
    all_checks_passed = True
    warnings = []
    
    # Check Python version
    print_info("Checking Python version...")
    python_ok, python_version = check_python_version()
    if python_ok:
        print_success(f"Python {python_version}")
    else:
        print_error(f"Python {python_version} (requires 3.8-3.12)")
        all_checks_passed = False
    
    # Check PyTorch
    print_info("\nChecking PyTorch installation...")
    torch_ok, torch_version, torch_msg = check_package_version('torch')
    if torch_ok:
        print_success(torch_msg)
    else:
        print_error(torch_msg)
        all_checks_passed = False
    
    # Check CUDA
    if torch_ok:
        print_info("\nChecking CUDA availability...")
        cuda_ok, cuda_info = check_cuda()
        if cuda_ok:
            print_success(f"CUDA available: {cuda_info['cuda_available']}")
            if cuda_info['cuda_available'] == 'True':
                print_success(f"CUDA version: {cuda_info['cuda_version']}")
                print_success(f"GPU: {cuda_info['gpu_name']}")
                print_success(f"GPU count: {cuda_info['gpu_count']}")
            else:
                print_warning("CUDA not available - will run on CPU (very slow)")
                warnings.append("CUDA not available")
        else:
            print_error("Cannot check CUDA")
    
    # Check transformers
    print_info("\nChecking transformers...")
    transformers_ok, transformers_version, transformers_msg = check_package_version('transformers')
    if transformers_ok:
        print_success(transformers_msg)
    else:
        print_error(transformers_msg)
        all_checks_passed = False
    
    # Check vLLM
    print_info("\nChecking vLLM...")
    vllm_ok, vllm_version, vllm_msg = check_package_version('vllm')
    if vllm_ok:
        print_success(vllm_msg)
    else:
        print_error(vllm_msg)
        all_checks_passed = False
    
    # Check vLLM compatibility
    if vllm_ok and transformers_ok:
        print_info("\nChecking vLLM compatibility...")
        installation_type, compatibility_issues = check_vllm_compatibility()
        print_info(f"Installation type: {installation_type}")
        
        if compatibility_issues:
            for issue in compatibility_issues:
                print_error(issue)
                all_checks_passed = False
        else:
            print_success("vLLM and transformers versions are compatible")
    
    # Check other required packages
    print_info("\nChecking other required packages...")
    required_packages = [
        'PIL',  # Pillow
        'einops',
        'easydict',
        'addict',
        'numpy',
    ]
    
    for package in required_packages:
        ok, version, msg = check_package_version(package)
        if ok:
            print_success(msg)
        else:
            print_error(msg)
            all_checks_passed = False
    
    # Check optional packages
    print_info("\nChecking optional packages...")
    flash_ok, flash_version = check_flash_attention()
    if flash_ok:
        print_success(f"flash-attn {flash_version}")
    else:
        print_warning(f"flash-attn {flash_version}")
        warnings.append("flash-attn not installed (optional but recommended)")
    
    # Summary
    print_header("Summary")
    
    if all_checks_passed and not warnings:
        print_success("All checks passed! Your environment is ready for DeepSeek-OCR.")
    elif all_checks_passed and warnings:
        print_warning("Environment is functional but has some warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print_error("Some checks failed. Please fix the issues above.")
        print_info("\nFor detailed installation instructions, see INSTALLATION.md")
        print_info("Common fixes:")
        print("  - For vLLM 0.8.5: pip install transformers==4.46.3")
        print("  - For vLLM nightly: pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly --force-reinstall")
        sys.exit(1)
    
    # Print quick start info
    print_header("Quick Start")
    print("To run DeepSeek-OCR:")
    print("\n1. Using vLLM:")
    print("   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm")
    print("   python run_dpsk_ocr_image.py")
    print("\n2. Using Transformers:")
    print("   cd DeepSeek-OCR-master/DeepSeek-OCR-hf")
    print("   python run_dpsk_ocr.py")
    print("\nFor more information, see INSTALLATION.md and README.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
