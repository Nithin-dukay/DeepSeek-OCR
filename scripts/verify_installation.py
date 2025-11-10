#!/usr/bin/env python3
"""
DeepSeek-OCR Installation Verification Script
Verifies that all required components are properly installed for CUDA 12.8 / RTX 5090
"""

import sys
import subprocess
from typing import Tuple, List, Dict

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'

def print_header():
    """Print verification header"""
    print("=" * 60)
    print("  DeepSeek-OCR Installation Verification")
    print("  CUDA 12.8 / RTX 5090 Configuration")
    print("=" * 60)
    print()

def print_section(title: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}[{title}]{Colors.NC}")
    print("-" * 60)

def check_result(success: bool, message: str, details: str = ""):
    """Print check result with color coding"""
    if success:
        print(f"{Colors.GREEN}✓{Colors.NC} {message}")
        if details:
            print(f"  {Colors.BLUE}→{Colors.NC} {details}")
    else:
        print(f"{Colors.RED}✗{Colors.NC} {message}")
        if details:
            print(f"  {Colors.YELLOW}→{Colors.NC} {details}")

def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor == 12:
        return True, f"Python {version_str} (Recommended)"
    elif version.major == 3 and version.minor >= 10:
        return True, f"Python {version_str} (Compatible, but 3.12 recommended)"
    else:
        return False, f"Python {version_str} (Incompatible - requires 3.10+)"

def check_module_import(module_name: str, display_name: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported"""
    if display_name is None:
        display_name = module_name
    
    try:
        __import__(module_name)
        return True, f"{display_name} is installed"
    except ImportError as e:
        return False, f"{display_name} not found: {str(e)}"

def check_torch() -> Tuple[bool, str, Dict]:
    """Check PyTorch installation and CUDA support"""
    try:
        import torch
        
        info = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A',
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name(0)
            info['device_capability'] = torch.cuda.get_device_capability(0)
        
        if not info['cuda_available']:
            return False, "PyTorch installed but CUDA not available", info
        
        if info['cuda_version'] and not info['cuda_version'].startswith('12.'):
            return False, f"CUDA version mismatch (expected 12.x, got {info['cuda_version']})", info
        
        return True, "PyTorch with CUDA support", info
        
    except ImportError:
        return False, "PyTorch not installed", {}

def check_vllm() -> Tuple[bool, str, str]:
    """Check vLLM installation"""
    try:
        import vllm
        version = vllm.__version__
        
        # Try to import key components
        try:
            from vllm import LLM, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            return True, f"vLLM {version} with all components", version
        except ImportError as e:
            return False, f"vLLM {version} installed but missing components: {str(e)}", version
            
    except ImportError as e:
        return False, f"vLLM not installed: {str(e)}", ""

def check_flash_attn() -> Tuple[bool, str]:
    """Check flash-attn installation"""
    try:
        import flash_attn
        version = getattr(flash_attn, '__version__', 'unknown')
        return True, f"flash-attn {version}"
    except ImportError as e:
        return False, f"flash-attn not found: {str(e)}"

def check_xformers() -> Tuple[bool, str]:
    """Check xformers installation"""
    try:
        import xformers
        version = getattr(xformers, '__version__', 'unknown')
        return True, f"xformers {version}"
    except ImportError as e:
        return False, f"xformers not found: {str(e)}"

def check_transformers() -> Tuple[bool, str]:
    """Check transformers installation"""
    try:
        import transformers
        version = transformers.__version__
        return True, f"transformers {version}"
    except ImportError as e:
        return False, f"transformers not found: {str(e)}"

def check_cuda_system() -> Tuple[bool, str, Dict]:
    """Check system CUDA installation"""
    info = {}
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse CUDA version from nvcc output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    info['nvcc_version'] = line.strip()
                    break
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info['nvcc_version'] = 'Not found'
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                               '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info['gpu_info'] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        info['gpu_info'] = 'nvidia-smi not found'
    
    has_cuda = 'nvcc_version' in info and info['nvcc_version'] != 'Not found'
    has_gpu = 'gpu_info' in info and info['gpu_info'] != 'nvidia-smi not found'
    
    if has_cuda and has_gpu:
        return True, "CUDA toolkit and GPU detected", info
    elif has_gpu:
        return True, "GPU detected (nvcc not found)", info
    else:
        return False, "CUDA/GPU not properly detected", info

def check_critical_dependencies() -> List[Tuple[str, bool, str]]:
    """Check critical Python dependencies"""
    dependencies = [
        ('pydantic', 'Pydantic'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('einops', 'Einops'),
        ('fastapi', 'FastAPI'),
        ('ray', 'Ray'),
    ]
    
    results = []
    for module, display_name in dependencies:
        success, message = check_module_import(module, display_name)
        results.append((display_name, success, message))
    
    return results

def main():
    """Main verification function"""
    print_header()
    
    all_checks_passed = True
    warnings = []
    
    # Check Python version
    print_section("Python Environment")
    success, message = check_python_version()
    check_result(success, "Python Version", message)
    if not success:
        all_checks_passed = False
    elif "Compatible" in message:
        warnings.append("Python 3.12.x is recommended for best compatibility")
    
    # Check system CUDA
    print_section("System CUDA Configuration")
    success, message, info = check_cuda_system()
    check_result(success, "CUDA Toolkit", message)
    if info.get('nvcc_version'):
        print(f"  {Colors.BLUE}→{Colors.NC} {info['nvcc_version']}")
    if info.get('gpu_info'):
        print(f"  {Colors.BLUE}→{Colors.NC} {info['gpu_info']}")
    if not success:
        warnings.append("CUDA toolkit not detected - GPU acceleration may not work")
    
    # Check PyTorch
    print_section("PyTorch Installation")
    success, message, info = check_torch()
    check_result(success, "PyTorch", message)
    if info:
        print(f"  {Colors.BLUE}→{Colors.NC} Version: {info.get('version', 'unknown')}")
        print(f"  {Colors.BLUE}→{Colors.NC} CUDA Available: {info.get('cuda_available', False)}")
        print(f"  {Colors.BLUE}→{Colors.NC} CUDA Version: {info.get('cuda_version', 'N/A')}")
        if info.get('cuda_available'):
            print(f"  {Colors.BLUE}→{Colors.NC} GPU: {info.get('device_name', 'unknown')}")
            print(f"  {Colors.BLUE}→{Colors.NC} Compute Capability: {info.get('device_capability', 'unknown')}")
    if not success:
        all_checks_passed = False
    
    # Check vLLM
    print_section("vLLM Installation")
    success, message, version = check_vllm()
    check_result(success, "vLLM", message)
    if not success:
        all_checks_passed = False
    
    # Check flash-attn
    print_section("Attention Mechanisms")
    success, message = check_flash_attn()
    check_result(success, "Flash Attention", message)
    if not success:
        all_checks_passed = False
    
    success, message = check_xformers()
    check_result(success, "xformers", message)
    if not success:
        all_checks_passed = False
    
    # Check transformers
    print_section("Model Libraries")
    success, message = check_transformers()
    check_result(success, "Transformers", message)
    if not success:
        all_checks_passed = False
    
    # Check critical dependencies
    print_section("Critical Dependencies")
    dep_results = check_critical_dependencies()
    for name, success, message in dep_results:
        check_result(success, name, message if not success else "")
        if not success:
            warnings.append(f"{name} is missing")
    
    # Print summary
    print_section("Verification Summary")
    print()
    
    if all_checks_passed and not warnings:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed!{Colors.NC}")
        print(f"\n{Colors.GREEN}Your installation is ready to use.{Colors.NC}")
        print("\nNext steps:")
        print("  1. Configure paths in DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py")
        print("  2. Test with: cd DeepSeek-OCR-master/DeepSeek-OCR-vllm && python run_dpsk_ocr_image.py")
        return 0
    elif warnings and all_checks_passed:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Installation complete with warnings:{Colors.NC}")
        for warning in warnings:
            print(f"  {Colors.YELLOW}•{Colors.NC} {warning}")
        print(f"\n{Colors.YELLOW}Your installation should work, but consider addressing the warnings.{Colors.NC}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Installation verification failed!{Colors.NC}")
        print(f"\n{Colors.RED}Please fix the errors above before proceeding.{Colors.NC}")
        print("\nTroubleshooting:")
        print("  1. Check docs/INSTALL_CUDA_12.8.md for detailed installation steps")
        print("  2. Review docs/TROUBLESHOOTING.md for common issues")
        print("  3. Ensure you followed the installation order exactly")
        print("  4. Try re-running: bash scripts/install_cuda128.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
