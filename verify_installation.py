#!/usr/bin/env python3
"""
DeepSeek-OCR Installation Verification Script
Checks if all required dependencies are correctly installed for CUDA 12.8
"""

import sys
import importlib
from typing import List, Tuple, Dict

# Color codes for terminal output
class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed and return version if available
    
    Args:
        package_name: Display name of the package
        import_name: Actual import name (if different from package_name)
    
    Returns:
        Tuple of (success: bool, version: str)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def check_cuda_availability() -> Dict[str, any]:
    """Check CUDA availability and GPU information"""
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        devices = []
        if cuda_available:
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                devices.append({
                    'id': i,
                    'name': device_name,
                    'memory_gb': device_memory
                })
        
        return {
            'available': cuda_available,
            'version': cuda_version,
            'device_count': device_count,
            'devices': devices
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }

def verify_vllm_functionality():
    """Test if vLLM can be imported and basic functionality works"""
    try:
        import vllm
        from vllm import LLM, SamplingParams
        print_success(f"vLLM imported successfully (version: {vllm.__version__})")
        print_success("vLLM core classes (LLM, SamplingParams) are accessible")
        return True
    except Exception as e:
        print_error(f"vLLM functionality check failed: {e}")
        return False

def main():
    """Main verification function"""
    print_header("DeepSeek-OCR Installation Verification")
    print_info("Checking installation for CUDA 12.8 compatibility...\n")
    
    # Track overall status
    all_checks_passed = True
    warnings = []
    
    # 1. Check Python version
    print_header("Python Environment")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_info(f"Python version: {python_version}")
    
    if sys.version_info.major == 3 and sys.version_info.minor == 12:
        print_success("Python 3.12 detected (recommended)")
    else:
        print_warning(f"Python {python_version} detected. Recommended: Python 3.12")
        warnings.append("Python version is not 3.12")
    
    # 2. Check core dependencies
    print_header("Core Dependencies")
    
    core_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("vllm", "vllm"),
        ("flash-attn", "flash_attn"),
        ("xformers", "xformers"),
    ]
    
    for package_name, import_name in core_packages:
        success, version = check_package(package_name, import_name)
        if success:
            print_success(f"{package_name}: {version}")
        else:
            print_error(f"{package_name}: NOT INSTALLED")
            all_checks_passed = False
    
    # 3. Check CUDA
    print_header("CUDA and GPU Information")
    
    cuda_info = check_cuda_availability()
    
    if cuda_info.get('available'):
        print_success(f"CUDA is available")
        print_info(f"CUDA version: {cuda_info['version']}")
        print_info(f"Number of GPUs: {cuda_info['device_count']}")
        
        for device in cuda_info['devices']:
            print_info(f"  GPU {device['id']}: {device['name']} ({device['memory_gb']:.2f} GB)")
        
        # Check if CUDA version matches expected
        if cuda_info['version'] and '12.8' in str(cuda_info['version']):
            print_success("CUDA 12.8 detected (expected version)")
        else:
            print_warning(f"CUDA version is {cuda_info['version']}, expected 12.8")
            warnings.append(f"CUDA version mismatch: {cuda_info['version']} vs 12.8")
    else:
        print_error("CUDA is NOT available")
        if 'error' in cuda_info:
            print_error(f"Error: {cuda_info['error']}")
        all_checks_passed = False
    
    # 4. Check vLLM specific dependencies
    print_header("vLLM Dependencies")
    
    vllm_deps = [
        ("pydantic", "pydantic"),
        ("transformers", "transformers"),
        ("cachetools", "cachetools"),
        ("cloudpickle", "cloudpickle"),
        ("psutil", "psutil"),
        ("zmq", "zmq"),
        ("msgspec", "msgspec"),
        ("blake3", "blake3"),
        ("hf_transfer", "hf_transfer"),
        ("prometheus_client", "prometheus_client"),
    ]
    
    for package_name, import_name in vllm_deps:
        success, version = check_package(package_name, import_name)
        if success:
            print_success(f"{package_name}: {version}")
        else:
            print_warning(f"{package_name}: NOT INSTALLED (may be optional)")
            warnings.append(f"{package_name} not installed")
    
    # 5. Check DeepSeek-OCR specific dependencies
    print_header("DeepSeek-OCR Dependencies")
    
    deepseek_deps = [
        ("einops", "einops"),
        ("easydict", "easydict"),
        ("addict", "addict"),
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("PyMuPDF", "fitz"),
        ("img2pdf", "img2pdf"),
    ]
    
    for package_name, import_name in deepseek_deps:
        success, version = check_package(package_name, import_name)
        if success:
            print_success(f"{package_name}: {version}")
        else:
            print_error(f"{package_name}: NOT INSTALLED")
            all_checks_passed = False
    
    # 6. Test vLLM functionality
    print_header("vLLM Functionality Test")
    
    if not verify_vllm_functionality():
        all_checks_passed = False
    
    # 7. Final summary
    print_header("Verification Summary")
    
    if all_checks_passed and len(warnings) == 0:
        print_success("✓ All checks passed! Installation is complete and ready to use.")
        print_info("\nNext steps:")
        print_info("  1. Configure settings in DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py")
        print_info("  2. Test with: cd DeepSeek-OCR-master/DeepSeek-OCR-vllm && python run_dpsk_ocr_image.py")
        return 0
    elif all_checks_passed and len(warnings) > 0:
        print_warning(f"✓ Core installation complete, but {len(warnings)} warning(s) detected:")
        for warning in warnings:
            print_warning(f"  - {warning}")
        print_info("\nThe installation should work, but you may want to address the warnings.")
        return 0
    else:
        print_error("✗ Installation verification FAILED!")
        print_error("\nSome required packages are missing or not working correctly.")
        print_info("\nTroubleshooting steps:")
        print_info("  1. Review the errors above")
        print_info("  2. Check INSTALL_CUDA_12.8.md for detailed installation instructions")
        print_info("  3. Run: ./install_cuda128.sh to attempt automatic installation")
        print_info("  4. For manual fixes, see the troubleshooting section in INSTALL_CUDA_12.8.md")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_warning("\n\nVerification interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"\n\nUnexpected error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
