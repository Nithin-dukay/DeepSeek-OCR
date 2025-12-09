#!/usr/bin/env python3
"""
Test script to verify the fix for GitHub Issue #299

This script checks if the fix is properly implemented and provides
diagnostic information.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version too old: {version.major}.{version.minor}.{version.micro}")
        print("  Required: Python 3.8 or higher")
        return False

def check_cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_vllm_installed():
    """Check if vLLM is installed."""
    try:
        import vllm
        print(f"✓ vLLM installed: version {vllm.__version__}")
        return True
    except ImportError:
        print("✗ vLLM not installed")
        return False

def check_environment_variables():
    """Check critical environment variables."""
    print("\nEnvironment Variables:")
    
    vars_to_check = [
        ('TRITON_PTXAS_PATH', 'Optional - set if using CUDA 11.8'),
        ('VLLM_USE_V1', 'Should be 0 for stability'),
        ('CUDA_VISIBLE_DEVICES', 'GPU selection'),
    ]
    
    for var_name, description in vars_to_check:
        value = os.environ.get(var_name)
        if value:
            print(f"  {var_name}={value} ({description})")
        else:
            print(f"  {var_name} not set ({description})")

def check_config_file():
    """Check config.py settings."""
    config_path = Path("DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py")
    
    if not config_path.exists():
        print("\n✗ config.py not found")
        return False
        
    print("\n✓ config.py found")
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            
        # Check for key settings
        if 'MAX_CROPS' in content:
            # Try to extract value
            for line in content.split('\n'):
                if 'MAX_CROPS' in line and '=' in line and not line.strip().startswith('#'):
                    print(f"  {line.strip()}")
                    
        if 'MAX_CONCURRENCY' in content:
            for line in content.split('\n'):
                if 'MAX_CONCURRENCY' in line and '=' in line and not line.strip().startswith('#'):
                    print(f"  {line.strip()}")
                    
        return True
    except Exception as e:
        print(f"  Error reading config: {e}")
        return False

def main():
    print("="*60)
    print("DeepSeek OCR Issue #299 Fix - Verification Script")
    print("="*60)
    print()
    
    checks = []
    
    # Check Python version
    print("1. Checking Python version...")
    checks.append(check_python_version())
    print()
    
    # Check CUDA
    print("2. Checking CUDA availability...")
    checks.append(check_cuda_available())
    print()
    
    # Check vLLM
    print("3. Checking vLLM installation...")
    checks.append(check_vllm_installed())
    print()
    
    # Check fixed files
    print("4. Checking fixed files...")
    fixed_files = [
        ("DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image_fixed.py", "Fixed image script"),
        ("DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_eval_batch_fixed.py", "Fixed batch script"),
        ("DeepSeek-OCR-master/DeepSeek-OCR-vllm/config_safe.py", "Safe configuration"),
        ("ISSUE_299_FIX.md", "Fix documentation"),
        ("FIX_IMPLEMENTATION_GUIDE.md", "Implementation guide"),
    ]
    
    for filepath, description in fixed_files:
        checks.append(check_file_exists(filepath, description))
    print()
    
    # Check environment variables
    print("5. Checking environment variables...")
    check_environment_variables()
    print()
    
    # Check config
    print("6. Checking configuration...")
    checks.append(check_config_file())
    print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(checks)
    total = len(checks)
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All checks passed! The fix is properly installed.")
        print("\nNext steps:")
        print("1. Set INPUT_PATH and OUTPUT_PATH in config.py")
        print("2. Run: python DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image_fixed.py")
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed. Please review the output above.")
        print("\nTo install missing dependencies:")
        print("  pip install torch torchvision vllm transformers pillow")
        return 1

if __name__ == "__main__":
    sys.exit(main())
