#!/usr/bin/env python3
"""
Test script for GitHub Issue #299 fix

This script helps verify that the Triton/CUDA illegal memory access fix is working.
It performs basic checks and provides diagnostic information.

Usage:
    python test_fix_issue_299.py
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_environment():
    """Check Python and CUDA environment"""
    print_header("Environment Check")
    
    # Python version
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Check if torch is available
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("⚠ PyTorch not installed")
        return False
    
    # Check if vLLM is available
    try:
        import vllm
        print(f"✓ vLLM version: {vllm.__version__}")
    except ImportError:
        print("⚠ vLLM not installed")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    print_header("File Check")
    
    base_dir = Path(__file__).parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return False
    
    required_files = [
        "config.py",
        "deepseek_ocr.py",
        "run_dpsk_ocr_image.py",
    ]
    
    fixed_files = [
        "run_dpsk_ocr_image_fixed.py",
        "config_fixed.py",
    ]
    
    all_good = True
    
    print("\nRequired files:")
    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ❌ {filename} (missing)")
            all_good = False
    
    print("\nFixed files:")
    for filename in fixed_files:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ⚠ {filename} (not created yet)")
    
    return all_good

def check_configuration():
    """Check if configuration has been applied"""
    print_header("Configuration Check")
    
    base_dir = Path(__file__).parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
    
    # Check run_dpsk_ocr_image.py
    image_file = base_dir / "run_dpsk_ocr_image.py"
    if image_file.exists():
        with open(image_file, 'r') as f:
            content = f.read()
        
        checks = {
            "VLLM_USE_V1 = '0'": "V1 engine disabled",
            "enforce_eager=True": "CUDA graphs disabled (enforce_eager)",
            "gpu_memory_utilization=0.75": "GPU memory reduced to 0.75",
            "max_num_seqs=255": "max_num_seqs set to 255 (not 256)",
        }
        
        print("\nrun_dpsk_ocr_image.py:")
        for check, description in checks.items():
            if check in content:
                print(f"  ✓ {description}")
            else:
                print(f"  ❌ {description} (not applied)")
    
    # Check config.py
    config_file = base_dir / "config.py"
    if config_file.exists():
        with open(config_file, 'r') as f:
            content = f.read()
        
        print("\nconfig.py:")
        
        # Check MAX_CROPS
        if "MAX_CROPS= 4" in content or "MAX_CROPS = 4" in content:
            print("  ✓ MAX_CROPS reduced to 4")
        elif "MAX_CROPS= 6" in content or "MAX_CROPS = 6" in content:
            print("  ⚠ MAX_CROPS still at 6 (consider reducing to 4)")
        
        # Check MAX_CONCURRENCY
        if "MAX_CONCURRENCY = 50" in content or "MAX_CONCURRENCY= 50" in content:
            print("  ✓ MAX_CONCURRENCY reduced to 50")
        elif "MAX_CONCURRENCY = 100" in content or "MAX_CONCURRENCY= 100" in content:
            print("  ⚠ MAX_CONCURRENCY still at 100 (consider reducing to 50)")

def check_gpu_memory():
    """Check GPU memory availability"""
    print_header("GPU Memory Check")
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("\nGPU Memory Status:")
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
        
        # Parse memory usage
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) >= 4:
                total = int(parts[1].split()[0])
                free = int(parts[2].split()[0])
                used = int(parts[3].split()[0])
                usage_percent = (used / total) * 100
                
                print(f"\n  Memory usage: {usage_percent:.1f}%")
                if usage_percent > 80:
                    print("  ⚠ High GPU memory usage detected")
                    print("    Consider reducing MAX_CROPS or closing other GPU processes")
                else:
                    print("  ✓ GPU memory usage is acceptable")
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ nvidia-smi not available or failed")
        return False

def print_recommendations():
    """Print recommendations based on checks"""
    print_header("Recommendations")
    
    print("""
1. If you haven't applied the fix yet:
   - Run: python apply_fix_issue_299.py --dry-run
   - Review the changes
   - Run: python apply_fix_issue_299.py (to apply)

2. If the fix is applied but issues persist:
   - Further reduce MAX_CROPS in config.py (try 3 or 2)
   - Reduce MAX_CONCURRENCY to 25
   - Try the piecewise CUDA graph mode (run_dpsk_ocr_image_piecewise.py)

3. For testing:
   - Start with a single problematic image
   - Monitor GPU memory with: watch -n 1 nvidia-smi
   - Check logs for any remaining CUDA errors

4. Alternative approaches:
   - Use vLLM 0.8.5 (mentioned in README as stable)
   - Fall back to HuggingFace transformers for problematic images
   - Try SGLang as an alternative inference engine

5. Performance tuning:
   - If enforce_eager=True is too slow, try piecewise mode
   - Adjust gpu_memory_utilization between 0.7-0.8
   - Balance MAX_CROPS vs processing speed

See ISSUE_299_FIX.md for detailed documentation.
""")

def main():
    print("=" * 70)
    print("  DeepSeek OCR Issue #299 Fix - Test & Diagnostic Script")
    print("=" * 70)
    
    # Run checks
    env_ok = check_environment()
    files_ok = check_files()
    
    if env_ok and files_ok:
        check_configuration()
        check_gpu_memory()
    
    print_recommendations()
    
    print("\n" + "=" * 70)
    print("  Diagnostic Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
