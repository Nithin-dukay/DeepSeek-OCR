#!/usr/bin/env python3
"""
Test script to verify the DeepSeek OCR fix for Issue #299

This script checks if the necessary fixes are in place and provides
a diagnostic report.

Usage:
    python test_fix.py [--path PATH]
"""

import os
import sys
import argparse
from pathlib import Path


def check_file_exists(file_path, description):
    """Check if a file exists."""
    exists = os.path.exists(file_path)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {file_path}")
    return exists


def check_file_content(file_path, search_strings, description):
    """Check if file contains specific strings."""
    if not os.path.exists(file_path):
        print(f"  ✗ {description}: File not found")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    all_found = all(s in content for s in search_strings)
    status = "✓" if all_found else "✗"
    
    if all_found:
        print(f"  {status} {description}: All checks passed")
    else:
        print(f"  {status} {description}: Missing required changes")
        for s in search_strings:
            if s not in content:
                print(f"      Missing: {s[:50]}...")
    
    return all_found


def check_environment():
    """Check Python environment."""
    print("\n" + "=" * 70)
    print("Environment Check")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 0
    
    # Check Python version
    total_checks += 1
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"  ✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        checks_passed += 1
    else:
        print(f"  ✗ Python version: {python_version.major}.{python_version.minor}.{python_version.micro} (need >= 3.8)")
    
    # Check for required packages
    required_packages = ['torch', 'vllm', 'transformers', 'PIL']
    
    for package in required_packages:
        total_checks += 1
        try:
            __import__(package)
            print(f"  ✓ Package installed: {package}")
            checks_passed += 1
        except ImportError:
            print(f"  ✗ Package missing: {package}")
    
    # Check CUDA availability
    total_checks += 1
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"      CUDA version: {torch.version.cuda}")
            checks_passed += 1
        else:
            print(f"  ✗ CUDA not available")
    except:
        print(f"  ✗ Cannot check CUDA")
    
    return checks_passed, total_checks


def check_fixed_files(base_path):
    """Check if fixed files exist."""
    print("\n" + "=" * 70)
    print("Fixed Files Check")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 3
    
    files = [
        (base_path / "run_dpsk_ocr_image_fixed.py", "Fixed run script"),
        (base_path / "deepseek_ocr_fixed.py", "Fixed model file"),
        (base_path.parent.parent / "FIX_README.md", "Fix documentation"),
    ]
    
    for file_path, description in files:
        if check_file_exists(file_path, description):
            checks_passed += 1
    
    return checks_passed, total_checks


def check_original_files_patched(base_path):
    """Check if original files have been patched."""
    print("\n" + "=" * 70)
    print("Original Files Patch Check")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 2
    
    # Check run script
    run_script = base_path / "run_dpsk_ocr_image.py"
    if check_file_content(
        run_script,
        ["enforce_eager=True", "block_size=128", "max_num_seqs=255"],
        "Run script patched"
    ):
        checks_passed += 1
    
    # Check model file
    model_file = base_path / "deepseek_ocr.py"
    if check_file_content(
        model_file,
        ["torch.cuda.synchronize()"],
        "Model file patched"
    ):
        checks_passed += 1
    
    return checks_passed, total_checks


def check_config(base_path):
    """Check configuration file."""
    print("\n" + "=" * 70)
    print("Configuration Check")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 3
    
    config_file = base_path / "config.py"
    
    if not os.path.exists(config_file):
        print(f"  ✗ Config file not found: {config_file}")
        return 0, total_checks
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check MODEL_PATH
    total_checks += 1
    if "MODEL_PATH" in content:
        print(f"  ✓ MODEL_PATH configured")
        checks_passed += 1
    else:
        print(f"  ✗ MODEL_PATH not configured")
    
    # Check INPUT_PATH
    if "INPUT_PATH" in content:
        print(f"  ✓ INPUT_PATH configured")
        checks_passed += 1
    else:
        print(f"  ✗ INPUT_PATH not configured")
    
    # Check OUTPUT_PATH
    if "OUTPUT_PATH" in content:
        print(f"  ✓ OUTPUT_PATH configured")
        checks_passed += 1
    else:
        print(f"  ✗ OUTPUT_PATH not configured")
    
    return checks_passed, total_checks


def main():
    parser = argparse.ArgumentParser(
        description="Test DeepSeek OCR fix for Issue #299"
    )
    parser.add_argument(
        '--path',
        type=str,
        default='DeepSeek-OCR-master/DeepSeek-OCR-vllm',
        help='Path to DeepSeek-OCR-vllm directory'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DeepSeek OCR Fix Verification")
    print("GitHub Issue #299: Triton CUDA Illegal Memory Access")
    print("=" * 70)
    
    base_path = Path(args.path)
    
    if not base_path.exists():
        print(f"\n✗ Directory not found: {base_path}")
        print("Please specify the correct path using --path option")
        return 1
    
    print(f"\nChecking directory: {base_path.absolute()}")
    
    # Run all checks
    total_passed = 0
    total_checks = 0
    
    # Environment check
    passed, total = check_environment()
    total_passed += passed
    total_checks += total
    
    # Fixed files check
    passed, total = check_fixed_files(base_path)
    total_passed += passed
    total_checks += total
    
    # Original files patch check
    passed, total = check_original_files_patched(base_path)
    total_passed += passed
    total_checks += total
    
    # Config check
    passed, total = check_config(base_path)
    total_passed += passed
    total_checks += total
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    percentage = (total_passed / total_checks * 100) if total_checks > 0 else 0
    print(f"\nChecks passed: {total_passed}/{total_checks} ({percentage:.1f}%)")
    
    if percentage == 100:
        print("\n✓ All checks passed! The fix is properly installed.")
        print("\nYou can now run:")
        print("  python run_dpsk_ocr_image_fixed.py")
        print("or")
        print("  python run_dpsk_ocr_image.py (if patched)")
        return 0
    elif percentage >= 80:
        print("\n⚠ Most checks passed, but some issues detected.")
        print("The fix should work, but review the warnings above.")
        return 0
    elif percentage >= 50:
        print("\n⚠ Some checks failed.")
        print("\nRecommended actions:")
        print("1. Run: python apply_fix.py --backup")
        print("2. Or use the fixed files directly:")
        print("   python run_dpsk_ocr_image_fixed.py")
        return 1
    else:
        print("\n✗ Many checks failed. Fix not properly installed.")
        print("\nRecommended actions:")
        print("1. Ensure you're in the correct directory")
        print("2. Run: python apply_fix.py --backup")
        print("3. Or copy the fixed files:")
        print("   cp run_dpsk_ocr_image_fixed.py run_dpsk_ocr_image.py")
        print("   cp deepseek_ocr_fixed.py deepseek_ocr.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
