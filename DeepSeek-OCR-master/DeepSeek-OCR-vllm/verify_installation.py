#!/usr/bin/env python3
"""
Simple verification script to check if all files are properly installed.
This script doesn't require torch or other heavy dependencies.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("DeepSeek OCR Installation Verification")
    print("="*60 + "\n")
    
    base_dir = "/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm"
    
    files_to_check = [
        ("process/memory_utils.py", "Memory Management Utilities"),
        ("config.py", "Configuration File"),
        ("run_dpsk_ocr_pdf.py", "PDF Processing Script"),
        ("process/image_process.py", "Image Processing Module"),
    ]
    
    all_ok = True
    
    print("Checking file existence and syntax:\n")
    for filename, description in files_to_check:
        filepath = os.path.join(base_dir, filename)
        exists = check_file_exists(filepath, description)
        
        if exists:
            if check_syntax(filepath):
                print(f"  ✓ Syntax OK\n")
            else:
                all_ok = False
                print()
        else:
            all_ok = False
            print()
    
    # Check for new configuration parameters
    print("Checking configuration parameters:\n")
    config_path = os.path.join(base_dir, "config.py")
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    required_params = [
        "PDF_BATCH_SIZE",
        "ENABLE_MEMORY_MONITORING",
        "MEMORY_CLEANUP_FREQUENCY",
        "CHECKPOINT_ENABLED",
        "CHECKPOINT_DIR",
        "GPU_MEMORY_THRESHOLD_GB"
    ]
    
    for param in required_params:
        if param in config_content:
            print(f"✓ {param} found in config.py")
        else:
            print(f"✗ {param} NOT FOUND in config.py")
            all_ok = False
    
    # Check requirements.txt
    print("\nChecking requirements.txt:\n")
    req_path = "/vercel/sandbox/requirements.txt"
    with open(req_path, 'r') as f:
        req_content = f.read()
    
    if "psutil" in req_content:
        print("✓ psutil added to requirements.txt")
    else:
        print("✗ psutil NOT FOUND in requirements.txt")
        all_ok = False
    
    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✓ All checks passed! Installation is complete.")
        print("\nYou can now process large PDFs with:")
        print("  cd DeepSeek-OCR-master/DeepSeek-OCR-vllm")
        print("  python run_dpsk_ocr_pdf.py")
    else:
        print("✗ Some checks failed. Please review the errors above.")
    print("="*60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
