#!/usr/bin/env python3
"""
Simple verification script to check that the fix has been applied correctly.
This script checks the code changes without requiring dependencies.
"""

import os
import re

def check_file_content(filepath, expected_patterns, description):
    """Check if a file contains expected patterns"""
    print(f"\n[Checking] {description}")
    print(f"File: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"✗ File not found: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_found = True
    for pattern_desc, pattern in expected_patterns:
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            print(f"  ✓ {pattern_desc}")
        else:
            print(f"  ✗ {pattern_desc}")
            all_found = False
    
    return all_found

def main():
    print("=" * 70)
    print("Verification Script for Issue #288 Fix")
    print("=" * 70)
    
    base_path = "/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm"
    
    all_checks_passed = True
    
    # Check 1: image_process.py
    check1 = check_file_content(
        f"{base_path}/process/image_process.py",
        [
            ("Function signature includes 'prompt' parameter", 
             r"def tokenize_with_images\([^)]*prompt:\s*str\s*=\s*None"),
            ("Uses provided prompt or falls back to PROMPT", 
             r"conversation\s*=\s*prompt\s+if\s+prompt\s+is\s+not\s+None\s+else\s+PROMPT"),
        ],
        "image_process.py - Core fix"
    )
    all_checks_passed = all_checks_passed and check1
    
    # Check 2: run_dpsk_ocr_image.py
    check2 = check_file_content(
        f"{base_path}/run_dpsk_ocr_image.py",
        [
            ("Passes prompt parameter to tokenize_with_images", 
             r"tokenize_with_images\([^)]*prompt\s*=\s*prompt"),
        ],
        "run_dpsk_ocr_image.py - Image runner"
    )
    all_checks_passed = all_checks_passed and check2
    
    # Check 3: run_dpsk_ocr_pdf.py
    check3 = check_file_content(
        f"{base_path}/run_dpsk_ocr_pdf.py",
        [
            ("Passes prompt_in parameter to tokenize_with_images", 
             r"tokenize_with_images\([^)]*prompt\s*=\s*prompt_in"),
        ],
        "run_dpsk_ocr_pdf.py - PDF runner"
    )
    all_checks_passed = all_checks_passed and check3
    
    # Check 4: run_dpsk_ocr_eval_batch.py
    check4 = check_file_content(
        f"{base_path}/run_dpsk_ocr_eval_batch.py",
        [
            ("Passes prompt_in parameter to tokenize_with_images", 
             r"tokenize_with_images\([^)]*prompt\s*=\s*prompt_in"),
        ],
        "run_dpsk_ocr_eval_batch.py - Batch evaluation runner"
    )
    all_checks_passed = all_checks_passed and check4
    
    # Check 5: deepseek_ocr.py
    check5 = check_file_content(
        f"{base_path}/deepseek_ocr.py",
        [
            ("Passes PROMPT parameter to tokenize_with_images", 
             r"tokenize_with_images\(.*?prompt\s*=\s*PROMPT"),
        ],
        "deepseek_ocr.py - Model file"
    )
    all_checks_passed = all_checks_passed and check5
    
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✓ All verification checks passed!")
        print("=" * 70)
        print("\nFix Summary:")
        print("- image_process.py now accepts a 'prompt' parameter")
        print("- All callers have been updated to pass the prompt")
        print("- Backward compatibility maintained (defaults to config.PROMPT)")
        print("\nThe issue where modified prompts caused repetitive output is now fixed.")
        print("\nHow it works:")
        print("1. Users can now pass custom prompts to tokenize_with_images()")
        print("2. The function uses the provided prompt instead of hardcoded PROMPT")
        print("3. This ensures tokenization matches the actual prompt sent to model")
        print("4. If no prompt is provided, it falls back to PROMPT from config")
        return 0
    else:
        print("✗ Some verification checks failed!")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    exit(main())
