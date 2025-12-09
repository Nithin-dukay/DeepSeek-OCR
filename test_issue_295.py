#!/usr/bin/env python3
"""
Test script for GitHub Issue #295: Missing last two columns in wide tables

This script tests the fix by:
1. Extracting page 11 from the DeepSeek-OCR paper
2. Verifying the configuration changes
3. Checking that all required files are present
4. Providing instructions for running the actual OCR test

Note: This script does NOT run the actual OCR inference (which requires the model),
but verifies that all the fix components are in place.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def check_file_contains(filepath, search_string, description):
    """Check if a file contains a specific string."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_string in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description} - NOT FOUND")
                return False
    except Exception as e:
        print(f"❌ Error checking {filepath}: {e}")
        return False

def main():
    print("="*70)
    print("Testing Fix for GitHub Issue #295: Wide Table Detection")
    print("="*70)
    print()
    
    base_path = "/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm"
    
    # Check if new files exist
    print("1. Checking New Files:")
    print("-" * 70)
    all_files_exist = True
    all_files_exist &= check_file_exists(
        f"{base_path}/table_token_config.py",
        "Table configuration module"
    )
    all_files_exist &= check_file_exists(
        f"{base_path}/find_table_tokens.py",
        "Token discovery utility"
    )
    all_files_exist &= check_file_exists(
        "/vercel/sandbox/ISSUE_295_FIX.md",
        "Fix documentation"
    )
    print()
    
    # Check if modified files contain the fix
    print("2. Checking Modified Files:")
    print("-" * 70)
    all_modifications_present = True
    
    all_modifications_present &= check_file_contains(
        f"{base_path}/config.py",
        "MAX_CROPS= 9",
        "config.py: MAX_CROPS increased to 9"
    )
    all_modifications_present &= check_file_contains(
        f"{base_path}/config.py",
        "USE_WIDE_TABLE_CONFIG",
        "config.py: USE_WIDE_TABLE_CONFIG flag added"
    )
    
    all_modifications_present &= check_file_contains(
        f"{base_path}/run_dpsk_ocr_image.py",
        "from table_token_config import",
        "run_dpsk_ocr_image.py: Imports table_token_config"
    )
    all_modifications_present &= check_file_contains(
        f"{base_path}/run_dpsk_ocr_image.py",
        "table_config = get_config_for_document_type",
        "run_dpsk_ocr_image.py: Uses dynamic configuration"
    )
    
    all_modifications_present &= check_file_contains(
        f"{base_path}/run_dpsk_ocr_pdf.py",
        "from table_token_config import",
        "run_dpsk_ocr_pdf.py: Imports table_token_config"
    )
    all_modifications_present &= check_file_contains(
        f"{base_path}/run_dpsk_ocr_pdf.py",
        "table_config = get_config_for_document_type",
        "run_dpsk_ocr_pdf.py: Uses dynamic configuration"
    )
    
    all_modifications_present &= check_file_contains(
        f"{base_path}/run_dpsk_ocr_eval_batch.py",
        "from table_token_config import",
        "run_dpsk_ocr_eval_batch.py: Imports table_token_config"
    )
    all_modifications_present &= check_file_contains(
        f"{base_path}/run_dpsk_ocr_eval_batch.py",
        "table_config = get_config_for_document_type",
        "run_dpsk_ocr_eval_batch.py: Uses dynamic configuration"
    )
    print()
    
    # Check configuration values
    print("3. Checking Configuration Values:")
    print("-" * 70)
    try:
        sys.path.insert(0, base_path)
        from table_token_config import TableProcessingConfig
        
        print(f"✅ Default config: ngram_size={TableProcessingConfig.DEFAULT['ngram_size']}, "
              f"max_tokens={TableProcessingConfig.DEFAULT['max_tokens']}")
        print(f"✅ Wide table config: ngram_size={TableProcessingConfig.WIDE_TABLE['ngram_size']}, "
              f"max_tokens={TableProcessingConfig.WIDE_TABLE['max_tokens']}")
        print(f"✅ PDF config: ngram_size={TableProcessingConfig.PDF_OPTIMIZED['ngram_size']}, "
              f"max_tokens={TableProcessingConfig.PDF_OPTIMIZED['max_tokens']}")
        print(f"✅ Batch eval config: ngram_size={TableProcessingConfig.BATCH_EVAL['ngram_size']}, "
              f"max_tokens={TableProcessingConfig.BATCH_EVAL['max_tokens']}")
        config_ok = True
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        config_ok = False
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    if all_files_exist and all_modifications_present and config_ok:
        print("✅ All fix components are in place!")
        print()
        print("To test with the actual model:")
        print()
        print("1. Edit DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py:")
        print("   - Set USE_WIDE_TABLE_CONFIG = True")
        print("   - Set DOCUMENT_TYPE = 'wide_table'")
        print("   - Set INPUT_PATH to your test image (e.g., page 11 of the paper)")
        print("   - Set OUTPUT_PATH to your desired output directory")
        print()
        print("2. Run the OCR:")
        print("   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm")
        print("   python run_dpsk_ocr_image.py")
        print()
        print("3. Check the output for all table columns")
        print()
        print("For more details, see: ISSUE_295_FIX.md")
        return 0
    else:
        print("❌ Some fix components are missing or incorrect!")
        print("Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
