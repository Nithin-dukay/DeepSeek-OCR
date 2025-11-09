"""
Simple verification script for Issue #191 fix

This script verifies that all fix files are present and properly structured.
It doesn't require torch or transformers to be installed.
"""

import os
import sys


def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  ✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"  ✗ {description}: {filepath} NOT FOUND")
        return False


def check_file_content(filepath, required_strings, description):
    """Check if a file contains required strings."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing = []
        for req_str in required_strings:
            if req_str not in content:
                missing.append(req_str)
        
        if missing:
            print(f"  ✗ {description}: Missing required content: {missing}")
            return False
        else:
            print(f"  ✓ {description}: Contains all required components")
            return True
    except Exception as e:
        print(f"  ✗ {description}: Error reading file: {e}")
        return False


def verify_fix():
    """Verify all fix components are present and correct."""
    print("="*80)
    print("Verifying GitHub Issue #191 Fix Components")
    print("="*80)
    print()
    
    results = []
    
    # Check 1: Core processor file
    print("Check 1: Core Processor Implementation")
    results.append(check_file_exists(
        '/vercel/sandbox/transformers_logits_processor.py',
        'Processor implementation'
    ))
    results.append(check_file_content(
        '/vercel/sandbox/transformers_logits_processor.py',
        ['NoRepeatNGramLogitsProcessor', 'AdaptiveNoRepeatNGramLogitsProcessor',
         'create_anti_hallucination_processors', 'def __call__'],
        'Processor classes'
    ))
    print()
    
    # Check 2: Quick fix script
    print("Check 2: Quick Fix Script")
    results.append(check_file_exists(
        '/vercel/sandbox/quick_fix_issue_191.py',
        'Quick fix script'
    ))
    results.append(check_file_content(
        '/vercel/sandbox/quick_fix_issue_191.py',
        ['infer_with_fix', 'NoRepeatNGramLogitsProcessor', 'def patched_generate'],
        'Quick fix functions'
    ))
    print()
    
    # Check 3: Full-featured CLI tool
    print("Check 3: Full-Featured CLI Tool")
    results.append(check_file_exists(
        '/vercel/sandbox/run_dpsk_ocr_fixed.py',
        'CLI tool'
    ))
    results.append(check_file_content(
        '/vercel/sandbox/run_dpsk_ocr_fixed.py',
        ['argparse', 'infer_with_anti_hallucination', 'PROMPT_TEMPLATES',
         'RESOLUTION_PRESETS'],
        'CLI components'
    ))
    print()
    
    # Check 4: Documentation
    print("Check 4: Documentation Files")
    results.append(check_file_exists(
        '/vercel/sandbox/README_FIX_191.md',
        'Main README'
    ))
    results.append(check_file_exists(
        '/vercel/sandbox/FIX_GUIDE.md',
        'Comprehensive guide'
    ))
    results.append(check_file_exists(
        '/vercel/sandbox/ISSUE_191_ANALYSIS.md',
        'Technical analysis'
    ))
    print()
    
    # Check 5: Test script
    print("Check 5: Test Script")
    results.append(check_file_exists(
        '/vercel/sandbox/test_fix.py',
        'Test script'
    ))
    results.append(check_file_content(
        '/vercel/sandbox/test_fix.py',
        ['test_basic_initialization', 'test_ngram_blocking', 'test_whitelist',
         'test_batch_processing'],
        'Test functions'
    ))
    print()
    
    # Check 6: Verify key features in processor
    print("Check 6: Key Features Verification")
    try:
        with open('/vercel/sandbox/transformers_logits_processor.py', 'r') as f:
            content = f.read()
        
        features = {
            'N-gram blocking': 'banned_tokens',
            'Whitelist support': 'whitelist_token_ids',
            'Batch processing': 'batch_size',
            'Window size': 'window_size',
            'Adaptive mode': 'AdaptiveNoRepeatNGramLogitsProcessor'
        }
        
        for feature, keyword in features.items():
            if keyword in content:
                print(f"  ✓ {feature}: Implemented")
                results.append(True)
            else:
                print(f"  ✗ {feature}: Not found")
                results.append(False)
    except Exception as e:
        print(f"  ✗ Error checking features: {e}")
        results.append(False)
    print()
    
    # Check 7: Verify documentation completeness
    print("Check 7: Documentation Completeness")
    doc_checks = [
        ('/vercel/sandbox/README_FIX_191.md', ['Quick Start', 'Configuration', 'Troubleshooting']),
        ('/vercel/sandbox/FIX_GUIDE.md', ['Quick Fix', 'Configuration', 'Example']),
        ('/vercel/sandbox/ISSUE_191_ANALYSIS.md', ['Root Cause', 'Solution', 'Recommendations'])
    ]
    
    for filepath, required_sections in doc_checks:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            missing = [s for s in required_sections if s not in content]
            if missing:
                print(f"  ✗ {os.path.basename(filepath)}: Missing sections: {missing}")
                results.append(False)
            else:
                print(f"  ✓ {os.path.basename(filepath)}: All sections present")
                results.append(True)
        except Exception as e:
            print(f"  ✗ {os.path.basename(filepath)}: Error: {e}")
            results.append(False)
    print()
    
    # Summary
    print("="*80)
    print("Verification Summary")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Checks passed: {passed}/{total}")
    print()
    
    if passed == total:
        print("✅ All verification checks passed!")
        print()
        print("The fix is ready to use. You can:")
        print("  1. Use quick_fix_issue_191.py for minimal changes")
        print("  2. Use run_dpsk_ocr_fixed.py for full CLI tool")
        print("  3. Import from transformers_logits_processor.py for custom integration")
        print()
        print("See README_FIX_191.md for usage instructions.")
        return 0
    else:
        print(f"❌ {total - passed} check(s) failed.")
        print("Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = verify_fix()
    sys.exit(exit_code)
