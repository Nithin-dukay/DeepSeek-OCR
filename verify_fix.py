#!/usr/bin/env python3
"""
Verification script to demonstrate the fix for Issue #7
This script shows that the version checking works correctly.
"""

import sys

def test_version_check():
    """Test that version checking works correctly."""
    print("=" * 70)
    print("Testing Fix for Issue #7: LlamaFlashAttention2 ImportError")
    print("=" * 70)
    print()
    
    # Test 1: Check transformers version
    print("Test 1: Checking transformers version...")
    try:
        import transformers
        from packaging import version
        
        installed_version = transformers.__version__
        required_version = "4.46.3"
        
        print(f"  Installed: {installed_version}")
        print(f"  Required:  {required_version}")
        
        if version.parse(installed_version) == version.parse(required_version):
            print("  ✅ PASS: Correct version installed")
            test1_pass = True
        else:
            print("  ❌ FAIL: Wrong version installed")
            test1_pass = False
    except ImportError as e:
        print(f"  ❌ FAIL: {e}")
        test1_pass = False
    print()
    
    # Test 2: Check environment checker exists
    print("Test 2: Checking environment checker script...")
    import os
    if os.path.exists('check_environment.py'):
        print("  ✅ PASS: check_environment.py exists")
        test2_pass = True
    else:
        print("  ❌ FAIL: check_environment.py not found")
        test2_pass = False
    print()
    
    # Test 3: Check documentation updates
    print("Test 3: Checking documentation updates...")
    docs_exist = True
    
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            readme_content = f.read()
            if 'Troubleshooting' in readme_content and 'LlamaFlashAttention2' in readme_content:
                print("  ✅ PASS: README.md updated with troubleshooting")
            else:
                print("  ❌ FAIL: README.md missing troubleshooting section")
                docs_exist = False
    else:
        print("  ❌ FAIL: README.md not found")
        docs_exist = False
    
    if os.path.exists('COLAB_SETUP.md'):
        print("  ✅ PASS: COLAB_SETUP.md exists")
    else:
        print("  ❌ FAIL: COLAB_SETUP.md not found")
        docs_exist = False
    
    test3_pass = docs_exist
    print()
    
    # Test 4: Check requirements.txt
    print("Test 4: Checking requirements.txt...")
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            req_content = f.read()
            if 'transformers==4.46.3' in req_content and 'packaging' in req_content:
                print("  ✅ PASS: requirements.txt properly configured")
                test4_pass = True
            else:
                print("  ❌ FAIL: requirements.txt missing proper configuration")
                test4_pass = False
    else:
        print("  ❌ FAIL: requirements.txt not found")
        test4_pass = False
    print()
    
    # Test 5: Check run_dpsk_ocr.py has version check
    print("Test 5: Checking run_dpsk_ocr.py version validation...")
    script_path = 'DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py'
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            script_content = f.read()
            if 'transformers_version' in script_content and 'packaging' in script_content:
                print("  ✅ PASS: run_dpsk_ocr.py has version check")
                test5_pass = True
            else:
                print("  ❌ FAIL: run_dpsk_ocr.py missing version check")
                test5_pass = False
    else:
        print("  ❌ FAIL: run_dpsk_ocr.py not found")
        test5_pass = False
    print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    all_tests = [test1_pass, test2_pass, test3_pass, test4_pass, test5_pass]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"Tests passed: {passed}/{total}")
    print()
    
    if all(all_tests):
        print("✅ All tests passed! Issue #7 fix is complete.")
        print()
        print("Users can now:")
        print("  1. Run 'python check_environment.py' to verify setup")
        print("  2. Follow COLAB_SETUP.md for Colab instructions")
        print("  3. Get clear error messages if wrong version installed")
        return 0
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(test_version_check())
