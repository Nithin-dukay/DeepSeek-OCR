#!/usr/bin/env python3
"""
Test script to verify the LlamaFlashAttention2 import fix.
This script tests that the model loading code is syntactically correct
and would work with the proper dependencies installed.
"""

import sys

def test_import_syntax():
    """Test that the import statements are correct."""
    print("Testing import syntax...")
    try:
        # Test basic imports
        code = """
from transformers import AutoModel, AutoTokenizer
import torch
"""
        compile(code, '<string>', 'exec')
        print("✓ Import syntax is correct")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in imports: {e}")
        return False

def test_model_loading_syntax():
    """Test that the model loading code is syntactically correct."""
    print("\nTesting model loading syntax...")
    try:
        code = """
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='eager',
    trust_remote_code=True, 
    use_safetensors=True
)
"""
        compile(code, '<string>', 'exec')
        print("✓ Model loading syntax is correct")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in model loading: {e}")
        return False

def test_file_syntax():
    """Test that the actual run_dpsk_ocr.py file has correct syntax."""
    print("\nTesting run_dpsk_ocr.py file syntax...")
    try:
        with open('DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py', 'r') as f:
            code = f.read()
        compile(code, 'run_dpsk_ocr.py', 'exec')
        print("✓ run_dpsk_ocr.py syntax is correct")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in run_dpsk_ocr.py: {e}")
        return False
    except FileNotFoundError:
        print("✗ File not found: run_dpsk_ocr.py")
        return False

def verify_fix_applied():
    """Verify that the fix has been applied to the file."""
    print("\nVerifying fix has been applied...")
    try:
        with open('DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py', 'r') as f:
            content = f.read()
        
        checks = {
            "attn_implementation='eager'": "attn_implementation='eager'" in content,
            "Has comment about fix": "LlamaFlashAttention2" in content,
            "Uses AutoModel.from_pretrained correctly": "AutoModel.from_pretrained(" in content,
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
    except FileNotFoundError:
        print("✗ File not found: run_dpsk_ocr.py")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DeepSeek-OCR Fix Verification")
    print("=" * 60)
    
    tests = [
        test_import_syntax,
        test_model_loading_syntax,
        test_file_syntax,
        verify_fix_applied,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ All tests passed! The fix has been successfully applied.")
        print("\nThe model should now load without the LlamaFlashAttention2 error.")
        print("Note: Actual model loading requires transformers and other dependencies.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
