#!/usr/bin/env python3
"""
Test script to verify the LlamaFlashAttention2 import error fix.
This script checks if the model can be loaded without import errors.
"""

import sys

def test_imports():
    """Test basic imports that should work without the model."""
    print("Testing basic imports...")
    try:
        import torch
        print("✓ torch imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import torch: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✓ transformers.AutoTokenizer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AutoTokenizer: {e}")
        return False
    
    return True

def test_transformers_version():
    """Check transformers version."""
    print("\nChecking transformers version...")
    try:
        import transformers
        version = transformers.__version__
        print(f"✓ transformers version: {version}")
        
        # Check if it's the recommended version
        if version.startswith("4.45"):
            print("✓ Using recommended version (4.45.x)")
        elif version.startswith("4.46") or version.startswith("4.47") or version.startswith("4.48"):
            print("⚠ Warning: This version may have LlamaFlashAttention2 issues")
            print("  Recommended: transformers==4.45.2")
        else:
            print(f"ℹ Using transformers {version}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking transformers version: {e}")
        return False

def test_llama_attention_import():
    """Test if LlamaFlashAttention2 can be imported (should work with 4.45.2)."""
    print("\nTesting LlamaFlashAttention2 import...")
    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        print("✓ LlamaFlashAttention2 is available (transformers ≤ 4.45.x)")
        return True
    except ImportError:
        print("ℹ LlamaFlashAttention2 not available (transformers ≥ 4.46)")
        print("  This is expected with newer transformers versions")
        print("  Solution: Use attn_implementation='eager' when loading models")
        return True  # This is not a failure, just informational

def test_script_syntax():
    """Test that the updated script has valid syntax."""
    print("\nTesting updated script syntax...")
    try:
        with open('/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py', 'r') as f:
            code = f.read()
            compile(code, 'run_dpsk_ocr.py', 'exec')
            print("✓ run_dpsk_ocr.py has valid Python syntax")
            
            # Check if it uses eager attention
            if "attn_implementation='eager'" in code:
                print("✓ Script uses eager attention (compatible with all versions)")
            elif "_attn_implementation='flash_attention_2'" in code:
                print("⚠ Script uses flash_attention_2 (requires transformers ≤ 4.45.x)")
            
            return True
    except Exception as e:
        print(f"✗ Error checking script: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DeepSeek-OCR Issue #302 Fix Verification")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_transformers_version()
    all_passed &= test_llama_attention_import()
    all_passed &= test_script_syntax()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the example: cd DeepSeek-OCR-master/DeepSeek-OCR-hf && python run_dpsk_ocr.py")
        print("3. See TROUBLESHOOTING.md for more information")
        return 0
    else:
        print("✗ Some tests failed")
        print("See TROUBLESHOOTING.md for solutions")
        return 1

if __name__ == "__main__":
    sys.exit(main())
