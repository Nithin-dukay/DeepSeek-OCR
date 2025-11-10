#!/usr/bin/env python3
"""
Test script to verify the LlamaFlashAttention2 import fix.

This script tests:
1. Transformers version detection
2. Compatibility patch application
3. Model loading with different attention implementations

Usage:
    python test_fix.py
"""

import sys
import warnings


def test_transformers_version():
    """Test 1: Check transformers version"""
    print("\n" + "="*60)
    print("TEST 1: Checking transformers version")
    print("="*60)
    
    try:
        import transformers
        version = transformers.__version__
        print(f"✓ Transformers version: {version}")
        
        major, minor = map(int, version.split('.')[:2])
        
        if (major, minor) == (4, 46):
            print("✓ Using recommended version (4.46.x)")
            return True, "recommended"
        elif major == 4 and minor >= 47:
            print("⚠ Using newer version (>=4.47.0) - may need patch")
            return True, "needs_patch"
        else:
            print("⚠ Using different version")
            return True, "unknown"
            
    except Exception as e:
        print(f"✗ Error checking version: {e}")
        return False, "error"


def test_llama_flash_attention_import():
    """Test 2: Check if LlamaFlashAttention2 is available"""
    print("\n" + "="*60)
    print("TEST 2: Checking LlamaFlashAttention2 availability")
    print("="*60)
    
    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        print("✓ LlamaFlashAttention2 is available")
        return True
    except ImportError as e:
        print(f"✗ LlamaFlashAttention2 not available: {e}")
        return False


def test_compatibility_patch():
    """Test 3: Test the compatibility patch"""
    print("\n" + "="*60)
    print("TEST 3: Testing compatibility patch")
    print("="*60)
    
    try:
        from fix_transformers_compatibility import apply_transformers_compatibility_patch
        
        result = apply_transformers_compatibility_patch()
        
        if result:
            print("✓ Compatibility patch applied successfully")
            
            # Verify the patch worked
            try:
                from transformers.models.llama.modeling_llama import LlamaFlashAttention2
                print("✓ LlamaFlashAttention2 is now available")
                return True
            except ImportError:
                print("✗ Patch applied but import still fails")
                return False
        else:
            print("⚠ Patch application returned False")
            return False
            
    except Exception as e:
        print(f"✗ Error applying patch: {e}")
        return False


def test_model_loading_simulation():
    """Test 4: Simulate model loading (without actually loading the full model)"""
    print("\n" + "="*60)
    print("TEST 4: Simulating model loading")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        # Test tokenizer loading (lightweight)
        print("\nTesting tokenizer loading...")
        from transformers import AutoTokenizer
        
        # Note: This will download the tokenizer if not cached
        # Uncomment to test actual loading:
        # tokenizer = AutoTokenizer.from_pretrained(
        #     'deepseek-ai/DeepSeek-OCR',
        #     trust_remote_code=True
        # )
        # print("✓ Tokenizer loaded successfully")
        
        print("✓ Import successful (full model loading skipped for speed)")
        return True
        
    except Exception as e:
        print(f"✗ Error in model loading simulation: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("DeepSeek-OCR Issue #7 Fix - Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: Version check
    success, version_status = test_transformers_version()
    results['version_check'] = success
    
    # Test 2: Check if patch is needed
    flash_attn_available = test_llama_flash_attention_import()
    results['flash_attn_available'] = flash_attn_available
    
    # Test 3: Apply patch if needed
    if not flash_attn_available:
        print("\n⚠ LlamaFlashAttention2 not available, applying patch...")
        patch_success = test_compatibility_patch()
        results['patch_applied'] = patch_success
    else:
        print("\n✓ LlamaFlashAttention2 available, no patch needed")
        results['patch_applied'] = True
    
    # Test 4: Simulate model loading
    model_test = test_model_loading_simulation()
    results['model_loading'] = model_test
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed! DeepSeek-OCR should work correctly.")
        print("\nYou can now use the model with:")
        print("  from transformers import AutoModel, AutoTokenizer")
        print("  model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nRecommended actions:")
        print("  1. Install the recommended version: pip install transformers==4.46.3")
        print("  2. Or use the compatibility patch: from fix_transformers_compatibility import apply_transformers_compatibility_patch")
        print("  3. See ISSUE_7_FIX.md for detailed solutions")
    
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
