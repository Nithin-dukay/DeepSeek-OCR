#!/usr/bin/env python3
"""
Test script to verify that custom prompts work correctly after the fix.
This script tests the tokenize_with_images function with different prompts.
"""

import sys
import os

# Add the DeepSeek-OCR-vllm directory to the path
sys.path.insert(0, '/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm')

from process.image_process import DeepseekOCRProcessor
from PIL import Image
import torch

def create_test_image():
    """Create a simple test image"""
    # Create a simple 640x640 white image with some text-like patterns
    img = Image.new('RGB', (640, 640), color='white')
    return img

def test_custom_prompt():
    """Test that custom prompts are properly tokenized"""
    
    print("=" * 60)
    print("Testing Custom Prompt Fix for Issue #288")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image()
    
    # Initialize processor
    processor = DeepseekOCRProcessor()
    
    # Test 1: Original prompt (should work)
    print("\n[Test 1] Testing original prompt...")
    original_prompt = "<image>\n<|grounding|>Convert the document to markdown."
    try:
        result1 = processor.tokenize_with_images(
            images=[test_image],
            bos=True,
            eos=True,
            cropping=False,
            prompt=original_prompt
        )
        print("✓ Original prompt tokenization successful")
        print(f"  Input IDs shape: {result1[0][0].shape}")
        print(f"  Number of image tokens: {result1[0][5]}")
    except Exception as e:
        print(f"✗ Original prompt failed: {e}")
        return False
    
    # Test 2: Modified prompt (the one that was failing)
    print("\n[Test 2] Testing modified prompt (Issue #288)...")
    modified_prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
    try:
        result2 = processor.tokenize_with_images(
            images=[test_image],
            bos=True,
            eos=True,
            cropping=False,
            prompt=modified_prompt
        )
        print("✓ Modified prompt tokenization successful")
        print(f"  Input IDs shape: {result2[0][0].shape}")
        print(f"  Number of image tokens: {result2[0][5]}")
    except Exception as e:
        print(f"✗ Modified prompt failed: {e}")
        return False
    
    # Test 3: Verify that different prompts produce different token sequences
    print("\n[Test 3] Verifying prompts produce different tokenizations...")
    input_ids_1 = result1[0][0]
    input_ids_2 = result2[0][0]
    
    # The token sequences should be different lengths due to different text
    if input_ids_1.shape != input_ids_2.shape:
        print("✓ Different prompts produce different token sequences")
        print(f"  Original prompt tokens: {input_ids_1.shape[1]}")
        print(f"  Modified prompt tokens: {input_ids_2.shape[1]}")
    else:
        # Even if same length, the actual tokens should differ
        if not torch.equal(input_ids_1, input_ids_2):
            print("✓ Different prompts produce different token sequences")
            print(f"  Both have {input_ids_1.shape[1]} tokens but different content")
        else:
            print("✗ WARNING: Different prompts produced identical tokenization!")
            print("  This suggests the prompt parameter might not be working correctly")
            return False
    
    # Test 4: Test with no prompt parameter (should use default from config)
    print("\n[Test 4] Testing backward compatibility (no prompt parameter)...")
    try:
        result3 = processor.tokenize_with_images(
            images=[test_image],
            bos=True,
            eos=True,
            cropping=False
        )
        print("✓ Backward compatibility maintained (uses default PROMPT from config)")
        print(f"  Input IDs shape: {result3[0][0].shape}")
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False
    
    # Test 5: Test with another custom prompt
    print("\n[Test 5] Testing another custom prompt...")
    custom_prompt = "<image>\nFree OCR."
    try:
        result4 = processor.tokenize_with_images(
            images=[test_image],
            bos=True,
            eos=True,
            cropping=False,
            prompt=custom_prompt
        )
        print("✓ Another custom prompt tokenization successful")
        print(f"  Input IDs shape: {result4[0][0].shape}")
    except Exception as e:
        print(f"✗ Custom prompt test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Issue #288 is fixed.")
    print("=" * 60)
    print("\nSummary:")
    print("- Custom prompts are now properly tokenized")
    print("- Different prompts produce different token sequences")
    print("- Backward compatibility is maintained")
    print("- The model should now work correctly with modified prompts")
    
    return True

if __name__ == "__main__":
    try:
        success = test_custom_prompt()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test script failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
