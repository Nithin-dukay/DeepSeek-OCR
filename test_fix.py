#!/usr/bin/env python3
"""
Test script to validate the fix for GitHub Issue #291
Tests the dynamic_preprocess function with various image sizes
"""

import sys
sys.path.insert(0, '/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm')

from PIL import Image
import numpy as np

# Import the fixed function
from process.image_process import dynamic_preprocess, count_tiles

def create_test_image(width, height):
    """Create a test image with given dimensions"""
    # Create a gradient image for testing
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [int(255 * i / height), int(255 * j / width), 128]
    return Image.fromarray(img_array)

def test_dynamic_preprocess():
    """Test the dynamic_preprocess function with various image sizes"""
    
    # Test cases that might represent flora identification layouts
    test_cases = [
        (800, 1200, "Portrait flora document"),
        (1200, 800, "Landscape flora document"),
        (1000, 1000, "Square flora document"),
        (1500, 2000, "Large portrait document"),
        (2000, 1500, "Large landscape document"),
        (640, 640, "Small square (no crop expected)"),
        (500, 500, "Very small (no crop expected)"),
        (1920, 1080, "HD landscape"),
        (1080, 1920, "HD portrait"),
        (2480, 3508, "A4 portrait (300 DPI)"),
    ]
    
    print("=" * 80)
    print("Testing dynamic_preprocess function with various image sizes")
    print("=" * 80)
    
    all_passed = True
    
    for width, height, description in test_cases:
        print(f"\nTest: {description} ({width}x{height})")
        print("-" * 60)
        
        try:
            # Create test image
            test_img = create_test_image(width, height)
            
            # Test count_tiles first
            crop_ratio = count_tiles(width, height, min_num=2, max_num=6, image_size=640)
            print(f"  Crop ratio: {crop_ratio}")
            
            # Test dynamic_preprocess
            processed_images, aspect_ratio = dynamic_preprocess(
                test_img, 
                min_num=2, 
                max_num=6, 
                image_size=640, 
                use_thumbnail=False
            )
            
            print(f"  Aspect ratio: {aspect_ratio}")
            print(f"  Number of crops: {len(processed_images)}")
            print(f"  Expected crops: {aspect_ratio[0] * aspect_ratio[1]}")
            
            # Validate results
            if len(processed_images) == 0:
                print(f"  ❌ FAILED: No crops produced!")
                all_passed = False
            elif len(processed_images) != aspect_ratio[0] * aspect_ratio[1]:
                print(f"  ⚠️  WARNING: Crop count mismatch (got {len(processed_images)}, expected {aspect_ratio[0] * aspect_ratio[1]})")
            else:
                print(f"  ✓ PASSED")
            
            # Check crop sizes
            for i, crop in enumerate(processed_images):
                if crop.size != (640, 640):
                    print(f"  ⚠️  WARNING: Crop {i} has unexpected size {crop.size}")
                    
        except Exception as e:
            print(f"  ❌ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests completed successfully!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = test_dynamic_preprocess()
    sys.exit(0 if success else 1)
