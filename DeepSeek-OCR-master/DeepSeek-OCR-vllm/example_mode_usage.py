#!/usr/bin/env python3
"""
Example script demonstrating how to use different OCR modes programmatically.

This script shows how to:
1. Get available modes
2. Get mode configurations
3. Use modes in your own code
"""

from modes import (
    get_available_modes,
    get_mode_config,
    get_mode_info,
    print_available_modes
)

def main():
    print("=" * 80)
    print("DeepSeek-OCR Mode Selection Examples")
    print("=" * 80)
    
    # Example 1: List all available modes
    print("\n1. Available Modes:")
    modes = get_available_modes()
    print(f"   {', '.join(modes)}")
    
    # Example 2: Get configuration for a specific mode
    print("\n2. Get Mode Configuration:")
    mode_name = "Gundam"
    base_size, image_size, crop_mode = get_mode_config(mode_name)
    print(f"   Mode: {mode_name}")
    print(f"   Base Size: {base_size}")
    print(f"   Image Size: {image_size}")
    print(f"   Crop Mode: {crop_mode}")
    
    # Example 3: Get detailed mode information
    print("\n3. Detailed Mode Information:")
    info = get_mode_info("Base")
    print(f"   Mode: {info['mode']}")
    print(f"   Base Size: {info['base_size']}")
    print(f"   Image Size: {info['image_size']}")
    print(f"   Crop Mode: {info['crop_mode']}")
    print(f"   Vision Tokens: {info['vision_tokens']}")
    print(f"   Description: {info['description']}")
    
    # Example 4: Compare all modes
    print("\n4. Mode Comparison:")
    print(f"   {'Mode':<10} {'Base Size':<12} {'Image Size':<12} {'Crop':<8} {'Tokens':<10}")
    print("   " + "-" * 60)
    for mode in modes:
        info = get_mode_info(mode)
        print(f"   {info['mode']:<10} {info['base_size']:<12} {info['image_size']:<12} "
              f"{str(info['crop_mode']):<8} {str(info['vision_tokens']):<10}")
    
    # Example 5: Use mode in your code
    print("\n5. Using Mode in Your Code:")
    print("   ```python")
    print("   from modes import get_mode_config")
    print("   from process.image_process import DeepseekOCRProcessor")
    print()
    print("   # Get mode configuration")
    print("   base_size, image_size, crop_mode = get_mode_config('Gundam')")
    print()
    print("   # Use in image processing")
    print("   processor = DeepseekOCRProcessor()")
    print("   result = processor.tokenize_with_images(")
    print("       images=[image],")
    print("       bos=True,")
    print("       eos=True,")
    print("       cropping=crop_mode,")
    print("       image_size=image_size,")
    print("       base_size=base_size")
    print("   )")
    print("   ```")
    
    # Example 6: Print all modes with details
    print("\n6. All Modes with Full Details:")
    print_available_modes()
    
    # Example 7: Mode selection recommendations
    print("\n7. Mode Selection Recommendations:")
    recommendations = {
        "Tiny": "Quick testing, low memory (2GB GPU)",
        "Small": "Fast batch processing, simple documents (3GB GPU)",
        "Base": "General purpose, balanced quality/speed (5GB GPU)",
        "Large": "High quality single images (8GB GPU)",
        "Gundam": "Best quality, complex documents (10-15GB GPU)"
    }
    
    for mode, recommendation in recommendations.items():
        print(f"   {mode:<10} → {recommendation}")
    
    print("\n" + "=" * 80)
    print("For more information, see MODE_SELECTION_GUIDE.md")
    print("=" * 80)

if __name__ == "__main__":
    main()
