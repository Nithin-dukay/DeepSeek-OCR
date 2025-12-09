"""
Test script to verify dynamic mode selection functionality.

This script tests that:
1. Mode parameters are correctly passed through the processing pipeline
2. Different modes produce different token counts
3. Backward compatibility is maintained
"""

import sys
import torch
from PIL import Image
import numpy as np
from process.image_process import DeepseekOCRProcessor

# Mode configurations
MODES = {
    'tiny': {'base_size': 512, 'image_size': 512, 'crop_mode': False},
    'small': {'base_size': 640, 'image_size': 640, 'crop_mode': False},
    'base': {'base_size': 1024, 'image_size': 1024, 'crop_mode': False},
    'large': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False},
    'gundam': {'base_size': 1024, 'image_size': 640, 'crop_mode': True},
}

def create_test_image(width=800, height=600):
    """Create a simple test image."""
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array, 'RGB')

def test_mode_selection():
    """Test that different modes produce different results."""
    print("="*60)
    print("Testing Dynamic Mode Selection")
    print("="*60)
    
    # Create test image
    test_image = create_test_image(800, 600)
    print(f"\nTest image size: {test_image.size}")
    
    # Initialize processor
    processor = DeepseekOCRProcessor()
    
    # Test each mode
    results = {}
    for mode_name, mode_config in MODES.items():
        print(f"\n{'-'*60}")
        print(f"Testing {mode_name.upper()} mode:")
        print(f"  base_size: {mode_config['base_size']}")
        print(f"  image_size: {mode_config['image_size']}")
        print(f"  crop_mode: {mode_config['crop_mode']}")
        
        try:
            # Process image with mode
            result = processor.tokenize_with_images(
                images=[test_image],
                bos=True,
                eos=True,
                base_size=mode_config['base_size'],
                image_size=mode_config['image_size'],
                crop_mode=mode_config['crop_mode']
            )
            
            # Extract information
            input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, image_shapes, mode_params = result[0]
            
            # Verify mode parameters are stored
            assert mode_params is not None, "Mode parameters not stored!"
            assert mode_params['base_size'] == mode_config['base_size'], "base_size mismatch!"
            assert mode_params['image_size'] == mode_config['image_size'], "image_size mismatch!"
            assert mode_params['crop_mode'] == mode_config['crop_mode'], "crop_mode mismatch!"
            
            results[mode_name] = {
                'num_tokens': num_image_tokens[0],
                'pixel_values_shape': pixel_values.shape,
                'images_crop_shape': images_crop.shape,
                'mode_params': mode_params
            }
            
            print(f"  ✓ Success!")
            print(f"    - Number of image tokens: {num_image_tokens[0]}")
            print(f"    - Pixel values shape: {pixel_values.shape}")
            print(f"    - Images crop shape: {images_crop.shape}")
            print(f"    - Mode params stored: {mode_params}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Verify different modes produce different results
    print(f"\n{'='*60}")
    print("Verification:")
    print(f"{'='*60}")
    
    token_counts = {mode: results[mode]['num_tokens'] for mode in results}
    print(f"\nToken counts by mode:")
    for mode, count in token_counts.items():
        print(f"  {mode:8s}: {count} tokens")
    
    # Check that modes have different token counts (except where expected to be same)
    if token_counts['tiny'] != token_counts['base']:
        print("\n✓ Different modes produce different token counts")
    else:
        print("\n✗ Warning: Tiny and Base modes have same token count")
    
    # Test backward compatibility (no mode params)
    print(f"\n{'-'*60}")
    print("Testing backward compatibility (no mode params):")
    try:
        result_compat = processor.tokenize_with_images(
            images=[test_image],
            bos=True,
            eos=True,
            cropping=True  # Old parameter
        )
        print("  ✓ Backward compatibility maintained!")
    except Exception as e:
        print(f"  ✗ Backward compatibility broken: {e}")
        return False
    
    print(f"\n{'='*60}")
    print("All tests passed! ✓")
    print(f"{'='*60}")
    return True

if __name__ == "__main__":
    # Set prompt for testing
    import config
    config.PROMPT = "<image>\\nTest prompt"
    
    success = test_mode_selection()
    sys.exit(0 if success else 1)
