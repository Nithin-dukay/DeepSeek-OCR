"""
Test script to demonstrate dynamic mode selection in vLLM deployment.

This script shows how to use different OCR modes (Tiny, Small, Base, Large, Gundam)
dynamically without modifying config.py.
"""

from process.image_process import DeepseekOCRProcessor
from PIL import Image
import torch

# Create a simple test image
def create_test_image(width=1000, height=800):
    """Create a simple test image for demonstration"""
    from PIL import ImageDraw, ImageFont
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some text
    text = "Test OCR Image"
    draw.text((width//2 - 100, height//2), text, fill='black')
    
    return img

def test_mode(mode_name, base_size, image_size, crop_mode):
    """Test a specific OCR mode"""
    print(f"\n{'='*60}")
    print(f"Testing {mode_name} Mode")
    print(f"  base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")
    print(f"{'='*60}")
    
    # Create test image
    test_image = create_test_image()
    print(f"Test image size: {test_image.size}")
    
    # Process with specified mode
    processor = DeepseekOCRProcessor()
    result = processor.tokenize_with_images(
        images=[test_image],
        bos=True,
        eos=True,
        cropping=crop_mode,
        base_size=base_size,
        image_size=image_size
    )
    
    # Extract information from result
    input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, image_shapes = result[0]
    
    print(f"Results:")
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Pixel values shape: {pixel_values.shape}")
    print(f"  Images crop shape: {images_crop.shape}")
    print(f"  Spatial crop: {images_spatial_crop}")
    print(f"  Number of image tokens: {num_image_tokens}")
    print(f"  Total vision tokens: {sum(num_image_tokens)}")
    
    return result

def main():
    print("="*60)
    print("DeepSeek-OCR Dynamic Mode Selection Test")
    print("="*60)
    
    # Test all supported modes
    modes = [
        ("Tiny", 512, 512, False),
        ("Small", 640, 640, False),
        ("Base", 1024, 1024, False),
        ("Large", 1280, 1280, False),
        ("Gundam", 1024, 640, True),
    ]
    
    for mode_name, base_size, image_size, crop_mode in modes:
        try:
            test_mode(mode_name, base_size, image_size, crop_mode)
        except Exception as e:
            print(f"Error testing {mode_name} mode: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All mode tests completed!")
    print("="*60)
    
    print("\n" + "="*60)
    print("Usage Example in Your Code:")
    print("="*60)
    print("""
# Example 1: Use Small mode (640x640, no cropping)
image_features = DeepseekOCRProcessor().tokenize_with_images(
    images=[your_image],
    bos=True,
    eos=True,
    cropping=False,
    base_size=640,
    image_size=640
)

# Example 2: Use Gundam mode (dynamic resolution)
image_features = DeepseekOCRProcessor().tokenize_with_images(
    images=[your_image],
    bos=True,
    eos=True,
    cropping=True,
    base_size=1024,
    image_size=640
)

# Example 3: Use config.py defaults (omit optional parameters)
image_features = DeepseekOCRProcessor().tokenize_with_images(
    images=[your_image],
    bos=True,
    eos=True,
    cropping=True  # Uses BASE_SIZE and IMAGE_SIZE from config.py
)
""")

if __name__ == "__main__":
    main()
