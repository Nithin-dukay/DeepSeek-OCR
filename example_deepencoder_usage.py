"""
Simple Example: Using DeepEncoder Standalone
============================================

This is a minimal example showing how to use the DeepEncoder
to extract vision embeddings from images.

Usage:
    python example_deepencoder_usage.py
"""

import torch
from PIL import Image
import sys
import os

# Add path to DeepSeek-OCR modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-vllm'))

from reproduce_deepencoder import DeepEncoderStandalone


def example_1_basic_encoding():
    """Example 1: Basic image encoding"""
    print("\n" + "="*60)
    print("Example 1: Basic Image Encoding")
    print("="*60)
    
    # Initialize encoder with Base mode (1024x1024)
    encoder = DeepEncoderStandalone(
        model_path=None,  # Use random weights for demo
        base_size=1024,
        image_size=640,
        crop_mode=False,
        device="cpu"  # Use CPU for demo
    )
    
    # Create a dummy image
    image = Image.new('RGB', (800, 600), color='white')
    
    # Encode the image
    embeddings = encoder.encode_image(image)
    
    print(f"\n✓ Encoded image to {embeddings.shape[0]} tokens")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings


def example_2_different_modes():
    """Example 2: Compare different encoding modes"""
    print("\n" + "="*60)
    print("Example 2: Different Encoding Modes")
    print("="*60)
    
    image = Image.new('RGB', (1024, 768), color='blue')
    
    modes = [
        ("Tiny", 512, 512, False),
        ("Small", 640, 640, False),
        ("Base", 1024, 1024, False),
        ("Gundam", 1024, 640, True),
    ]
    
    for mode_name, base_size, image_size, crop_mode in modes:
        print(f"\n{mode_name} Mode:")
        print(f"  base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")
        
        encoder = DeepEncoderStandalone(
            model_path=None,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            device="cpu"
        )
        
        num_tokens = encoder.get_num_tokens(image)
        print(f"  → Expected tokens: {num_tokens}")


def example_3_intermediate_features():
    """Example 3: Access intermediate features"""
    print("\n" + "="*60)
    print("Example 3: Accessing Intermediate Features")
    print("="*60)
    
    encoder = DeepEncoderStandalone(
        model_path=None,
        base_size=640,
        image_size=640,
        crop_mode=False,
        device="cpu"
    )
    
    image = Image.new('RGB', (640, 640), color='red')
    
    # Get all intermediate features
    result = encoder.encode_image(image, return_intermediate=True)
    
    print("\nIntermediate Features:")
    print(f"  SAM features: {result['sam_features'].shape}")
    print(f"  CLIP features: {result['clip_features'].shape}")
    print(f"  Combined features: {result['combined_features'].shape}")
    print(f"  Final embeddings: {result['embeddings'].shape}")
    print(f"  Total tokens: {result['num_tokens']}")
    
    return result


def example_4_batch_processing():
    """Example 4: Batch processing multiple images"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)
    
    encoder = DeepEncoderStandalone(
        model_path=None,
        base_size=512,
        image_size=512,
        crop_mode=False,
        device="cpu"
    )
    
    # Create multiple dummy images
    images = [
        Image.new('RGB', (400, 300), color='red'),
        Image.new('RGB', (600, 400), color='green'),
        Image.new('RGB', (800, 600), color='blue'),
    ]
    
    print(f"\nProcessing {len(images)} images...")
    embeddings_list = encoder.encode_batch(images)
    
    print("\nResults:")
    for i, emb in enumerate(embeddings_list):
        print(f"  Image {i+1}: {emb.shape[0]} tokens, shape {emb.shape}")
    
    return embeddings_list


def example_5_understanding_architecture():
    """Example 5: Understanding the DeepEncoder architecture"""
    print("\n" + "="*60)
    print("Example 5: Understanding DeepEncoder Architecture")
    print("="*60)
    
    encoder = DeepEncoderStandalone(
        model_path=None,
        base_size=1024,
        image_size=640,
        crop_mode=False,
        device="cpu"
    )
    
    print("\nDeepEncoder Components:")
    print("  1. SAM Model (Spatial Features):")
    print(f"     - Type: {type(encoder.sam_model).__name__}")
    print(f"     - Output: 1024-dimensional features")
    
    print("\n  2. CLIP Model (Semantic Features):")
    print(f"     - Type: {type(encoder.vision_model).__name__}")
    print(f"     - Output: 1024-dimensional features")
    
    print("\n  3. MLP Projector:")
    print(f"     - Type: {type(encoder.projector).__name__}")
    print(f"     - Input: 2048-dim (SAM + CLIP)")
    print(f"     - Output: 1280-dim")
    
    print("\n  4. Special Tokens:")
    print(f"     - image_newline: {encoder.image_newline.shape}")
    print(f"     - view_separator: {encoder.view_separator.shape}")
    
    print("\nEncoding Pipeline:")
    print("  Image → SAM → CLIP → Concat → Projector → Add Tokens → Embeddings")


def example_6_token_calculation():
    """Example 6: Understanding token calculation"""
    print("\n" + "="*60)
    print("Example 6: Token Calculation")
    print("="*60)
    
    encoder = DeepEncoderStandalone(
        model_path=None,
        base_size=1024,
        image_size=640,
        crop_mode=False,
        device="cpu"
    )
    
    print("\nToken calculation formula:")
    print("  patch_size = 16")
    print("  downsample_ratio = 4")
    print("  h = w = (base_size / patch_size) / downsample_ratio")
    print("  tokens = h * (w + 1) + 1")
    print("           ↑     ↑       ↑")
    print("         rows  cols+    separator")
    print("               newline")
    
    test_sizes = [512, 640, 1024, 1280]
    
    print("\nToken counts for different base sizes:")
    for size in test_sizes:
        encoder.base_size = size
        image = Image.new('RGB', (size, size), color='white')
        num_tokens = encoder.get_num_tokens(image)
        
        h = w = int((size // 16) / 4)
        print(f"  {size}x{size}: h={h}, w={w} → {num_tokens} tokens")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("DeepEncoder Standalone - Usage Examples")
    print("="*60)
    print("\nThese examples demonstrate how to use the DeepEncoder")
    print("independently to extract vision embeddings from images.")
    
    try:
        # Run examples
        example_1_basic_encoding()
        example_2_different_modes()
        example_3_intermediate_features()
        example_4_batch_processing()
        example_5_understanding_architecture()
        example_6_token_calculation()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        
        print("\nNext Steps:")
        print("  1. Load real model weights from 'deepseek-ai/DeepSeek-OCR'")
        print("  2. Process your own images")
        print("  3. Use embeddings for downstream tasks")
        print("\nFor full usage, see: reproduce_deepencoder.py")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("\nNote: These examples use random weights for demonstration.")
        print("To use real weights, download the DeepSeek-OCR model.")


if __name__ == "__main__":
    main()
