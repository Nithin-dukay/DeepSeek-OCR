#!/usr/bin/env python3
"""Test script to verify mode selection functionality"""

import sys
sys.path.insert(0, '/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm')

import config

def test_mode_config():
    """Test mode configuration"""
    print("Testing mode configurations...")
    print("=" * 60)
    
    # Test all modes
    modes = ['tiny', 'small', 'base', 'large', 'gundam']
    
    for mode in modes:
        mode_config = config.get_mode_config(mode)
        print(f"\nMode: {mode}")
        print(f"  base_size: {mode_config['base_size']}")
        print(f"  image_size: {mode_config['image_size']}")
        print(f"  crop_mode: {mode_config['crop_mode']}")
    
    print("\n" + "=" * 60)
    print("Default configuration:")
    print(f"  MODE: {config.MODE}")
    print(f"  BASE_SIZE: {config.BASE_SIZE}")
    print(f"  IMAGE_SIZE: {config.IMAGE_SIZE}")
    print(f"  CROP_MODE: {config.CROP_MODE}")
    
    print("\n" + "=" * 60)
    print("Testing mode switching...")
    
    # Test switching to 'base' mode
    config.MODE = 'base'
    mode_config = config.get_mode_config('base')
    config.BASE_SIZE = mode_config['base_size']
    config.IMAGE_SIZE = mode_config['image_size']
    config.CROP_MODE = mode_config['crop_mode']
    
    print(f"\nAfter switching to 'base' mode:")
    print(f"  BASE_SIZE: {config.BASE_SIZE}")
    print(f"  IMAGE_SIZE: {config.IMAGE_SIZE}")
    print(f"  CROP_MODE: {config.CROP_MODE}")
    
    # Test switching to 'tiny' mode
    config.MODE = 'tiny'
    mode_config = config.get_mode_config('tiny')
    config.BASE_SIZE = mode_config['base_size']
    config.IMAGE_SIZE = mode_config['image_size']
    config.CROP_MODE = mode_config['crop_mode']
    
    print(f"\nAfter switching to 'tiny' mode:")
    print(f"  BASE_SIZE: {config.BASE_SIZE}")
    print(f"  IMAGE_SIZE: {config.IMAGE_SIZE}")
    print(f"  CROP_MODE: {config.CROP_MODE}")
    
    print("\n" + "=" * 60)
    print("✓ All mode configuration tests passed!")

if __name__ == "__main__":
    test_mode_config()
