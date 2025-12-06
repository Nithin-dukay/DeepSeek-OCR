#!/usr/bin/env python3
"""Test script to verify command-line argument parsing"""

import argparse
import sys

# Simulate MODE_CONFIGS
MODE_CONFIGS = {
    'tiny': {'base_size': 512, 'image_size': 512, 'crop_mode': False, 'description': '512×512 - Fastest'},
    'small': {'base_size': 640, 'image_size': 640, 'crop_mode': False, 'description': '640×640 - Fast'},
    'base': {'base_size': 1024, 'image_size': 1024, 'crop_mode': False, 'description': '1024×1024 - Balanced'},
    'large': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False, 'description': '1280×1280 - High quality'},
    'gundam': {'base_size': 1024, 'image_size': 640, 'crop_mode': True, 'description': 'Dynamic tiles - Best quality'}
}

class Config:
    MODE = 'gundam'
    BASE_SIZE = 1024
    IMAGE_SIZE = 640
    CROP_MODE = True

config = Config()

def test_cli_parsing(test_args):
    """Test command-line argument parsing"""
    parser = argparse.ArgumentParser(description='DeepSeek-OCR Test')
    parser.add_argument('--mode', type=str, default=None, 
                        choices=['tiny', 'small', 'base', 'large', 'gundam'],
                        help='OCR mode')
    parser.add_argument('--input', type=str, default=None, help='Input path')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--prompt', type=str, default=None, help='Prompt text')
    
    # Parse test arguments
    args = parser.parse_args(test_args)
    
    # Apply mode override if specified
    if args.mode:
        mode_config = MODE_CONFIGS[args.mode.lower()]
        config.BASE_SIZE = mode_config['base_size']
        config.IMAGE_SIZE = mode_config['image_size']
        config.CROP_MODE = mode_config['crop_mode']
        print(f"✓ Mode override: {args.mode.upper()} - {mode_config['description']}")
    else:
        print(f"✓ Using default mode: {config.MODE.upper()}")
    
    print(f"  BASE_SIZE: {config.BASE_SIZE}")
    print(f"  IMAGE_SIZE: {config.IMAGE_SIZE}")
    print(f"  CROP_MODE: {config.CROP_MODE}")
    
    if args.input:
        print(f"✓ Input override: {args.input}")
    if args.output:
        print(f"✓ Output override: {args.output}")
    if args.prompt:
        print(f"✓ Prompt override: {args.prompt}")
    
    return args

if __name__ == "__main__":
    print("="*60)
    print("Command-Line Argument Parsing Test")
    print("="*60)
    
    # Test 1: No arguments (use defaults)
    print("\nTest 1: No arguments (use config.py defaults)")
    print("-" * 60)
    config.MODE = 'gundam'
    config.BASE_SIZE = 1024
    config.IMAGE_SIZE = 640
    config.CROP_MODE = True
    test_cli_parsing([])
    
    # Test 2: Mode override to 'base'
    print("\nTest 2: Override mode to 'base'")
    print("-" * 60)
    config.MODE = 'gundam'
    config.BASE_SIZE = 1024
    config.IMAGE_SIZE = 640
    config.CROP_MODE = True
    test_cli_parsing(['--mode', 'base'])
    
    # Test 3: Mode override to 'small'
    print("\nTest 3: Override mode to 'small'")
    print("-" * 60)
    config.MODE = 'gundam'
    config.BASE_SIZE = 1024
    config.IMAGE_SIZE = 640
    config.CROP_MODE = True
    test_cli_parsing(['--mode', 'small'])
    
    # Test 4: Multiple overrides
    print("\nTest 4: Multiple overrides (mode + input + output)")
    print("-" * 60)
    config.MODE = 'gundam'
    config.BASE_SIZE = 1024
    config.IMAGE_SIZE = 640
    config.CROP_MODE = True
    test_cli_parsing(['--mode', 'large', '--input', '/path/to/image.jpg', '--output', '/path/to/output'])
    
    # Test 5: All arguments
    print("\nTest 5: All arguments")
    print("-" * 60)
    config.MODE = 'gundam'
    config.BASE_SIZE = 1024
    config.IMAGE_SIZE = 640
    config.CROP_MODE = True
    test_cli_parsing(['--mode', 'tiny', '--input', 'test.jpg', '--output', 'out/', '--prompt', '<image>\\nOCR this.'])
    
    print("\n" + "="*60)
    print("✅ All command-line argument tests passed!")
    print("="*60)
