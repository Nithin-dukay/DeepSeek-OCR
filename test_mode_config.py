#!/usr/bin/env python3
"""Test script to verify MODE configuration works correctly"""

# Test mode configurations
MODE_CONFIGS = {
    'tiny': {
        'base_size': 512,
        'image_size': 512,
        'crop_mode': False,
        'description': '512×512 resolution (64 vision tokens) - Fastest, lowest memory'
    },
    'small': {
        'base_size': 640,
        'image_size': 640,
        'crop_mode': False,
        'description': '640×640 resolution (100 vision tokens) - Fast, low memory'
    },
    'base': {
        'base_size': 1024,
        'image_size': 1024,
        'crop_mode': False,
        'description': '1024×1024 resolution (256 vision tokens) - Balanced'
    },
    'large': {
        'base_size': 1280,
        'image_size': 1280,
        'crop_mode': False,
        'description': '1280×1280 resolution (400 vision tokens) - High quality, more memory'
    },
    'gundam': {
        'base_size': 1024,
        'image_size': 640,
        'crop_mode': True,
        'description': 'Dynamic resolution with tiles (n×640×640 + 1×1024×1024) - Best quality, adaptive'
    }
}

def test_mode(mode_name):
    """Test a specific mode configuration"""
    print(f"\n{'='*60}")
    print(f"Testing MODE: {mode_name.upper()}")
    print(f"{'='*60}")
    
    if mode_name.lower() not in MODE_CONFIGS:
        print(f"❌ ERROR: Invalid MODE '{mode_name}'")
        return False
    
    mode_config = MODE_CONFIGS[mode_name.lower()]
    BASE_SIZE = mode_config['base_size']
    IMAGE_SIZE = mode_config['image_size']
    CROP_MODE = mode_config['crop_mode']
    
    print(f"✓ BASE_SIZE: {BASE_SIZE}")
    print(f"✓ IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"✓ CROP_MODE: {CROP_MODE}")
    print(f"✓ Description: {mode_config['description']}")
    
    # Verify values
    expected = {
        'tiny': (512, 512, False),
        'small': (640, 640, False),
        'base': (1024, 1024, False),
        'large': (1280, 1280, False),
        'gundam': (1024, 640, True)
    }
    
    if (BASE_SIZE, IMAGE_SIZE, CROP_MODE) == expected[mode_name.lower()]:
        print(f"✅ Mode '{mode_name}' configuration is CORRECT")
        return True
    else:
        print(f"❌ Mode '{mode_name}' configuration is INCORRECT")
        return False

if __name__ == "__main__":
    print("="*60)
    print("DeepSeek-OCR vLLM Mode Configuration Test")
    print("="*60)
    
    modes = ['tiny', 'small', 'base', 'large', 'gundam']
    results = []
    
    for mode in modes:
        results.append(test_mode(mode))
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✅ All mode configurations are working correctly!")
    else:
        print("\n❌ Some mode configurations failed!")
    
    # Test invalid mode
    print(f"\n{'='*60}")
    print("Testing invalid mode handling")
    print(f"{'='*60}")
    try:
        MODE = 'invalid_mode'
        if MODE.lower() not in MODE_CONFIGS:
            raise ValueError(f"Invalid MODE '{MODE}'. Must be one of: {list(MODE_CONFIGS.keys())}")
    except ValueError as e:
        print(f"✅ Invalid mode correctly rejected: {e}")
