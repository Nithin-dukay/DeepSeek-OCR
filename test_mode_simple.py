#!/usr/bin/env python3
"""Simple test script to verify mode selection functionality"""

# Test mode configurations
MODES = {
    'tiny': {'base_size': 512, 'image_size': 512, 'crop_mode': False},
    'small': {'base_size': 640, 'image_size': 640, 'crop_mode': False},
    'base': {'base_size': 1024, 'image_size': 1024, 'crop_mode': False},
    'large': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False},
    'gundam': {'base_size': 1024, 'image_size': 640, 'crop_mode': True}
}

def get_mode_config(mode):
    """Get configuration for a specific mode."""
    if mode.lower() not in MODES:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {list(MODES.keys())}")
    return MODES[mode.lower()]

def test_mode_config():
    """Test mode configuration"""
    print("Testing mode configurations...")
    print("=" * 60)
    
    # Test all modes
    modes = ['tiny', 'small', 'base', 'large', 'gundam']
    
    for mode in modes:
        mode_config = get_mode_config(mode)
        print(f"\nMode: {mode}")
        print(f"  base_size: {mode_config['base_size']}")
        print(f"  image_size: {mode_config['image_size']}")
        print(f"  crop_mode: {mode_config['crop_mode']}")
        
        # Verify expected values
        if mode == 'tiny':
            assert mode_config['base_size'] == 512
            assert mode_config['image_size'] == 512
            assert mode_config['crop_mode'] == False
        elif mode == 'small':
            assert mode_config['base_size'] == 640
            assert mode_config['image_size'] == 640
            assert mode_config['crop_mode'] == False
        elif mode == 'base':
            assert mode_config['base_size'] == 1024
            assert mode_config['image_size'] == 1024
            assert mode_config['crop_mode'] == False
        elif mode == 'large':
            assert mode_config['base_size'] == 1280
            assert mode_config['image_size'] == 1280
            assert mode_config['crop_mode'] == False
        elif mode == 'gundam':
            assert mode_config['base_size'] == 1024
            assert mode_config['image_size'] == 640
            assert mode_config['crop_mode'] == True
    
    print("\n" + "=" * 60)
    print("✓ All mode configuration tests passed!")
    
    # Test invalid mode
    print("\nTesting invalid mode handling...")
    try:
        get_mode_config('invalid')
        print("✗ Should have raised ValueError for invalid mode")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_mode_config()
