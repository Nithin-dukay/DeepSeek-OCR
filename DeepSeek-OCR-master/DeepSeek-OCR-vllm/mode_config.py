"""
Mode Configuration Helper for DeepSeek-OCR vLLM

This module provides mode presets for different OCR configurations.
Supports: Tiny, Small, Base, Large, and Gundam modes.
"""

from typing import Dict, Literal

# Type definition for supported modes
ModeType = Literal["tiny", "small", "base", "large", "gundam"]


class OCRModeConfig:
    """Configuration class for OCR modes"""
    
    # Mode presets based on official documentation
    MODES: Dict[str, Dict[str, any]] = {
        "tiny": {
            "base_size": 512,
            "image_size": 512,
            "crop_mode": False,
            "description": "Tiny mode: 512×512 (64 vision tokens)",
            "vision_tokens": 64,
        },
        "small": {
            "base_size": 640,
            "image_size": 640,
            "crop_mode": False,
            "description": "Small mode: 640×640 (100 vision tokens)",
            "vision_tokens": 100,
        },
        "base": {
            "base_size": 1024,
            "image_size": 1024,
            "crop_mode": False,
            "description": "Base mode: 1024×1024 (256 vision tokens)",
            "vision_tokens": 256,
        },
        "large": {
            "base_size": 1280,
            "image_size": 1280,
            "crop_mode": False,
            "description": "Large mode: 1280×1280 (400 vision tokens)",
            "vision_tokens": 400,
        },
        "gundam": {
            "base_size": 1024,
            "image_size": 640,
            "crop_mode": True,
            "description": "Gundam mode: n×640×640 + 1×1024×1024 (dynamic resolution)",
            "vision_tokens": "dynamic",
        },
    }
    
    @classmethod
    def get_mode_config(cls, mode: str) -> Dict[str, any]:
        """
        Get configuration for a specific mode.
        
        Args:
            mode: Mode name (tiny, small, base, large, gundam)
            
        Returns:
            Dictionary containing base_size, image_size, and crop_mode
            
        Raises:
            ValueError: If mode is not supported
        """
        mode_lower = mode.lower()
        if mode_lower not in cls.MODES:
            available_modes = ", ".join(cls.MODES.keys())
            raise ValueError(
                f"Unsupported mode: '{mode}'. "
                f"Available modes: {available_modes}"
            )
        return cls.MODES[mode_lower].copy()
    
    @classmethod
    def list_modes(cls) -> Dict[str, str]:
        """
        List all available modes with descriptions.
        
        Returns:
            Dictionary mapping mode names to descriptions
        """
        return {
            mode: config["description"] 
            for mode, config in cls.MODES.items()
        }
    
    @classmethod
    def get_base_size(cls, mode: str) -> int:
        """Get base_size for a mode"""
        return cls.get_mode_config(mode)["base_size"]
    
    @classmethod
    def get_image_size(cls, mode: str) -> int:
        """Get image_size for a mode"""
        return cls.get_mode_config(mode)["image_size"]
    
    @classmethod
    def get_crop_mode(cls, mode: str) -> bool:
        """Get crop_mode for a mode"""
        return cls.get_mode_config(mode)["crop_mode"]
    
    @classmethod
    def print_modes(cls):
        """Print all available modes with descriptions"""
        print("Available OCR Modes:")
        print("=" * 60)
        for mode, config in cls.MODES.items():
            print(f"  {mode.upper():8} - {config['description']}")
            print(f"           base_size={config['base_size']}, "
                  f"image_size={config['image_size']}, "
                  f"crop_mode={config['crop_mode']}")
        print("=" * 60)


def get_mode_params(mode: str) -> tuple:
    """
    Convenience function to get mode parameters as a tuple.
    
    Args:
        mode: Mode name (tiny, small, base, large, gundam)
        
    Returns:
        Tuple of (base_size, image_size, crop_mode)
    """
    config = OCRModeConfig.get_mode_config(mode)
    return config["base_size"], config["image_size"], config["crop_mode"]


if __name__ == "__main__":
    # Example usage and testing
    OCRModeConfig.print_modes()
    
    print("\nExample: Getting Gundam mode configuration:")
    gundam_config = OCRModeConfig.get_mode_config("gundam")
    print(f"  base_size: {gundam_config['base_size']}")
    print(f"  image_size: {gundam_config['image_size']}")
    print(f"  crop_mode: {gundam_config['crop_mode']}")
    
    print("\nExample: Using convenience function:")
    base_size, image_size, crop_mode = get_mode_params("small")
    print(f"  Small mode: base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")
