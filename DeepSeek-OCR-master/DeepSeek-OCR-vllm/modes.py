"""
Mode configurations for DeepSeek-OCR model.

This module provides predefined mode configurations for different use cases:
- Tiny: 512×512 (64 vision tokens) - Fastest, lowest memory
- Small: 640×640 (100 vision tokens) - Fast, low memory
- Base: 1024×1024 (256 vision tokens) - Balanced performance
- Large: 1280×1280 (400 vision tokens) - High quality
- Gundam: Dynamic resolution with crops (variable tokens) - Best quality for documents
"""

from typing import Dict, Tuple, Literal

# Mode type definition
ModeType = Literal["Tiny", "Small", "Base", "Large", "Gundam"]

# Mode configurations: (base_size, image_size, crop_mode)
MODE_CONFIGS: Dict[str, Tuple[int, int, bool]] = {
    "Tiny": (512, 512, False),
    "Small": (640, 640, False),
    "Base": (1024, 1024, False),
    "Large": (1280, 1280, False),
    "Gundam": (1024, 640, True),
}

# Vision token counts for each mode (approximate for non-crop modes)
MODE_VISION_TOKENS: Dict[str, int] = {
    "Tiny": 64,
    "Small": 100,
    "Base": 256,
    "Large": 400,
    "Gundam": -1,  # Variable, depends on image size
}


def get_mode_config(mode: str) -> Tuple[int, int, bool]:
    """
    Get the configuration for a specific mode.
    
    Args:
        mode: Mode name (Tiny, Small, Base, Large, or Gundam)
        
    Returns:
        Tuple of (base_size, image_size, crop_mode)
        
    Raises:
        ValueError: If mode is not recognized
    """
    mode = mode.capitalize()  # Normalize case
    if mode not in MODE_CONFIGS:
        available_modes = ", ".join(MODE_CONFIGS.keys())
        raise ValueError(
            f"Unknown mode: {mode}. Available modes: {available_modes}"
        )
    return MODE_CONFIGS[mode]


def get_available_modes() -> list:
    """
    Get list of available mode names.
    
    Returns:
        List of mode names
    """
    return list(MODE_CONFIGS.keys())


def get_mode_info(mode: str) -> Dict[str, any]:
    """
    Get detailed information about a mode.
    
    Args:
        mode: Mode name
        
    Returns:
        Dictionary with mode configuration details
    """
    mode = mode.capitalize()
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}")
    
    base_size, image_size, crop_mode = MODE_CONFIGS[mode]
    vision_tokens = MODE_VISION_TOKENS[mode]
    
    return {
        "mode": mode,
        "base_size": base_size,
        "image_size": image_size,
        "crop_mode": crop_mode,
        "vision_tokens": vision_tokens if vision_tokens > 0 else "Variable",
        "description": _get_mode_description(mode),
    }


def _get_mode_description(mode: str) -> str:
    """Get a human-readable description of the mode."""
    descriptions = {
        "Tiny": "Fastest processing, lowest memory usage (512×512)",
        "Small": "Fast processing, low memory usage (640×640)",
        "Base": "Balanced performance and quality (1024×1024)",
        "Large": "High quality, higher memory usage (1280×1280)",
        "Gundam": "Best quality with dynamic resolution and cropping (1024 base + 640 crops)",
    }
    return descriptions.get(mode, "")


def print_available_modes():
    """Print information about all available modes."""
    print("Available DeepSeek-OCR Modes:")
    print("=" * 80)
    for mode in MODE_CONFIGS.keys():
        info = get_mode_info(mode)
        print(f"\n{info['mode']}:")
        print(f"  Base Size: {info['base_size']}")
        print(f"  Image Size: {info['image_size']}")
        print(f"  Crop Mode: {info['crop_mode']}")
        print(f"  Vision Tokens: {info['vision_tokens']}")
        print(f"  Description: {info['description']}")
    print("=" * 80)


if __name__ == "__main__":
    # Demo usage
    print_available_modes()
