#!/usr/bin/env python3
"""
MonkeyPatch script for vLLM DeepSeek-OCR custom modes and resolutions.

This script modifies vLLM package files to support different OCR modes:
- Tiny: 512×512 (64 vision tokens)
- Small: 640×640 (100 vision tokens)
- Base: 1024×1024 (256 vision tokens)
- Large: 1280×1280 (400 vision tokens)
- Gundam: n×640×640 + 1×1024×1024 (dynamic)

Usage:
    python monkeypatch_vllm.py --mode large
    python monkeypatch_vllm.py --base-size 1280 --image-size 1280 --crop-mode false
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


# Predefined modes
MODES = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}


def find_vllm_package() -> Optional[Path]:
    """Find the vLLM package installation directory."""
    try:
        import vllm
        vllm_path = Path(vllm.__file__).parent
        return vllm_path
    except ImportError:
        print("ERROR: vLLM package not found. Please install vLLM first.")
        return None


def find_processor_file(vllm_path: Path) -> Optional[Path]:
    """Find the DeepSeek OCR processor file in vLLM."""
    # Common locations
    possible_paths = [
        vllm_path / "transformers_utils" / "processors" / "deepseek_ocr.py",
        vllm_path / "model_executor" / "models" / "deepseek_ocr.py",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Search recursively
    for path in vllm_path.rglob("*deepseek_ocr.py"):
        if "processors" in str(path) or "transformers_utils" in str(path):
            return path
    
    return None


def find_model_file(vllm_path: Path) -> Optional[Path]:
    """Find the DeepSeek OCR model file in vLLM."""
    # Common location
    model_path = vllm_path / "model_executor" / "models" / "deepseek_ocr.py"
    
    if model_path.exists():
        return model_path
    
    # Search recursively
    for path in vllm_path.rglob("*deepseek_ocr.py"):
        if "model_executor" in str(path) and "models" in str(path):
            return path
    
    return None


def backup_file(file_path: Path) -> Path:
    """Create a backup of the file if it doesn't exist."""
    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
    
    if not backup_path.exists():
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"✓ Created backup: {backup_path}")
    
    return backup_path


def patch_processor_file(
    file_path: Path,
    base_size: int,
    image_size: int,
    crop_mode: bool
) -> bool:
    """Patch the processor file with new values."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Patch BASE_SIZE
        content = re.sub(
            r'BASE_SIZE\s*=\s*\d+',
            f'BASE_SIZE = {base_size}',
            content
        )
        
        # Patch IMAGE_SIZE
        content = re.sub(
            r'IMAGE_SIZE\s*=\s*\d+',
            f'IMAGE_SIZE = {image_size}',
            content
        )
        
        # Patch CROP_MODE
        crop_mode_str = "True" if crop_mode else "False"
        content = re.sub(
            r'CROP_MODE\s*=\s*(True|False)',
            f'CROP_MODE = {crop_mode_str}',
            content
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            print(f"WARNING: No changes made to {file_path}")
            return False
            
    except Exception as e:
        print(f"ERROR patching processor file: {e}")
        return False


def patch_model_file(file_path: Path, base_size: int) -> bool:
    """Patch the model file to use BASE_SIZE instead of vision_config.image_size."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Check if BASE_SIZE is imported or defined
        if 'from config import' not in content and 'BASE_SIZE' not in content:
            # Add BASE_SIZE constant at the top of the file (after imports)
            import_section_end = content.find('\n\nclass')
            if import_section_end == -1:
                import_section_end = content.find('\nclass')
            
            if import_section_end != -1:
                base_size_def = f'\n# MonkeyPatch: Custom BASE_SIZE\nBASE_SIZE = {base_size}\n'
                content = content[:import_section_end] + base_size_def + content[import_section_end:]
        
        # Replace vision_config.image_size with BASE_SIZE
        content = re.sub(
            r'base_size\s*=\s*self\.vision_config\.image_size',
            'base_size = BASE_SIZE',
            content
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            print(f"WARNING: No changes made to {file_path}")
            return False
            
    except Exception as e:
        print(f"ERROR patching model file: {e}")
        return False


def verify_patches(
    processor_file: Path,
    model_file: Optional[Path],
    base_size: int,
    image_size: int,
    crop_mode: bool
) -> bool:
    """Verify that patches were applied correctly."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    success = True
    
    # Verify processor file
    try:
        with open(processor_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\nProcessor file: {processor_file}")
        
        base_size_match = re.search(r'BASE_SIZE\s*=\s*(\d+)', content)
        image_size_match = re.search(r'IMAGE_SIZE\s*=\s*(\d+)', content)
        crop_mode_match = re.search(r'CROP_MODE\s*=\s*(True|False)', content)
        
        if base_size_match:
            found_base = int(base_size_match.group(1))
            status = "✓" if found_base == base_size else "✗"
            print(f"  {status} BASE_SIZE = {found_base} (expected: {base_size})")
            if found_base != base_size:
                success = False
        else:
            print(f"  ✗ BASE_SIZE not found")
            success = False
        
        if image_size_match:
            found_image = int(image_size_match.group(1))
            status = "✓" if found_image == image_size else "✗"
            print(f"  {status} IMAGE_SIZE = {found_image} (expected: {image_size})")
            if found_image != image_size:
                success = False
        else:
            print(f"  ✗ IMAGE_SIZE not found")
            success = False
        
        if crop_mode_match:
            found_crop = crop_mode_match.group(1) == "True"
            status = "✓" if found_crop == crop_mode else "✗"
            print(f"  {status} CROP_MODE = {found_crop} (expected: {crop_mode})")
            if found_crop != crop_mode:
                success = False
        else:
            print(f"  ✗ CROP_MODE not found")
            success = False
    
    except Exception as e:
        print(f"  ✗ Error reading processor file: {e}")
        success = False
    
    # Verify model file
    if model_file:
        try:
            with open(model_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"\nModel file: {model_file}")
            
            if 'base_size = BASE_SIZE' in content:
                print(f"  ✓ Using BASE_SIZE instead of vision_config.image_size")
            else:
                print(f"  ✗ Not using BASE_SIZE")
                success = False
        
        except Exception as e:
            print(f"  ✗ Error reading model file: {e}")
            success = False
    
    print("="*60)
    return success


def restore_backups(vllm_path: Path) -> bool:
    """Restore files from backups."""
    restored = False
    
    for backup_file in vllm_path.rglob("*.backup"):
        original_file = backup_file.with_suffix('')
        try:
            import shutil
            shutil.copy2(backup_file, original_file)
            print(f"✓ Restored: {original_file}")
            restored = True
        except Exception as e:
            print(f"✗ Failed to restore {original_file}: {e}")
    
    return restored


def main():
    parser = argparse.ArgumentParser(
        description="MonkeyPatch vLLM for DeepSeek-OCR custom modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined mode
  python monkeypatch_vllm.py --mode large
  
  # Custom configuration
  python monkeypatch_vllm.py --base-size 1280 --image-size 1280 --crop-mode false
  
  # Restore from backups
  python monkeypatch_vllm.py --restore
  
Available modes:
  tiny:   512×512   (64 vision tokens)
  small:  640×640   (100 vision tokens)
  base:   1024×1024 (256 vision tokens)
  large:  1280×1280 (400 vision tokens)
  gundam: 1024 base + 640 crops (dynamic)
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=list(MODES.keys()),
        help='Predefined mode to use'
    )
    parser.add_argument(
        '--base-size',
        type=int,
        help='Base size for global view (e.g., 1024, 1280)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        help='Image size for local crops (e.g., 640, 1280)'
    )
    parser.add_argument(
        '--crop-mode',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        help='Enable crop mode (true/false)'
    )
    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore files from backups'
    )
    parser.add_argument(
        '--vllm-path',
        type=Path,
        help='Path to vLLM package (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    # Find vLLM package
    vllm_path = args.vllm_path or find_vllm_package()
    if not vllm_path:
        sys.exit(1)
    
    print(f"vLLM package found: {vllm_path}")
    
    # Restore mode
    if args.restore:
        print("\nRestoring from backups...")
        if restore_backups(vllm_path):
            print("✓ Restoration complete")
        else:
            print("✗ No backups found or restoration failed")
        sys.exit(0)
    
    # Determine configuration
    if args.mode:
        config = MODES[args.mode]
        base_size = config["base_size"]
        image_size = config["image_size"]
        crop_mode = config["crop_mode"]
        print(f"\nUsing mode: {args.mode.upper()}")
    elif args.base_size and args.image_size and args.crop_mode is not None:
        base_size = args.base_size
        image_size = args.image_size
        crop_mode = args.crop_mode
        print(f"\nUsing custom configuration")
    else:
        print("ERROR: Either --mode or all of (--base-size, --image-size, --crop-mode) must be specified")
        parser.print_help()
        sys.exit(1)
    
    print(f"  BASE_SIZE: {base_size}")
    print(f"  IMAGE_SIZE: {image_size}")
    print(f"  CROP_MODE: {crop_mode}")
    
    # Find files to patch
    processor_file = find_processor_file(vllm_path)
    model_file = find_model_file(vllm_path)
    
    if not processor_file:
        print("\nERROR: Could not find DeepSeek OCR processor file in vLLM package")
        print("Please ensure vLLM is installed with DeepSeek-OCR support")
        sys.exit(1)
    
    print(f"\nFound processor file: {processor_file}")
    if model_file:
        print(f"Found model file: {model_file}")
    else:
        print("WARNING: Model file not found (may not be needed for newer vLLM versions)")
    
    # Create backups
    print("\nCreating backups...")
    backup_file(processor_file)
    if model_file:
        backup_file(model_file)
    
    # Apply patches
    print("\nApplying patches...")
    
    processor_success = patch_processor_file(processor_file, base_size, image_size, crop_mode)
    if processor_success:
        print(f"✓ Patched processor file")
    else:
        print(f"✗ Failed to patch processor file")
    
    model_success = True
    if model_file:
        model_success = patch_model_file(model_file, base_size)
        if model_success:
            print(f"✓ Patched model file")
        else:
            print(f"✗ Failed to patch model file")
    
    # Verify patches
    if verify_patches(processor_file, model_file, base_size, image_size, crop_mode):
        print("\n✓ All patches applied successfully!")
        print("\nYou can now start vLLM server with the custom configuration.")
    else:
        print("\n✗ Some patches failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
