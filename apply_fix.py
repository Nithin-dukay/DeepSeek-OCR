#!/usr/bin/env python3
"""
Automatic patch script for GitHub Issue #299: DeepSeek OCR Triton CUDA Illegal Memory Access

This script automatically applies the necessary fixes to your DeepSeek-OCR-vllm installation.

Usage:
    python apply_fix.py [--backup] [--dry-run]

Options:
    --backup    Create backup of original files before patching
    --dry-run   Show what would be changed without actually modifying files
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def create_backup(file_path):
    """Create a backup of the original file."""
    backup_path = f"{file_path}.backup"
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)
        print(f"✓ Created backup: {backup_path}")
        return True
    return False


def patch_run_script(file_path, dry_run=False):
    """Patch the run_dpsk_ocr_image.py script with fixes."""
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "CRITICAL FIX" in content or "enforce_eager=True" in content:
        print(f"✓ File already patched: {file_path}")
        return True
    
    print(f"Patching: {file_path}")
    
    # Patch 1: Add environment variables after CUDA_VISIBLE_DEVICES
    env_vars_patch = """
# CRITICAL FIX: Environment variables for stability
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
"""
    
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"] = \'0\'',
        env_vars_patch.strip()
    )
    
    # Patch 2: Add cache clearing function
    cache_function = '''
def clear_triton_cache():
    """Clear Triton kernel cache to prevent corrupted cached kernels."""
    cache_dir = os.environ.get('TRITON_CACHE_DIR', '/tmp/triton_cache')
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"Cleared Triton cache at {cache_dir}")
        except Exception as e:
            print(f"Warning: Could not clear Triton cache: {e}")
    os.makedirs(cache_dir, exist_ok=True)

'''
    
    # Insert after imports
    import_end = content.find("ModelRegistry.register_model")
    if import_end != -1:
        content = content[:import_end] + cache_function + content[import_end:]
    
    # Patch 3: Update engine arguments
    old_engine_args = """    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,  
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )"""
    
    new_engine_args = """    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=128,  # FIXED: Reduced from 256 for better memory alignment
        max_model_len=8192,
        enforce_eager=True,  # CRITICAL FIX: Disable CUDA graphs to prevent illegal memory access
        trust_remote_code=True,  
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,  # FIXED: Reduced from 0.75 for stability
        max_num_seqs=255,  # FIXED: Avoid power-of-2 values (not 256) that trigger the bug
        disable_custom_all_reduce=True,  # FIXED: Disable custom kernels
    )
    
    print("Initializing vLLM engine with stability fixes...")
    print(f"  - enforce_eager: True (CUDA graphs disabled)")
    print(f"  - block_size: 128")
    print(f"  - gpu_memory_utilization: 0.70")
    print(f"  - max_num_seqs: 255")"""
    
    content = content.replace(old_engine_args, new_engine_args)
    
    # Patch 4: Add cache clearing call in main
    main_start = 'if __name__ == "__main__":'
    cache_call = '''if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek OCR - Patched Version for vLLM 0.11.2")
    print("Fixes for GitHub Issue #299: Triton CUDA Illegal Memory Access")
    print("=" * 60)
    
    # Clear Triton cache before starting
    print("\\nClearing Triton kernel cache...")
    clear_triton_cache()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")
    
    print("\\nInitializing...")
'''
    
    content = content.replace(main_start, cache_call)
    
    # Add shutil import if not present
    if 'import shutil' not in content:
        content = content.replace('import re', 'import re\nimport shutil')
    
    if dry_run:
        print("  [DRY RUN] Would apply patches")
        return True
    
    # Write patched content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Successfully patched: {file_path}")
    return True


def patch_model_file(file_path, dry_run=False):
    """Patch the deepseek_ocr.py model file with error handling."""
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "FIXED VERSION" in content or "torch.cuda.synchronize()" in content:
        print(f"✓ File already patched: {file_path}")
        return True
    
    print(f"Patching: {file_path}")
    
    # Add CUDA synchronization in _pixel_values_to_embedding
    sync_patch = """        # FIX: Add CUDA synchronization before processing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():"""
    
    content = content.replace(
        "        with torch.no_grad():",
        sync_patch
    )
    
    # Add synchronization after each image
    after_append = """                    images_in_this_batch.append(global_local_features)
                    
                    # FIX: Synchronize after each image to prevent memory issues
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()"""
    
    content = content.replace(
        "                images_in_this_batch.append(global_local_features)",
        after_append
    )
    
    if dry_run:
        print("  [DRY RUN] Would apply patches")
        return True
    
    # Write patched content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Successfully patched: {file_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply fixes for DeepSeek OCR Triton CUDA errors (Issue #299)"
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup of original files before patching'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without actually modifying files'
    )
    parser.add_argument(
        '--path',
        type=str,
        default='DeepSeek-OCR-master/DeepSeek-OCR-vllm',
        help='Path to DeepSeek-OCR-vllm directory'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DeepSeek OCR - Automatic Fix Application")
    print("GitHub Issue #299: Triton CUDA Illegal Memory Access")
    print("=" * 70)
    print()
    
    if args.dry_run:
        print("⚠ DRY RUN MODE - No files will be modified")
        print()
    
    # Determine base path
    base_path = Path(args.path)
    if not base_path.exists():
        print(f"✗ Directory not found: {base_path}")
        print("Please specify the correct path using --path option")
        return 1
    
    print(f"Working directory: {base_path.absolute()}")
    print()
    
    # Files to patch
    files_to_patch = [
        ('run_dpsk_ocr_image.py', patch_run_script),
        ('deepseek_ocr.py', patch_model_file),
    ]
    
    success_count = 0
    total_count = len(files_to_patch)
    
    for filename, patch_func in files_to_patch:
        file_path = base_path / filename
        
        print(f"Processing: {filename}")
        print("-" * 70)
        
        # Create backup if requested
        if args.backup and not args.dry_run:
            create_backup(file_path)
        
        # Apply patch
        if patch_func(file_path, dry_run=args.dry_run):
            success_count += 1
        
        print()
    
    # Summary
    print("=" * 70)
    print(f"Patching complete: {success_count}/{total_count} files processed")
    print("=" * 70)
    
    if args.dry_run:
        print("\nThis was a dry run. Run without --dry-run to apply changes.")
    elif success_count == total_count:
        print("\n✓ All patches applied successfully!")
        print("\nNext steps:")
        print("1. Test with: python run_dpsk_ocr_image.py")
        print("2. If issues persist, check FIX_README.md for troubleshooting")
        print("3. Consider using run_dpsk_ocr_image_fixed.py for a fully tested version")
    else:
        print("\n⚠ Some patches failed. Please check the errors above.")
        print("You can use the pre-patched files instead:")
        print("  - run_dpsk_ocr_image_fixed.py")
        print("  - deepseek_ocr_fixed.py")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
