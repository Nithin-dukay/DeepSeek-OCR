#!/usr/bin/env python3
"""
Apply fix for GitHub Issue #299: DeepSeek OCR Triton Error [CUDA] Illegal Memory Access

This script patches the existing DeepSeek-OCR-vllm files to fix the Triton/CUDA
illegal memory access error that occurs with vLLM 0.11.2.

Usage:
    python apply_fix_issue_299.py [--dry-run]
    
Options:
    --dry-run    Show what would be changed without actually modifying files
"""

import os
import sys
import shutil
import re
from pathlib import Path

def backup_file(filepath):
    """Create a backup of the original file"""
    backup_path = f"{filepath}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"✓ Backed up: {filepath} -> {backup_path}")
    return backup_path

def patch_run_dpsk_ocr_image(filepath, dry_run=False):
    """Patch run_dpsk_ocr_image.py with fixes"""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Patching {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    changes = []
    
    # Add environment variable for multi-node setups
    if "os.environ['VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE']" not in content:
        old_pattern = r"(os\.environ\['VLLM_USE_V1'\] = '0')"
        new_text = r"\1\n\n# For multi-node setups (optional but recommended)\nos.environ['VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE'] = 'shm'"
        content = re.sub(old_pattern, new_text, content)
        changes.append("Added VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE environment variable")
    
    # Fix engine_args configuration
    engine_args_pattern = r'engine_args = AsyncEngineArgs\((.*?)\)'
    
    if 'enforce_eager=False' in content or 'gpu_memory_utilization=0.75' not in content:
        # Replace the entire engine_args block
        old_engine_args = re.search(
            r'engine_args = AsyncEngineArgs\((.*?)\)',
            content,
            re.DOTALL
        )
        
        if old_engine_args:
            new_engine_args = """engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=True,  # CRITICAL FIX: Disable CUDA graphs
        trust_remote_code=True,  
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,  # CRITICAL FIX: Reduced memory utilization
        max_num_seqs=255,  # CRITICAL FIX: Avoid power-of-2 value (not 256)
    )"""
            
            content = content.replace(old_engine_args.group(0), new_engine_args)
            changes.append("Updated engine_args with critical fixes")
    
    if changes:
        if not dry_run:
            backup_file(filepath)
            with open(filepath, 'w') as f:
                f.write(content)
        
        for change in changes:
            print(f"  ✓ {change}")
        return True
    else:
        print("  ℹ No changes needed (already patched)")
        return False

def patch_config_py(filepath, dry_run=False):
    """Patch config.py with safer defaults"""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Patching {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    changes = []
    
    # Fix MAX_CROPS
    if 'MAX_CROPS= 6' in content or 'MAX_CROPS = 6' in content:
        content = re.sub(
            r'MAX_CROPS\s*=\s*6\s*#.*',
            'MAX_CROPS= 4  # max:9; Reduced from 6 to prevent CUDA memory issues',
            content
        )
        changes.append("Reduced MAX_CROPS from 6 to 4")
    
    # Fix MAX_CONCURRENCY
    if 'MAX_CONCURRENCY = 100' in content or 'MAX_CONCURRENCY= 100' in content:
        content = re.sub(
            r'MAX_CONCURRENCY\s*=\s*100\s*#.*',
            'MAX_CONCURRENCY = 50  # Reduced from 100 for better memory management',
            content
        )
        changes.append("Reduced MAX_CONCURRENCY from 100 to 50")
    
    if changes:
        if not dry_run:
            backup_file(filepath)
            with open(filepath, 'w') as f:
                f.write(content)
        
        for change in changes:
            print(f"  ✓ {change}")
        return True
    else:
        print("  ℹ No changes needed (already patched)")
        return False

def patch_run_dpsk_ocr_pdf(filepath, dry_run=False):
    """Patch run_dpsk_ocr_pdf.py with fixes"""
    if not os.path.exists(filepath):
        print(f"\n⚠ Skipping {filepath} (file not found)")
        return False
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Patching {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    changes = []
    
    # Similar fixes as image script
    if 'enforce_eager=False' in content or 'gpu_memory_utilization=0.9' in content:
        content = re.sub(
            r'enforce_eager\s*=\s*False',
            'enforce_eager=True  # CRITICAL FIX: Disable CUDA graphs',
            content
        )
        content = re.sub(
            r'gpu_memory_utilization\s*=\s*0\.9',
            'gpu_memory_utilization=0.75  # CRITICAL FIX: Reduced memory utilization',
            content
        )
        changes.append("Updated engine configuration with critical fixes")
    
    if changes:
        if not dry_run:
            backup_file(filepath)
            with open(filepath, 'w') as f:
                f.write(content)
        
        for change in changes:
            print(f"  ✓ {change}")
        return True
    else:
        print("  ℹ No changes needed (already patched)")
        return False

def patch_run_dpsk_ocr_eval_batch(filepath, dry_run=False):
    """Patch run_dpsk_ocr_eval_batch.py with fixes"""
    if not os.path.exists(filepath):
        print(f"\n⚠ Skipping {filepath} (file not found)")
        return False
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Patching {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    changes = []
    
    # Similar fixes as other scripts
    if 'enforce_eager=False' in content or 'gpu_memory_utilization=0.9' in content:
        content = re.sub(
            r'enforce_eager\s*=\s*False',
            'enforce_eager=True  # CRITICAL FIX: Disable CUDA graphs',
            content
        )
        content = re.sub(
            r'gpu_memory_utilization\s*=\s*0\.9',
            'gpu_memory_utilization=0.75  # CRITICAL FIX: Reduced memory utilization',
            content
        )
        changes.append("Updated engine configuration with critical fixes")
    
    if changes:
        if not dry_run:
            backup_file(filepath)
            with open(filepath, 'w') as f:
                f.write(content)
        
        for change in changes:
            print(f"  ✓ {change}")
        return True
    else:
        print("  ℹ No changes needed (already patched)")
        return False

def main():
    dry_run = '--dry-run' in sys.argv
    
    print("=" * 70)
    print("DeepSeek OCR Issue #299 Fix Application Script")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠ DRY RUN MODE - No files will be modified\n")
    
    # Find the DeepSeek-OCR-vllm directory
    base_dir = Path(__file__).parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
    
    if not base_dir.exists():
        print(f"\n❌ Error: Directory not found: {base_dir}")
        print("Please run this script from the repository root directory.")
        sys.exit(1)
    
    print(f"\nTarget directory: {base_dir}\n")
    
    # Apply patches
    files_patched = 0
    
    # Patch config.py
    config_file = base_dir / "config.py"
    if config_file.exists():
        if patch_config_py(str(config_file), dry_run):
            files_patched += 1
    
    # Patch run_dpsk_ocr_image.py
    image_file = base_dir / "run_dpsk_ocr_image.py"
    if image_file.exists():
        if patch_run_dpsk_ocr_image(str(image_file), dry_run):
            files_patched += 1
    
    # Patch run_dpsk_ocr_pdf.py
    pdf_file = base_dir / "run_dpsk_ocr_pdf.py"
    if pdf_file.exists():
        if patch_run_dpsk_ocr_pdf(str(pdf_file), dry_run):
            files_patched += 1
    
    # Patch run_dpsk_ocr_eval_batch.py
    batch_file = base_dir / "run_dpsk_ocr_eval_batch.py"
    if batch_file.exists():
        if patch_run_dpsk_ocr_eval_batch(str(batch_file), dry_run):
            files_patched += 1
    
    # Summary
    print("\n" + "=" * 70)
    if dry_run:
        print(f"DRY RUN COMPLETE - {files_patched} file(s) would be modified")
        print("\nRun without --dry-run to apply changes")
    else:
        print(f"PATCHING COMPLETE - {files_patched} file(s) modified")
        print("\n✓ Backup files created with .backup extension")
        print("✓ Original files can be restored from backups if needed")
    print("=" * 70)
    
    print("\n📋 Next Steps:")
    print("1. Review the changes in the patched files")
    print("2. Test with a problematic image that previously caused crashes")
    print("3. Monitor GPU memory usage with nvidia-smi")
    print("4. If issues persist, try further reducing MAX_CROPS in config.py")
    print("\n📖 See ISSUE_299_FIX.md for detailed documentation")

if __name__ == "__main__":
    main()
