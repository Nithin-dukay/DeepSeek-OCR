#!/usr/bin/env python3
"""
Fix DeepSeek-OCR Model Config

This utility fixes the architecture name typo in the DeepSeek-OCR model's config.json file.
The typo is 'DeepseekOCRForCausallM' (double 'l') which should be 'DeepseekOCRForCausalLM'.

Usage:
    python fix_model_config.py [model_path]

Arguments:
    model_path: Path to the model directory or HuggingFace cache directory
                If not provided, will attempt to find the model in HuggingFace cache

Examples:
    # Fix config in a local model directory
    python fix_model_config.py /path/to/DeepSeek-OCR
    
    # Fix config in HuggingFace cache (auto-detect)
    python fix_model_config.py
    
    # Fix config for a specific HuggingFace cached model
    python fix_model_config.py ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR
"""

import json
import os
import sys
import argparse
from pathlib import Path
import shutil


def find_huggingface_cache():
    """Find the HuggingFace cache directory."""
    # Check common HuggingFace cache locations
    possible_paths = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface"),
        os.environ.get("HF_HOME", ""),
        os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    
    return None


def find_model_config(model_path=None):
    """Find the config.json file for DeepSeek-OCR model."""
    if model_path:
        # Check if it's a direct path to config.json
        if os.path.isfile(model_path) and model_path.endswith("config.json"):
            return model_path
        
        # Check if it's a model directory
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            return config_path
        
        # Check for snapshots directory (HuggingFace cache structure)
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_dir):
            # Find the latest snapshot
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                latest_snapshot = sorted(snapshots)[-1]
                config_path = os.path.join(snapshots_dir, latest_snapshot, "config.json")
                if os.path.exists(config_path):
                    return config_path
    
    # Try to find in HuggingFace cache
    cache_dir = find_huggingface_cache()
    if cache_dir:
        # Look for DeepSeek-OCR model
        model_patterns = [
            "models--deepseek-ai--DeepSeek-OCR",
            "deepseek-ai--DeepSeek-OCR",
        ]
        
        for pattern in model_patterns:
            model_dir = os.path.join(cache_dir, pattern)
            if os.path.exists(model_dir):
                # Check snapshots
                snapshots_dir = os.path.join(model_dir, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        latest_snapshot = sorted(snapshots)[-1]
                        config_path = os.path.join(snapshots_dir, latest_snapshot, "config.json")
                        if os.path.exists(config_path):
                            return config_path
    
    return None


def fix_config(config_path, backup=True):
    """Fix the architecture name in the config.json file."""
    print(f"Reading config from: {config_path}")
    
    # Read the config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        return False
    
    # Check if fix is needed
    architectures = config.get("architectures", [])
    needs_fix = False
    
    print(f"Current architectures: {architectures}")
    
    # Fix the typo
    fixed_architectures = []
    for arch in architectures:
        if arch == "DeepseekOCRForCausallM":  # Typo with double 'l'
            fixed_architectures.append("DeepseekOCRForCausalLM")  # Correct with single 'l'
            needs_fix = True
            print(f"  Found typo: {arch} -> DeepseekOCRForCausalLM")
        else:
            fixed_architectures.append(arch)
    
    if not needs_fix:
        print("✓ No fix needed - config is already correct!")
        return True
    
    # Create backup if requested
    if backup:
        backup_path = config_path + ".backup"
        try:
            shutil.copy2(config_path, backup_path)
            print(f"✓ Backup created: {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
    
    # Update config
    config["architectures"] = fixed_architectures
    
    # Write the fixed config
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✓ Config fixed successfully!")
        print(f"  New architectures: {fixed_architectures}")
        return True
    except Exception as e:
        print(f"Error writing config file: {e}")
        if backup and os.path.exists(backup_path):
            print(f"Restoring from backup...")
            shutil.copy2(backup_path, config_path)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix architecture name typo in DeepSeek-OCR config.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to model directory or config.json (auto-detect if not provided)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup of the original config"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepSeek-OCR Config Fix Utility")
    print("=" * 60)
    
    # Find the config file
    config_path = find_model_config(args.model_path)
    
    if not config_path:
        print("\n❌ Could not find config.json for DeepSeek-OCR model")
        print("\nTroubleshooting:")
        print("1. Specify the model path explicitly:")
        print("   python fix_model_config.py /path/to/model")
        print("2. Make sure the model is downloaded:")
        print("   huggingface-cli download deepseek-ai/DeepSeek-OCR")
        print("3. Check HuggingFace cache location:")
        cache_dir = find_huggingface_cache()
        if cache_dir:
            print(f"   Cache directory: {cache_dir}")
        else:
            print("   Cache directory not found")
        sys.exit(1)
    
    # Fix the config
    success = fix_config(config_path, backup=not args.no_backup)
    
    if success:
        print("\n✓ Done! You can now use the model with vLLM")
        print("\nNext steps:")
        print("1. Use the serve_deepseek_ocr.py script:")
        print("   python serve_deepseek_ocr.py")
        print("2. Or use vLLM directly:")
        print("   vllm serve deepseek-ai/DeepSeek-OCR --trust-remote-code")
        sys.exit(0)
    else:
        print("\n❌ Failed to fix config")
        sys.exit(1)


if __name__ == "__main__":
    main()
