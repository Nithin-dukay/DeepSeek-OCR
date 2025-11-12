#!/usr/bin/env python3
"""
Test script to verify the fix for GitHub Issue #244

This script tests:
1. Model registration logic
2. Config fix utility
3. Architecture name handling
"""

import json
import os
import sys
import tempfile
from pathlib import Path


def test_config_fix():
    """Test the config fix utility."""
    print("=" * 60)
    print("Testing Config Fix Utility")
    print("=" * 60)
    
    # Create a temporary config file with the typo
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        
        # Create config with typo
        config_with_typo = {
            "architectures": ["DeepseekOCRForCausallM"],  # Typo: double 'l'
            "model_type": "deepseek_vl_v2",
            "other_config": "value"
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_with_typo, f, indent=2)
        
        print(f"✓ Created test config with typo: {config_path}")
        print(f"  Original: {config_with_typo['architectures']}")
        
        # Import and use the fix function
        sys.path.insert(0, '/vercel/sandbox')
        from fix_model_config import fix_config
        
        # Fix the config
        success = fix_config(config_path, backup=False)
        
        if not success:
            print("❌ Config fix failed")
            return False
        
        # Verify the fix
        with open(config_path, 'r') as f:
            fixed_config = json.load(f)
        
        expected_arch = "DeepseekOCRForCausalLM"
        actual_arch = fixed_config['architectures'][0]
        
        if actual_arch == expected_arch:
            print(f"✓ Config fixed correctly: {actual_arch}")
            return True
        else:
            print(f"❌ Config fix incorrect: expected {expected_arch}, got {actual_arch}")
            return False


def test_registration_module():
    """Test the registration module."""
    print("\n" + "=" * 60)
    print("Testing Model Registration Module")
    print("=" * 60)
    
    try:
        # Check if the registration module exists
        reg_path = "/vercel/sandbox/register_deepseek_ocr.py"
        if not os.path.exists(reg_path):
            print(f"❌ Registration module not found: {reg_path}")
            return False
        
        print(f"✓ Registration module exists: {reg_path}")
        
        # Check the module content
        with open(reg_path, 'r') as f:
            content = f.read()
        
        # Verify key components
        checks = [
            ("ModelRegistry import", "from vllm.model_executor.models.registry import ModelRegistry"),
            ("Model import", "from deepseek_ocr import DeepseekOCRForCausalLM"),
            ("Correct registration", 'register_model("DeepseekOCRForCausalLM"'),
            ("Typo registration", 'register_model("DeepseekOCRForCausallM"'),
        ]
        
        all_passed = True
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✓ {check_name}: Found")
            else:
                print(f"❌ {check_name}: Not found")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing registration module: {e}")
        return False


def test_serve_script():
    """Test the serve script."""
    print("\n" + "=" * 60)
    print("Testing Serve Script")
    print("=" * 60)
    
    try:
        serve_path = "/vercel/sandbox/serve_deepseek_ocr.py"
        if not os.path.exists(serve_path):
            print(f"❌ Serve script not found: {serve_path}")
            return False
        
        print(f"✓ Serve script exists: {serve_path}")
        
        # Check if it's executable
        if os.access(serve_path, os.X_OK):
            print("✓ Serve script is executable")
        else:
            print("⚠ Serve script is not executable (not critical)")
        
        # Check the script content
        with open(serve_path, 'r') as f:
            content = f.read()
        
        checks = [
            ("Registration import", "import register_deepseek_ocr"),
            ("Argument parser", "argparse.ArgumentParser"),
            ("Model parameter", '"--model"'),
            ("Port parameter", '"--port"'),
        ]
        
        all_passed = True
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✓ {check_name}: Found")
            else:
                print(f"❌ {check_name}: Not found")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing serve script: {e}")
        return False


def test_readme_update():
    """Test that README was updated."""
    print("\n" + "=" * 60)
    print("Testing README Update")
    print("=" * 60)
    
    try:
        readme_path = "/vercel/sandbox/README.md"
        if not os.path.exists(readme_path):
            print(f"❌ README not found: {readme_path}")
            return False
        
        with open(readme_path, 'r') as f:
            content = f.read()
        
        checks = [
            ("Troubleshooting section", "## Troubleshooting"),
            ("Issue #244 reference", "Issue #244"),
            ("ValidationError mention", "ValidationError"),
            ("serve_deepseek_ocr.py reference", "serve_deepseek_ocr.py"),
            ("fix_model_config.py reference", "fix_model_config.py"),
            ("register_deepseek_ocr reference", "register_deepseek_ocr"),
        ]
        
        all_passed = True
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✓ {check_name}: Found")
            else:
                print(f"❌ {check_name}: Not found")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing README: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GitHub Issue #244 Fix - Test Suite")
    print("=" * 60 + "\n")
    
    results = {
        "Config Fix Utility": test_config_fix(),
        "Registration Module": test_registration_module(),
        "Serve Script": test_serve_script(),
        "README Update": test_readme_update(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nThe fix for GitHub Issue #244 is ready to use.")
        print("\nQuick Start:")
        print("  1. Use the serving script:")
        print("     python serve_deepseek_ocr.py")
        print("  2. Or fix your model config:")
        print("     python fix_model_config.py")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
