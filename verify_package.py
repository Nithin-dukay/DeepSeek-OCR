#!/usr/bin/env python3
"""
Verification script for DeepSeek-OCR package setup.
This script checks if all packaging files are correctly configured.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} MISSING: {filepath}")
        return False

def check_file_content(filepath, search_string, description):
    """Check if a file contains specific content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_string in content:
                print(f"✓ {description}")
                return True
            else:
                print(f"✗ {description} - content not found")
                return False
    except Exception as e:
        print(f"✗ {description} - error reading file: {e}")
        return False

def main():
    """Main verification function."""
    print("=" * 70)
    print("DeepSeek-OCR Package Setup Verification")
    print("=" * 70)
    print()
    
    all_checks_passed = True
    
    # Check main packaging files
    print("1. Checking Main Packaging Files:")
    print("-" * 70)
    all_checks_passed &= check_file_exists("setup.py", "setup.py")
    all_checks_passed &= check_file_exists("MANIFEST.in", "MANIFEST.in")
    all_checks_passed &= check_file_exists("pyproject.toml", "pyproject.toml")
    all_checks_passed &= check_file_exists(".gitignore", ".gitignore")
    all_checks_passed &= check_file_exists("INSTALL.md", "INSTALL.md")
    all_checks_passed &= check_file_exists("requirements.txt", "requirements.txt")
    print()
    
    # Check __init__.py files
    print("2. Checking Package Structure (__init__.py files):")
    print("-" * 70)
    init_files = [
        "DeepSeek-OCR-master/__init__.py",
        "DeepSeek-OCR-master/DeepSeek-OCR-hf/__init__.py",
        "DeepSeek-OCR-master/DeepSeek-OCR-vllm/__init__.py",
        "DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/__init__.py",
        "DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/__init__.py",
    ]
    for init_file in init_files:
        all_checks_passed &= check_file_exists(init_file, f"__init__.py")
    print()
    
    # Check setup.py content
    print("3. Checking setup.py Configuration:")
    print("-" * 70)
    all_checks_passed &= check_file_content(
        "setup.py", 
        "name='deepseek-ocr'",
        "Package name is 'deepseek-ocr'"
    )
    all_checks_passed &= check_file_content(
        "setup.py",
        "version='1.0.0'",
        "Version is '1.0.0'"
    )
    all_checks_passed &= check_file_content(
        "setup.py",
        "python_requires='>=3.8'",
        "Python requirement is >=3.8"
    )
    print()
    
    # Check MANIFEST.in content
    print("4. Checking MANIFEST.in Configuration:")
    print("-" * 70)
    all_checks_passed &= check_file_content(
        "MANIFEST.in",
        "include README.md",
        "Includes README.md"
    )
    all_checks_passed &= check_file_content(
        "MANIFEST.in",
        "include LICENSE",
        "Includes LICENSE"
    )
    all_checks_passed &= check_file_content(
        "MANIFEST.in",
        "recursive-include assets",
        "Includes assets directory"
    )
    print()
    
    # Check pyproject.toml content
    print("5. Checking pyproject.toml Configuration:")
    print("-" * 70)
    all_checks_passed &= check_file_content(
        "pyproject.toml",
        'name = "deepseek-ocr"',
        "Package name configured"
    )
    all_checks_passed &= check_file_content(
        "pyproject.toml",
        "[project.optional-dependencies]",
        "Optional dependencies configured"
    )
    print()
    
    # Test setup.py syntax
    print("6. Testing setup.py Syntax:")
    print("-" * 70)
    try:
        import ast
        with open("setup.py", 'r') as f:
            ast.parse(f.read())
        print("✓ setup.py has valid Python syntax")
    except SyntaxError as e:
        print(f"✗ setup.py has syntax error: {e}")
        all_checks_passed = False
    print()
    
    # Test pyproject.toml syntax
    print("7. Testing pyproject.toml Syntax:")
    print("-" * 70)
    try:
        import tomli
        with open("pyproject.toml", 'rb') as f:
            tomli.load(f)
        print("✓ pyproject.toml has valid TOML syntax")
    except ImportError:
        try:
            import tomllib
            with open("pyproject.toml", 'rb') as f:
                tomllib.load(f)
            print("✓ pyproject.toml has valid TOML syntax")
        except ImportError:
            print("⚠ Cannot verify TOML syntax (tomli/tomllib not available)")
        except Exception as e:
            print(f"✗ pyproject.toml has syntax error: {e}")
            all_checks_passed = False
    except Exception as e:
        print(f"✗ pyproject.toml has syntax error: {e}")
        all_checks_passed = False
    print()
    
    # Summary
    print("=" * 70)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED - Package setup is complete!")
        print()
        print("Next steps:")
        print("  1. Install the package: pip install -e .")
        print("  2. Test imports: python -c 'import sys; sys.path.insert(0, \"DeepSeek-OCR-master\")'")
        print("  3. Build distributions: python setup.py sdist bdist_wheel")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please review the errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
