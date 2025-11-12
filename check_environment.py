#!/usr/bin/env python3
"""
Environment checker for DeepSeek-OCR
Verifies that all dependencies are correctly installed with compatible versions.
"""

import sys
import importlib.util
from packaging import version


def check_package_version(package_name, required_version=None, min_version=None, max_version=None):
    """Check if a package is installed and meets version requirements."""
    try:
        module = __import__(package_name)
        installed_version = getattr(module, '__version__', 'unknown')
        
        if installed_version == 'unknown':
            return True, installed_version, "⚠️  Warning: Could not determine version"
        
        v = version.parse(installed_version)
        
        if required_version:
            req_v = version.parse(required_version)
            if v != req_v:
                return False, installed_version, f"❌ Required: {required_version}"
        
        if min_version:
            min_v = version.parse(min_version)
            if v < min_v:
                return False, installed_version, f"❌ Minimum required: {min_version}"
        
        if max_version:
            max_v = version.parse(max_version)
            if v >= max_v:
                return False, installed_version, f"❌ Maximum allowed: < {max_version}"
        
        return True, installed_version, "✅ OK"
    
    except ImportError:
        return False, "Not installed", "❌ Not installed"
    except Exception as e:
        return False, "Error", f"❌ Error: {str(e)}"


def main():
    print("=" * 70)
    print("DeepSeek-OCR Environment Check")
    print("=" * 70)
    print()
    
    checks = [
        ("transformers", "4.46.3", None, "4.47.0"),  # Must be 4.46.3, not 4.47+
        ("tokenizers", None, "0.20.0", None),
        ("torch", None, "2.0.0", None),
        ("einops", None, None, None),
        ("PIL", None, None, None),  # Pillow
        ("numpy", None, None, None),
    ]
    
    all_passed = True
    results = []
    
    print("Checking required packages:")
    print("-" * 70)
    
    for package_info in checks:
        package_name = package_info[0]
        required_version = package_info[1]
        min_version = package_info[2]
        max_version = package_info[3]
        
        passed, installed_version, message = check_package_version(
            package_name, required_version, min_version, max_version
        )
        
        results.append((package_name, installed_version, message, passed))
        
        if not passed:
            all_passed = False
    
    # Print results
    for package_name, installed_version, message, passed in results:
        status_icon = "✅" if passed else "❌"
        print(f"{status_icon} {package_name:20s} {installed_version:15s} {message}")
    
    print("-" * 70)
    print()
    
    # Check for flash-attn (optional but recommended)
    print("Checking optional packages:")
    print("-" * 70)
    try:
        import flash_attn
        flash_version = getattr(flash_attn, '__version__', 'unknown')
        print(f"✅ flash-attn          {flash_version:15s} ✅ OK (recommended)")
    except ImportError:
        print(f"⚠️  flash-attn          Not installed      ⚠️  Recommended for better performance")
    print("-" * 70)
    print()
    
    # Final verdict
    if all_passed:
        print("✅ All required packages are correctly installed!")
        print()
        print("You can now run DeepSeek-OCR:")
        print("  cd DeepSeek-OCR-master/DeepSeek-OCR-hf")
        print("  python run_dpsk_ocr.py")
        return 0
    else:
        print("❌ Some packages need attention!")
        print()
        print("To fix the issues:")
        print()
        print("1. For transformers version mismatch:")
        print("   pip install transformers==4.46.3")
        print()
        print("2. For missing packages:")
        print("   pip install -r requirements.txt")
        print()
        print("3. For Colab users:")
        print("   !pip install transformers==4.46.3 --force-reinstall")
        print("   Then restart the runtime: Runtime -> Restart runtime")
        print()
        print("⚠️  IMPORTANT: transformers 4.47+ has breaking changes!")
        print("   You MUST use version 4.46.3 for DeepSeek-OCR to work.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
