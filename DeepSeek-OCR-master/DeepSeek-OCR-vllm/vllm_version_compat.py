"""
vLLM Version Compatibility Module

This module automatically detects the vLLM version and sets the appropriate
environment variables to prevent the "Engine core proc died" error.

Usage:
    Import this module at the beginning of your script:
    
    from vllm_version_compat import setup_vllm_environment
    setup_vllm_environment()
"""

import os
import sys
from packaging import version


def setup_vllm_environment(force_v0=False, force_v1=False, verbose=True):
    """
    Automatically configure vLLM environment based on installed version.
    
    Args:
        force_v0 (bool): Force v0 architecture regardless of version
        force_v1 (bool): Force v1 architecture regardless of version
        verbose (bool): Print configuration information
    
    Returns:
        str: The vLLM version being used ('v0' or 'v1')
    """
    
    try:
        import vllm
        vllm_version = version.parse(vllm.__version__.split('+')[0])  # Remove build suffix
        
        if force_v0:
            os.environ['VLLM_USE_V1'] = '0'
            mode = 'v0'
            reason = '(forced)'
        elif force_v1:
            os.environ['VLLM_USE_V1'] = '1'
            mode = 'v1'
            reason = '(forced)'
        else:
            # Automatic detection based on version
            # vLLM v0.6.0+ introduced v1 architecture
            # vLLM v0.8.5 and below should use v0
            if vllm_version >= version.parse("0.9.0"):
                # Newer versions default to v1
                os.environ['VLLM_USE_V1'] = '1'
                mode = 'v1'
                reason = '(auto-detected: version >= 0.9.0)'
            elif vllm_version >= version.parse("0.6.0"):
                # Middle versions - check if nightly
                if 'dev' in vllm.__version__ or 'nightly' in vllm.__version__:
                    # Nightly builds may support v1
                    os.environ['VLLM_USE_V1'] = '0'
                    mode = 'v0'
                    reason = '(auto-detected: nightly build, using v0 for stability)'
                else:
                    os.environ['VLLM_USE_V1'] = '0'
                    mode = 'v0'
                    reason = '(auto-detected: stable release)'
            else:
                # Older versions use v0
                os.environ['VLLM_USE_V1'] = '0'
                mode = 'v0'
                reason = '(auto-detected: version < 0.6.0)'
        
        if verbose:
            print(f"=" * 60)
            print(f"vLLM Configuration:")
            print(f"  Version: {vllm.__version__}")
            print(f"  Architecture: {mode} {reason}")
            print(f"  VLLM_USE_V1: {os.environ.get('VLLM_USE_V1', 'not set')}")
            print(f"=" * 60)
        
        return mode
        
    except ImportError:
        print("ERROR: vLLM is not installed. Please install vLLM first.")
        print("Recommended: pip install vllm==0.8.5+cu118")
        sys.exit(1)
    except Exception as e:
        print(f"WARNING: Could not detect vLLM version: {e}")
        print("Defaulting to v0 architecture")
        os.environ['VLLM_USE_V1'] = '0'
        return 'v0'


def check_vllm_compatibility():
    """
    Check if the current vLLM installation is compatible with DeepSeek-OCR.
    
    Returns:
        tuple: (is_compatible, message)
    """
    try:
        import vllm
        vllm_version = version.parse(vllm.__version__.split('+')[0])
        
        # Recommended version
        recommended = version.parse("0.8.5")
        
        if vllm_version == recommended:
            return True, f"✓ Using recommended vLLM version {vllm.__version__}"
        elif vllm_version < version.parse("0.6.0"):
            return True, f"✓ Using compatible vLLM version {vllm.__version__}"
        elif vllm_version < version.parse("0.9.0"):
            return True, f"⚠ Using vLLM version {vllm.__version__} (may have compatibility issues)"
        else:
            return False, f"✗ vLLM version {vllm.__version__} may not be compatible. Recommended: 0.8.5"
            
    except ImportError:
        return False, "✗ vLLM is not installed"
    except Exception as e:
        return False, f"✗ Error checking vLLM: {e}"


if __name__ == "__main__":
    # Test the compatibility checker
    print("DeepSeek-OCR vLLM Compatibility Checker")
    print("=" * 60)
    
    is_compatible, message = check_vllm_compatibility()
    print(message)
    print()
    
    if is_compatible:
        setup_vllm_environment(verbose=True)
    else:
        print("\nRecommended installation:")
        print("  pip install vllm==0.8.5+cu118")
        print("\nOr download the wheel from:")
        print("  https://github.com/vllm-project/vllm/releases/tag/v0.8.5")
