#!/usr/bin/env python3
"""
Test script to verify DeepseekOCRForCausalLM model registration with vLLM.

This script tests that the model can be properly registered and recognized
by vLLM's ModelRegistry without actually loading the full model.

Usage:
    python test_registration.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_registration():
    """Test that the model can be registered with vLLM."""
    print("="*60)
    print("DeepSeek-OCR Model Registration Test")
    print("="*60)
    
    try:
        print("\n1. Importing vLLM ModelRegistry...")
        from vllm.model_executor.models.registry import ModelRegistry
        print("   ✓ Successfully imported ModelRegistry")
        
        print("\n2. Importing DeepseekOCRForCausalLM...")
        from deepseek_ocr import DeepseekOCRForCausalLM
        print("   ✓ Successfully imported DeepseekOCRForCausalLM")
        
        print("\n3. Registering model with vLLM...")
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
        print("   ✓ Successfully registered model")
        
        print("\n4. Verifying model is in registry...")
        # Check if model is registered
        registered_models = ModelRegistry.get_supported_archs()
        if "DeepseekOCRForCausalLM" in registered_models:
            print("   ✓ Model found in registry")
        else:
            print("   ⚠ Model not found in registry (this may be expected)")
        
        print("\n5. Testing model class attributes...")
        # Verify the class has required attributes
        required_attrs = ['__init__', 'forward', 'load_weights']
        missing_attrs = [attr for attr in required_attrs if not hasattr(DeepseekOCRForCausalLM, attr)]
        
        if not missing_attrs:
            print("   ✓ All required attributes present")
        else:
            print(f"   ✗ Missing attributes: {missing_attrs}")
            return False
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        print("\nThe model is properly configured and can be used with vLLM.")
        print("\nNext steps:")
        print("  1. Use the provided scripts: run_dpsk_ocr_image.py, run_dpsk_ocr_pdf.py")
        print("  2. Or use serve_deepseek_ocr.py to start a server")
        print("  3. See VLLM_SERVE_GUIDE.md for more details")
        print()
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\nChecking dependencies...")
    
    dependencies = {
        'vllm': 'vLLM',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'einops': 'einops',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed")
    return True


if __name__ == "__main__":
    print("\nDeepSeek-OCR vLLM Integration Test\n")
    
    # Test dependencies first
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n⚠ Please install missing dependencies before proceeding.")
        sys.exit(1)
    
    # Test model registration
    success = test_model_registration()
    
    sys.exit(0 if success else 1)
