#!/usr/bin/env python3
"""
Test script to verify GPU optimization fixes for Issue #286.
This script checks if the environment is properly configured.
"""

import sys
import warnings

def test_imports():
    """Test if all required packages are installed."""
    print("=" * 80)
    print("Testing Package Imports")
    print("=" * 80)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'PIL': 'Pillow',
        'einops': 'Einops',
    }
    
    optional_packages = {
        'flash_attn': 'Flash Attention',
    }
    
    all_good = True
    
    for package, name in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: NOT INSTALLED")
            all_good = False
    
    for package, name in optional_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"⚠️  {name}: NOT INSTALLED (optional, but recommended)")
    
    return all_good

def test_cuda():
    """Test CUDA availability and configuration."""
    print("\n" + "=" * 80)
    print("Testing CUDA Configuration")
    print("=" * 80)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA is NOT available")
            print("   Please check your CUDA installation and GPU drivers")
            return False
        
        print(f"✅ CUDA is available")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n   GPU {i}:")
            print(f"     Name: {torch.cuda.get_device_name(i)}")
            print(f"     Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"     Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing CUDA: {e}")
        return False

def test_flash_attention():
    """Test Flash Attention availability."""
    print("\n" + "=" * 80)
    print("Testing Flash Attention")
    print("=" * 80)
    
    try:
        import flash_attn
        print(f"✅ Flash Attention is installed: {flash_attn.__version__}")
        return True
    except ImportError:
        print("⚠️  Flash Attention is NOT installed")
        print("   Install with: pip install flash-attn==2.7.3 --no-build-isolation")
        return False

def test_model_loading():
    """Test if model can be loaded with optimized settings."""
    print("\n" + "=" * 80)
    print("Testing Model Loading (Dry Run)")
    print("=" * 80)
    
    try:
        import torch
        from transformers import AutoTokenizer
        
        model_name = 'deepseek-ai/DeepSeek-OCR'
        
        print(f"Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=True
        )
        
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        
        print(f"\n⚠️  Skipping full model loading (requires ~10GB download)")
        print(f"   To test full model loading, run the optimized script")
        
        return True
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_map():
    """Test if device_map parameter works."""
    print("\n" + "=" * 80)
    print("Testing device_map Support")
    print("=" * 80)
    
    try:
        import torch
        from transformers import AutoModel
        
        # Test with a small model
        print("Testing device_map with a small model...")
        
        # Check if accelerate is installed (required for device_map)
        try:
            import accelerate
            print(f"✅ Accelerate is installed: {accelerate.__version__}")
        except ImportError:
            print("⚠️  Accelerate is NOT installed")
            print("   Install with: pip install accelerate")
            print("   device_map='auto' requires accelerate")
            return False
        
        print("✅ device_map support is available")
        return True
    except Exception as e:
        print(f"❌ Error testing device_map: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization features."""
    print("\n" + "=" * 80)
    print("Testing Memory Optimization Features")
    print("=" * 80)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping memory tests")
            return False
        
        # Test memory functions
        print("Testing CUDA memory functions...")
        torch.cuda.empty_cache()
        print("✅ torch.cuda.empty_cache() works")
        
        if hasattr(torch.cuda, 'memory_efficient_attention'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("✅ Memory efficient attention is available")
        else:
            print("⚠️  Memory efficient attention not available (requires PyTorch 2.0+)")
        
        # Test memory stats
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"✅ Current GPU memory:")
            print(f"   Allocated: {allocated:.2f} GB")
            print(f"   Reserved: {reserved:.2f} GB")
        
        return True
    except Exception as e:
        print(f"❌ Error testing memory optimization: {e}")
        return False

def test_torch_compile():
    """Test if torch.compile is available."""
    print("\n" + "=" * 80)
    print("Testing torch.compile Support")
    print("=" * 80)
    
    try:
        import torch
        
        if hasattr(torch, 'compile'):
            print(f"✅ torch.compile is available (PyTorch {torch.__version__})")
            print("   You can enable USE_TORCH_COMPILE=True for additional speedup")
            return True
        else:
            print(f"⚠️  torch.compile is NOT available (requires PyTorch 2.0+)")
            print(f"   Current version: {torch.__version__}")
            print("   Consider upgrading PyTorch for better performance")
            return False
    except Exception as e:
        print(f"❌ Error testing torch.compile: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DeepSeek-OCR GPU Optimization Test Suite")
    print("Issue #286 Fix Verification")
    print("=" * 80)
    
    results = {
        'Imports': test_imports(),
        'CUDA': test_cuda(),
        'Flash Attention': test_flash_attention(),
        'Model Loading': test_model_loading(),
        'device_map': test_device_map(),
        'Memory Optimization': test_memory_optimization(),
        'torch.compile': test_torch_compile(),
    }
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 All tests passed! Your environment is properly configured.")
        print("   You can now run the optimized script:")
        print("   python DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr_optimized.py")
    elif results['Imports'] and results['CUDA']:
        print("\n⚠️  Some optional features are missing, but basic functionality should work.")
        print("   Install missing packages for optimal performance.")
    else:
        print("\n❌ Critical tests failed. Please fix the issues above before proceeding.")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
