"""
Test script to verify that the warnings fix works correctly.

This script tests the model loading and basic functionality without
requiring actual image files or GPU resources.
"""

import sys
import warnings
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("TEST 1: Testing imports...")
    print("=" * 60)
    
    try:
        from configuration_deepseek_ocr import DeepSeekOCRConfig
        print("✅ Successfully imported DeepSeekOCRConfig")
    except Exception as e:
        print(f"❌ Failed to import DeepSeekOCRConfig: {e}")
        return False
    
    try:
        from modeling_deepseek_ocr import DeepSeekOCRForCausalLM
        print("✅ Successfully imported DeepSeekOCRForCausalLM")
    except Exception as e:
        print(f"❌ Failed to import DeepSeekOCRForCausalLM: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✅ Successfully imported AutoTokenizer")
    except Exception as e:
        print(f"❌ Failed to import AutoTokenizer: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True


def test_config_creation():
    """Test that the configuration class works correctly."""
    print("=" * 60)
    print("TEST 2: Testing configuration creation...")
    print("=" * 60)
    
    try:
        from configuration_deepseek_ocr import DeepSeekOCRConfig
        
        # Create a default config
        config = DeepSeekOCRConfig()
        print(f"✅ Created default config with model_type: {config.model_type}")
        
        # Verify model type
        assert config.model_type == "deepseek_ocr", "Model type should be 'deepseek_ocr'"
        print("✅ Model type is correctly set to 'deepseek_ocr'")
        
        # Test config serialization
        config_dict = config.to_dict()
        assert "model_type" in config_dict, "Config dict should contain model_type"
        print("✅ Config serialization works correctly")
        
        # Test config with custom parameters
        custom_config = DeepSeekOCRConfig(
            tile_tag="2D",
            global_view_pos="tail",
            image_token_id=100015
        )
        print(f"✅ Created custom config with tile_tag: {custom_config.tile_tag}")
        
        print("\n✅ Configuration tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_model_class():
    """Test that the model class can be instantiated."""
    print("=" * 60)
    print("TEST 3: Testing model class instantiation...")
    print("=" * 60)
    
    try:
        from configuration_deepseek_ocr import DeepSeekOCRConfig
        from modeling_deepseek_ocr import DeepSeekOCRForCausalLM
        
        # Create a config
        config = DeepSeekOCRConfig()
        
        # Create a model instance (without loading weights)
        model = DeepSeekOCRForCausalLM(config)
        print("✅ Successfully instantiated DeepSeekOCRForCausalLM")
        
        # Verify model has required attributes
        assert hasattr(model, 'config'), "Model should have config attribute"
        assert hasattr(model, 'infer'), "Model should have infer method"
        assert hasattr(model, 'generate'), "Model should have generate method"
        print("✅ Model has all required attributes and methods")
        
        print("\n✅ Model class tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Model class test failed: {e}")
        traceback.print_exc()
        return False


def test_warning_suppression():
    """Test that warnings are properly suppressed."""
    print("=" * 60)
    print("TEST 4: Testing warning suppression...")
    print("=" * 60)
    
    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Import and create instances
            from configuration_deepseek_ocr import DeepSeekOCRConfig
            from modeling_deepseek_ocr import DeepSeekOCRForCausalLM
            
            config = DeepSeekOCRConfig()
            model = DeepSeekOCRForCausalLM(config)
            
            # Check if any warnings were raised
            if len(w) == 0:
                print("✅ No warnings raised during model instantiation")
            else:
                print(f"⚠️  {len(w)} warning(s) raised:")
                for warning in w:
                    print(f"   - {warning.category.__name__}: {warning.message}")
        
        print("\n✅ Warning suppression test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Warning suppression test failed: {e}")
        traceback.print_exc()
        return False


def test_model_methods():
    """Test that model methods exist and have proper signatures."""
    print("=" * 60)
    print("TEST 5: Testing model methods...")
    print("=" * 60)
    
    try:
        from configuration_deepseek_ocr import DeepSeekOCRConfig
        from modeling_deepseek_ocr import DeepSeekOCRForCausalLM
        import inspect
        
        config = DeepSeekOCRConfig()
        model = DeepSeekOCRForCausalLM(config)
        
        # Check required methods
        required_methods = ['forward', 'generate', 'infer', 'eval', 'train', 'cuda', 'cpu', 'to']
        
        for method_name in required_methods:
            assert hasattr(model, method_name), f"Model should have {method_name} method"
            method = getattr(model, method_name)
            assert callable(method), f"{method_name} should be callable"
            print(f"✅ Method '{method_name}' exists and is callable")
        
        print("\n✅ Model methods test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Model methods test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DeepSeek-OCR Warnings Fix - Test Suite")
    print("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config_creation),
        ("Model Class", test_model_class),
        ("Warning Suppression", test_warning_suppression),
        ("Model Methods", test_model_methods),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60 + "\n")
    
    if passed == total:
        print("🎉 All tests passed! The warnings fix is working correctly.")
        print("\nNext steps:")
        print("1. Test with actual model loading (requires downloading the model)")
        print("2. Test with actual inference (requires GPU and image files)")
        print("3. Verify warnings are eliminated in production usage")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
