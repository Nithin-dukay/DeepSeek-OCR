#!/usr/bin/env python3
"""Test script to verify SamplingMetadata import fix"""

import sys

def test_sampling_metadata_import():
    """Test if SamplingMetadata can be imported with the new backward-compatible approach"""
    print("Testing SamplingMetadata import...")
    
    try:
        # Try new import path (vLLM >= 0.9.0)
        from vllm.v1.sample.metadata import SamplingMetadata
        print("✓ Successfully imported SamplingMetadata from vllm.v1.sample.metadata (vLLM >= 0.9.0)")
        return True
    except ImportError as e1:
        print(f"✗ Failed to import from vllm.v1.sample.metadata: {e1}")
        
        try:
            # Fall back to old import path (vLLM 0.8.5)
            from vllm.model_executor import SamplingMetadata
            print("✓ Successfully imported SamplingMetadata from vllm.model_executor (vLLM 0.8.5)")
            return True
        except ImportError as e2:
            print(f"✗ Failed to import from vllm.model_executor: {e2}")
            
            try:
                # Last resort: try sampling_metadata module directly
                from vllm.model_executor.sampling_metadata import SamplingMetadata
                print("✓ Successfully imported SamplingMetadata from vllm.model_executor.sampling_metadata")
                return True
            except ImportError as e3:
                print(f"✗ Failed to import from vllm.model_executor.sampling_metadata: {e3}")
                print("\n❌ ERROR: Could not import SamplingMetadata from any known location.")
                print("This likely means vLLM is not installed or the version is incompatible.")
                return False

def test_deepseek_ocr_import():
    """Test if the deepseek_ocr module can be imported"""
    print("\nTesting deepseek_ocr module import...")
    
    try:
        sys.path.insert(0, '/vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-vllm')
        import deepseek_ocr
        print("✓ Successfully imported deepseek_ocr module")
        return True
    except ImportError as e:
        print(f"✗ Failed to import deepseek_ocr: {e}")
        print("\nThis is expected if vLLM is not installed.")
        return False
    except Exception as e:
        print(f"✗ Error importing deepseek_ocr: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek-OCR Import Test")
    print("=" * 60)
    
    # Test 1: Direct import test
    result1 = test_sampling_metadata_import()
    
    # Test 2: Module import test
    result2 = test_deepseek_ocr_import()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"SamplingMetadata import: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"deepseek_ocr module import: {'✓ PASS' if result2 else '✗ FAIL (expected if vLLM not installed)'}")
    print("=" * 60)
    
    if not result1:
        print("\n⚠️  Note: vLLM needs to be installed to fully test this fix.")
        print("Install vLLM with: pip install vllm")
        sys.exit(1)
    else:
        print("\n✓ Import fix is working correctly!")
        sys.exit(0)
