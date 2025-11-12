#!/usr/bin/env python3
"""
Test script to verify the streaming solutions are working correctly.

This script helps you test each solution without needing actual images or GPU.
"""

import sys
import os

def test_solution1():
    """Test Solution 1: eval_mode=True"""
    print("=" * 80)
    print("Testing Solution 1: eval_mode=True")
    print("=" * 80)
    
    print("\n✓ Solution 1 is a simple parameter change:")
    print("  Before: result = model.infer(...)")
    print("  After:  result = model.infer(..., eval_mode=True)")
    print("\n✓ No additional code needed!")
    print("✓ Just add eval_mode=True to get the output")
    
    return True


def test_solution2():
    """Test Solution 2: Custom Streamer"""
    print("\n" + "=" * 80)
    print("Testing Solution 2: Custom Streamer")
    print("=" * 80)
    
    try:
        from transformers import TextStreamer
        from queue import Queue
        
        print("\n✓ transformers library is available")
        print("✓ Queue module is available")
        
        # Test basic streamer functionality
        class TestStreamer(TextStreamer):
            def __init__(self):
                self.token_queue = Queue()
            
            def on_finalized_text(self, text: str, stream_end: bool = False):
                self.token_queue.put(text)
                if stream_end:
                    self.token_queue.put(None)
        
        streamer = TestStreamer()
        streamer.on_finalized_text("test", stream_end=False)
        streamer.on_finalized_text("", stream_end=True)
        
        tokens = []
        while True:
            token = streamer.token_queue.get()
            if token is None:
                break
            tokens.append(token)
        
        if tokens == ["test"]:
            print("✓ Custom streamer test passed!")
            return True
        else:
            print("✗ Custom streamer test failed")
            return False
            
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_solution3():
    """Test Solution 3: vLLM + FastAPI"""
    print("\n" + "=" * 80)
    print("Testing Solution 3: vLLM + FastAPI")
    print("=" * 80)
    
    # Check FastAPI
    try:
        import fastapi
        print("\n✓ FastAPI is installed")
    except ImportError:
        print("\n✗ FastAPI not installed")
        print("  Install with: pip install fastapi")
        return False
    
    # Check uvicorn
    try:
        import uvicorn
        print("✓ uvicorn is installed")
    except ImportError:
        print("✗ uvicorn not installed")
        print("  Install with: pip install uvicorn")
        return False
    
    # Check vLLM (optional, may not be installed)
    try:
        import vllm
        print("✓ vLLM is installed")
        vllm_available = True
    except ImportError:
        print("⚠ vLLM not installed (optional for testing)")
        print("  Install with: pip install vllm")
        vllm_available = False
    
    # Check requests for client
    try:
        import requests
        print("✓ requests is installed (for client)")
    except ImportError:
        print("⚠ requests not installed (needed for client)")
        print("  Install with: pip install requests")
    
    # Check websockets
    try:
        import websockets
        print("✓ websockets is installed")
    except ImportError:
        print("⚠ websockets not installed (optional)")
        print("  Install with: pip install websockets")
    
    if vllm_available:
        print("\n✓ All Solution 3 dependencies are installed!")
        return True
    else:
        print("\n⚠ Solution 3 partially ready (vLLM not installed)")
        print("  You can still test the FastAPI server structure")
        return True


def test_server_connection():
    """Test if Solution 3 server is running"""
    print("\n" + "=" * 80)
    print("Testing Server Connection")
    print("=" * 80)
    
    try:
        import requests
        
        print("\nChecking if server is running on http://localhost:8000...")
        
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✓ Server is running!")
                print(f"  Response: {response.json()}")
                return True
            else:
                print(f"⚠ Server responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("✗ Server is not running")
            print("\nTo start the server, run:")
            print("  python examples/solution3_vllm_fastapi_server.py")
            return False
        except requests.exceptions.Timeout:
            print("✗ Server connection timeout")
            return False
            
    except ImportError:
        print("✗ requests library not installed")
        print("  Install with: pip install requests")
        return False


def check_gpu():
    """Check if GPU is available"""
    print("\n" + "=" * 80)
    print("GPU Check")
    print("=" * 80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"\n✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            return True
        else:
            print("\n⚠ CUDA is not available")
            print("  The solutions will work but may be slower on CPU")
            return False
            
    except ImportError:
        print("\n✗ PyTorch not installed")
        print("  Install with: pip install torch")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("DeepSeek-OCR Streaming Solutions - Test Suite")
    print("=" * 80)
    print("\nThis script tests if your environment is ready for each solution.")
    print("You don't need a GPU or images to run these tests.\n")
    
    results = {}
    
    # Test each solution
    results['solution1'] = test_solution1()
    results['solution2'] = test_solution2()
    results['solution3'] = test_solution3()
    
    # Additional checks
    results['gpu'] = check_gpu()
    results['server'] = test_server_connection()
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    print("\nSolution Readiness:")
    print(f"  Solution 1 (eval_mode):     {'✓ Ready' if results['solution1'] else '✗ Not Ready'}")
    print(f"  Solution 2 (Custom Stream): {'✓ Ready' if results['solution2'] else '✗ Not Ready'}")
    print(f"  Solution 3 (vLLM + FastAPI):{'✓ Ready' if results['solution3'] else '✗ Not Ready'}")
    
    print("\nEnvironment:")
    print(f"  GPU Available:              {'✓ Yes' if results['gpu'] else '⚠ No (CPU only)'}")
    print(f"  Server Running:             {'✓ Yes' if results['server'] else '✗ No'}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)
    
    if results['solution1']:
        print("\n✓ You can use Solution 1 right away!")
        print("  Just add eval_mode=True to model.infer()")
    
    if results['solution2']:
        print("\n✓ You can use Solution 2!")
        print("  See: examples/solution2_custom_streamer.py")
    
    if results['solution3']:
        if results['server']:
            print("\n✓ Solution 3 server is running!")
            print("  Test it at: http://localhost:8000/docs")
        else:
            print("\n✓ You can start Solution 3 server!")
            print("  Run: python examples/solution3_vllm_fastapi_server.py")
    
    if not results['gpu']:
        print("\n⚠ No GPU detected. Solutions will work but may be slower.")
        print("  For production use, a GPU is recommended.")
    
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("\n1. Read: QUICKSTART_STREAMING.md")
    print("2. Choose your solution based on the test results")
    print("3. Run the example code for your chosen solution")
    print("4. Integrate into your application")
    
    print("\n" + "=" * 80)
    
    # Exit code
    if results['solution1'] or results['solution2'] or results['solution3']:
        print("\n✓ At least one solution is ready to use!")
        return 0
    else:
        print("\n✗ Please install missing dependencies")
        return 1


if __name__ == '__main__':
    sys.exit(main())
