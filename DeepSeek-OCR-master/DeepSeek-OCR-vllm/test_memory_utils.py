#!/usr/bin/env python3
"""
Test script for memory management utilities.
This script verifies that the memory management functions work correctly.
"""

import sys
import torch
from process.memory_utils import (
    MemoryMonitor, 
    clear_memory, 
    clear_memory_aggressive,
    get_optimal_batch_size,
    estimate_pdf_memory_requirement,
    MemoryContext,
    print_memory_summary
)

def test_memory_monitor():
    """Test MemoryMonitor class."""
    print("\n" + "="*60)
    print("Testing MemoryMonitor")
    print("="*60)
    
    monitor = MemoryMonitor(enable_monitoring=True)
    
    # Test GPU memory info
    gpu_info = monitor.get_gpu_memory_info()
    print(f"GPU Memory Info: {gpu_info}")
    
    # Test CPU memory info
    cpu_info = monitor.get_cpu_memory_info()
    print(f"CPU Memory Info: {cpu_info}")
    
    # Test memory logging
    monitor.log_memory_usage("Test Stage")
    
    # Test memory threshold check
    is_safe = monitor.check_memory_threshold(threshold_gb=20.0)
    print(f"Memory is safe: {is_safe}")
    
    print("✓ MemoryMonitor test passed")


def test_memory_cleanup():
    """Test memory cleanup functions."""
    print("\n" + "="*60)
    print("Testing Memory Cleanup")
    print("="*60)
    
    # Create some tensors to use memory
    if torch.cuda.is_available():
        tensors = [torch.randn(1000, 1000).cuda() for _ in range(10)]
        print(f"Created {len(tensors)} tensors on GPU")
        
        # Test basic cleanup
        clear_memory(verbose=True)
        
        # Delete tensors
        del tensors
        
        # Test aggressive cleanup
        clear_memory_aggressive()
        print("✓ Memory cleanup test passed")
    else:
        print("⚠ CUDA not available, skipping GPU memory cleanup test")


def test_batch_size_calculation():
    """Test optimal batch size calculation."""
    print("\n" + "="*60)
    print("Testing Batch Size Calculation")
    print("="*60)
    
    # Test with different memory sizes
    test_cases = [
        (24.0, 0.5),  # 24GB GPU, 0.5GB per page
        (16.0, 0.5),  # 16GB GPU, 0.5GB per page
        (8.0, 0.5),   # 8GB GPU, 0.5GB per page
    ]
    
    for available_memory, per_page_memory in test_cases:
        batch_size = get_optimal_batch_size(available_memory, per_page_memory)
        print(f"Available: {available_memory}GB, Per-page: {per_page_memory}GB -> Batch size: {batch_size}")
    
    print("✓ Batch size calculation test passed")


def test_pdf_memory_estimation():
    """Test PDF memory requirement estimation."""
    print("\n" + "="*60)
    print("Testing PDF Memory Estimation")
    print("="*60)
    
    # Test with different PDF sizes
    test_cases = [400, 1000, 2800]
    
    for num_pages in test_cases:
        estimate = estimate_pdf_memory_requirement(num_pages)
        print(f"\nPDF with {num_pages} pages:")
        print(f"  Total estimated: {estimate['total_estimated_gb']:.2f} GB")
        print(f"  Recommended batch size: {estimate['recommended_batch_size']}")
    
    print("\n✓ PDF memory estimation test passed")


def test_memory_context():
    """Test MemoryContext context manager."""
    print("\n" + "="*60)
    print("Testing MemoryContext")
    print("="*60)
    
    monitor = MemoryMonitor(enable_monitoring=True)
    
    with MemoryContext("Test Operation", monitor):
        # Simulate some work
        if torch.cuda.is_available():
            temp_tensor = torch.randn(1000, 1000).cuda()
            del temp_tensor
    
    print("✓ MemoryContext test passed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DeepSeek OCR Memory Management Tests")
    print("="*60)
    
    try:
        test_memory_monitor()
        test_memory_cleanup()
        test_batch_size_calculation()
        test_pdf_memory_estimation()
        test_memory_context()
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60 + "\n")
        
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
