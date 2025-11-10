"""
Memory management utilities for DeepSeek OCR
Provides functions to monitor and clean up memory during large document processing
"""

import gc
import torch
from typing import Optional, List, Any


def cleanup_memory(verbose: bool = False):
    """
    Perform comprehensive memory cleanup for both CPU and GPU.
    
    Args:
        verbose: If True, print memory statistics before and after cleanup
    """
    if verbose and torch.cuda.is_available():
        print(f"GPU Memory before cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
    # Clear Python garbage collector
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if verbose and torch.cuda.is_available():
        print(f"GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")


def delete_and_cleanup(*objects, verbose: bool = False):
    """
    Delete objects and perform memory cleanup.
    
    Args:
        *objects: Variable number of objects to delete
        verbose: If True, print cleanup information
    """
    for obj in objects:
        if obj is not None:
            del obj
    
    cleanup_memory(verbose=verbose)


def get_gpu_memory_info() -> dict:
    """
    Get current GPU memory usage information.
    
    Returns:
        Dictionary with memory statistics or empty dict if CUDA not available
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


def print_memory_stats(prefix: str = ""):
    """
    Print current memory statistics with optional prefix.
    
    Args:
        prefix: String to print before memory stats
    """
    if torch.cuda.is_available():
        stats = get_gpu_memory_info()
        print(f"{prefix}GPU Memory: {stats['allocated_gb']:.2f} GB allocated, "
              f"{stats['reserved_gb']:.2f} GB reserved")


def reset_peak_memory_stats():
    """Reset peak memory statistics for monitoring."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def clear_image_list(image_list: List[Any]) -> None:
    """
    Clear a list of images and free memory.
    
    Args:
        image_list: List of PIL Images or other objects to clear
    """
    if image_list:
        for img in image_list:
            if img is not None:
                del img
        image_list.clear()
    gc.collect()


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations."""
    
    def __init__(self, operation_name: str = "Operation", verbose: bool = True):
        self.operation_name = operation_name
        self.verbose = verbose
        self.start_memory = None
    
    def __enter__(self):
        if self.verbose and torch.cuda.is_available():
            self.start_memory = get_gpu_memory_info()
            print(f"\n[{self.operation_name}] Starting - GPU Memory: "
                  f"{self.start_memory['allocated_gb']:.2f} GB allocated")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose and torch.cuda.is_available():
            end_memory = get_gpu_memory_info()
            memory_diff = end_memory['allocated_gb'] - self.start_memory['allocated_gb']
            print(f"[{self.operation_name}] Completed - GPU Memory: "
                  f"{end_memory['allocated_gb']:.2f} GB allocated "
                  f"(Δ {memory_diff:+.2f} GB)")
        
        # Cleanup on exit
        cleanup_memory(verbose=False)
        return False
