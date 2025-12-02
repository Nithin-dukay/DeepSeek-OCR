"""
Memory management utilities for DeepSeek OCR.
Provides functions for monitoring and managing GPU/CPU memory during large document processing.
"""

import gc
import torch
import psutil
import os
from typing import Optional, Dict


class MemoryMonitor:
    """Monitor and manage memory usage during processing."""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.peak_memory = 0
        self.initial_memory = 0
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "free": 0.0}
        
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        
        # Get total GPU memory
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free = total - allocated
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total
        }
    
    def get_cpu_memory_info(self) -> Dict[str, float]:
        """Get current CPU memory usage in GB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / (1024 ** 3),  # Resident Set Size in GB
            "vms": memory_info.vms / (1024 ** 3),  # Virtual Memory Size in GB
            "percent": process.memory_percent()
        }
    
    def log_memory_usage(self, stage: str = ""):
        """Log current memory usage."""
        if not self.enable_monitoring:
            return
        
        gpu_info = self.get_gpu_memory_info()
        cpu_info = self.get_cpu_memory_info()
        
        print(f"\n{'='*60}")
        print(f"Memory Usage {f'[{stage}]' if stage else ''}")
        print(f"{'='*60}")
        print(f"GPU Memory:")
        print(f"  Allocated: {gpu_info['allocated']:.2f} GB")
        print(f"  Reserved:  {gpu_info['reserved']:.2f} GB")
        print(f"  Free:      {gpu_info['free']:.2f} GB")
        print(f"  Total:     {gpu_info['total']:.2f} GB")
        print(f"CPU Memory:")
        print(f"  RSS:       {cpu_info['rss']:.2f} GB")
        print(f"  Percent:   {cpu_info['percent']:.1f}%")
        print(f"{'='*60}\n")
        
        # Track peak memory
        if gpu_info['allocated'] > self.peak_memory:
            self.peak_memory = gpu_info['allocated']
    
    def check_memory_threshold(self, threshold_gb: float = 20.0) -> bool:
        """
        Check if GPU memory usage exceeds threshold.
        Returns True if memory is below threshold (safe to continue).
        """
        gpu_info = self.get_gpu_memory_info()
        if gpu_info['allocated'] > threshold_gb:
            print(f"\n⚠️  WARNING: GPU memory usage ({gpu_info['allocated']:.2f} GB) "
                  f"exceeds threshold ({threshold_gb:.2f} GB)")
            return False
        return True
    
    def get_peak_memory(self) -> float:
        """Get peak GPU memory usage in GB."""
        return self.peak_memory


def clear_memory(verbose: bool = False):
    """
    Aggressively clear GPU and CPU memory.
    
    Args:
        verbose: If True, print memory info before and after cleanup
    """
    if verbose:
        print("\n🧹 Clearing memory...")
    
    # Clear Python garbage
    gc.collect()
    
    # Clear PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force another garbage collection
    gc.collect()
    
    if verbose:
        print("✓ Memory cleared\n")


def clear_memory_aggressive():
    """
    More aggressive memory clearing for critical situations.
    Includes multiple passes of garbage collection.
    """
    # Multiple passes of garbage collection
    for _ in range(3):
        gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
    
    # Final garbage collection
    gc.collect()


def get_optimal_batch_size(available_memory_gb: float, 
                          per_page_memory_gb: float = 0.5) -> int:
    """
    Calculate optimal batch size based on available memory.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        per_page_memory_gb: Estimated memory per page in GB
    
    Returns:
        Recommended batch size
    """
    # Reserve 20% of memory as buffer
    usable_memory = available_memory_gb * 0.8
    batch_size = int(usable_memory / per_page_memory_gb)
    
    # Ensure minimum batch size of 1 and maximum of 100
    return max(1, min(batch_size, 100))


def estimate_pdf_memory_requirement(num_pages: int, 
                                   per_page_memory_gb: float = 0.5) -> Dict[str, float]:
    """
    Estimate memory requirements for processing a PDF.
    
    Args:
        num_pages: Number of pages in PDF
        per_page_memory_gb: Estimated memory per page
    
    Returns:
        Dictionary with memory estimates
    """
    total_memory = num_pages * per_page_memory_gb
    
    return {
        "total_estimated_gb": total_memory,
        "per_page_gb": per_page_memory_gb,
        "num_pages": num_pages,
        "recommended_batch_size": get_optimal_batch_size(20.0, per_page_memory_gb)
    }


class MemoryContext:
    """Context manager for automatic memory cleanup."""
    
    def __init__(self, stage_name: str = "", monitor: Optional[MemoryMonitor] = None):
        self.stage_name = stage_name
        self.monitor = monitor or MemoryMonitor()
    
    def __enter__(self):
        if self.stage_name:
            self.monitor.log_memory_usage(f"Before {self.stage_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_memory(verbose=False)
        if self.stage_name:
            self.monitor.log_memory_usage(f"After {self.stage_name}")
        return False


def print_memory_summary(monitor: MemoryMonitor):
    """Print a summary of memory usage."""
    print(f"\n{'='*60}")
    print("Memory Usage Summary")
    print(f"{'='*60}")
    print(f"Peak GPU Memory: {monitor.get_peak_memory():.2f} GB")
    
    current_gpu = monitor.get_gpu_memory_info()
    current_cpu = monitor.get_cpu_memory_info()
    
    print(f"Current GPU Memory: {current_gpu['allocated']:.2f} GB")
    print(f"Current CPU Memory: {current_cpu['rss']:.2f} GB ({current_cpu['percent']:.1f}%)")
    print(f"{'='*60}\n")
