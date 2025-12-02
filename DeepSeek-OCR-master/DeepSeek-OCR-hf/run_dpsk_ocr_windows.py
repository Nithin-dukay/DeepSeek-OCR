"""
DeepSeek-OCR Windows 11 Compatible Inference Script

This script automatically detects available hardware and configures the model accordingly.
Works on Windows 11 with or without CUDA-capable GPU.

For GPUs with limited compute capability (like RTX 1060), this script avoids flash-attention
and uses standard attention mechanisms instead.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
import warnings

def get_device_info():
    """Detect available hardware and return appropriate configuration."""
    if torch.cuda.is_available():
        device = "cuda"
        # Check GPU compute capability
        compute_capability = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ CUDA is available")
        print(f"  GPU: {gpu_name}")
        print(f"  Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        
        # Determine if bfloat16 is supported
        # bfloat16 is well supported on Ampere (8.x) and newer
        if compute_capability[0] >= 8:
            dtype = torch.bfloat16
            print(f"  Using dtype: bfloat16 (recommended for this GPU)")
        else:
            dtype = torch.float16
            print(f"  Using dtype: float16 (bfloat16 not optimal for this GPU)")
            warnings.warn(
                "Your GPU has older compute capability. Using float16 instead of bfloat16. "
                "Results should still be good, but may differ slightly from the paper."
            )
    else:
        device = "cpu"
        dtype = torch.float32
        print("⚠ CUDA is not available. Using CPU.")
        print("  Using dtype: float32")
        print("  WARNING: CPU inference will be significantly slower than GPU.")
    
    return device, dtype

def main():
    # Model configuration
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    print("=" * 60)
    print("DeepSeek-OCR Windows 11 Inference")
    print("=" * 60)
    print("\nDetecting hardware configuration...")
    
    device, dtype = get_device_info()
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("Loading model...")
    print("Note: First-time download may take several minutes.")
    
    # Load model WITHOUT flash_attention_2 for Windows compatibility
    # Flash-attention has known issues on Windows and with older GPUs
    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=dtype,
            # Do NOT use flash_attention_2 on Windows or older GPUs
            # The default eager attention works fine
        )
        
        # Move model to appropriate device
        if device == "cuda":
            model = model.cuda()
        
        model = model.eval()
        
        print(f"\n✓ Model loaded successfully!")
        print(f"  Device: {next(model.parameters()).device}")
        print(f"  Dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have enough RAM (16GB+ recommended)")
        print("2. Close other applications to free up memory")
        print("3. Try the CPU-only script: run_dpsk_ocr_cpu.py")
        return
    
    # Configure your inference
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    # Alternative prompts:
    # prompt = "<image>\nFree OCR. "
    # prompt = "<image>\n<|grounding|>OCR this image."
    
    image_file = 'test.png'  # Change this to your image path
    output_path = 'output'   # Change this to your output directory
    
    # Inference parameters for different model sizes:
    # Tiny: base_size = 512, image_size = 512, crop_mode = False (fastest)
    # Small: base_size = 640, image_size = 640, crop_mode = False
    # Base: base_size = 1024, image_size = 1024, crop_mode = False
    # Large: base_size = 1280, image_size = 1280, crop_mode = False
    # Gundam: base_size = 1024, image_size = 640, crop_mode = True (best quality)
    
    # Recommended settings based on device
    if device == "cpu":
        # Use smaller size for CPU to reduce processing time
        base_size = 512
        image_size = 512
        crop_mode = False
        print("\n⚙ Using Tiny configuration for CPU (faster processing)")
    else:
        # Use Gundam mode for GPU (better quality)
        base_size = 1024
        image_size = 640
        crop_mode = True
        print("\n⚙ Using Gundam configuration for GPU (best quality)")
    
    print(f"\nStarting inference on: {image_file}")
    if device == "cpu":
        print("⏳ This may take several minutes on CPU...")
    
    try:
        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=True,
            test_compress=True
        )
        
        print("\n✓ Inference completed successfully!")
        print(f"  Results saved to: {output_path}")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify the image file exists and is readable")
        print("2. Try a smaller image size (use Tiny configuration)")
        print("3. Ensure output directory is writable")
        print("4. Check available memory (RAM/VRAM)")

if __name__ == "__main__":
    main()
