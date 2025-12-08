#!/usr/bin/env python3
"""
DeepSeek-OCR Installation Script
This script ensures proper installation order to avoid dependency issues,
particularly the xformers/torch dependency problem.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            text=True,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Command failed with exit code {e.returncode}")
        return False


def verify_import(module_name):
    """Verify that a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         DeepSeek-OCR Installation Script                  ║
    ║  Fixes Issue #296: xformers torch dependency problem      ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Check for vLLM wheel
    vllm_whl = "vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl"
    if not Path(vllm_whl).exists():
        print(f"\n❌ Error: vLLM wheel file not found: {vllm_whl}")
        print("Please download it from: https://github.com/vllm-project/vllm/releases/tag/v0.8.5")
        sys.exit(1)
    
    print(f"✓ Found vLLM wheel: {vllm_whl}\n")
    
    # Step 1: Install PyTorch (CRITICAL - must be before vLLM)
    print("\n" + "="*60)
    print("STEP 1/4: Installing PyTorch 2.6.0 with CUDA 11.8")
    print("="*60)
    print("\n⚠️  IMPORTANT: PyTorch MUST be installed before vLLM/xformers")
    print("This fixes the 'ModuleNotFoundError: No module named torch' error\n")
    
    torch_cmd = (
        "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    
    if not run_command(torch_cmd, "Installing PyTorch..."):
        print("\n❌ Failed to install PyTorch")
        sys.exit(1)
    
    # Verify PyTorch installation
    print("\n" + "-"*60)
    print("Verifying PyTorch installation...")
    print("-"*60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"❌ Error: Failed to import torch: {e}")
        sys.exit(1)
    
    # Step 2: Install vLLM
    print("\n" + "="*60)
    print("STEP 2/4: Installing vLLM")
    print("="*60)
    print("\n⚠️  This may take a while as xformers needs to be built from source")
    print("xformers will now be able to find torch during its build process\n")
    
    vllm_cmd = f"pip install {vllm_whl}"
    
    if not run_command(vllm_cmd, "Installing vLLM...", check=False):
        print("\n⚠️  vLLM installation failed. Trying alternative method...")
        print("Installing xformers separately first...\n")
        
        # Try installing xformers separately
        if not run_command("pip install xformers==0.0.29.post2", "Installing xformers..."):
            print("\n❌ Failed to install xformers")
            sys.exit(1)
        
        # Retry vLLM installation
        if not run_command(vllm_cmd, "Retrying vLLM installation..."):
            print("\n❌ Failed to install vLLM even after installing xformers separately")
            sys.exit(1)
    
    print("\n✓ vLLM installed successfully")
    
    # Step 3: Install other requirements
    print("\n" + "="*60)
    print("STEP 3/4: Installing other requirements")
    print("="*60)
    
    if not run_command("pip install -r requirements.txt", "Installing requirements..."):
        print("\n❌ Failed to install requirements")
        sys.exit(1)
    
    print("\n✓ Requirements installed successfully")
    
    # Step 4: Install flash-attention
    print("\n" + "="*60)
    print("STEP 4/4: Installing flash-attention")
    print("="*60)
    print("\n⚠️  This may take a while to compile\n")
    
    if not run_command(
        "pip install flash-attn==2.7.3 --no-build-isolation",
        "Installing flash-attention..."
    ):
        print("\n⚠️  Warning: flash-attention installation failed")
        print("You may continue without it, but performance may be affected")
    else:
        print("\n✓ Flash-attention installed successfully")
    
    # Final verification
    print("\n" + "="*60)
    print("INSTALLATION COMPLETE - VERIFYING")
    print("="*60)
    
    modules_to_check = [
        ("torch", "PyTorch"),
        ("vllm", "vLLM"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("einops", "Einops"),
    ]
    
    all_ok = True
    for module, name in modules_to_check:
        if verify_import(module):
            print(f"✓ {name} imported successfully")
        else:
            print(f"❌ {name} import failed")
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ INSTALLATION SUCCESSFUL!")
        print("="*60)
        print("\nYou can now use DeepSeek-OCR. Try running:")
        print("  cd DeepSeek-OCR-master/DeepSeek-OCR-vllm")
        print("  python run_dpsk_ocr_image.py")
    else:
        print("⚠️  INSTALLATION COMPLETED WITH WARNINGS")
        print("="*60)
        print("\nSome packages may not have been installed correctly.")
        print("Please check the error messages above.")
    
    print("\n" + "="*60)
    print("Issue #296 Fix Applied:")
    print("- PyTorch is now installed BEFORE vLLM")
    print("- This ensures xformers can find torch during build")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
