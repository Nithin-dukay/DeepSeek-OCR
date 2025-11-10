"""
Compatibility fix for DeepSeek-OCR with newer transformers versions.
This patches the missing LlamaFlashAttention2 import.

Usage:
    from fix_transformers_compatibility import apply_transformers_compatibility_patch
    apply_transformers_compatibility_patch()
    
    # Now import and use the model
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
"""
import sys
import warnings


def apply_transformers_compatibility_patch():
    """
    Apply compatibility patch for transformers>=4.47.0
    This adds LlamaFlashAttention2 back to the module if it's missing.
    
    This function should be called BEFORE importing the DeepSeek-OCR model.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        # If import succeeds, no patch needed
        print("✓ LlamaFlashAttention2 is available, no patch needed")
        return True
    except ImportError:
        pass
    
    # Patch is needed
    warnings.warn(
        "LlamaFlashAttention2 not found in transformers. Applying compatibility patch. "
        "For best compatibility, consider using transformers==4.46.3",
        UserWarning
    )
    
    try:
        import transformers.models.llama.modeling_llama as llama_module
        
        # Try to find FlashAttention2 implementation or create a fallback
        try:
            from transformers.models.llama.modeling_llama import LlamaAttention
            
            # Create a fallback class that uses standard attention
            class LlamaFlashAttention2(LlamaAttention):
                """
                Fallback implementation when FlashAttention2 is not available.
                Falls back to standard LlamaAttention.
                
                Note: This may impact performance compared to native FlashAttention2.
                """
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    # Only warn once
                    if not hasattr(LlamaFlashAttention2, '_warned'):
                        warnings.warn(
                            "Using fallback LlamaAttention instead of FlashAttention2. "
                            "Performance may be impacted. Consider using transformers==4.46.3",
                            UserWarning
                        )
                        LlamaFlashAttention2._warned = True
            
            # Inject the class into the module
            llama_module.LlamaFlashAttention2 = LlamaFlashAttention2
            
        except ImportError as e:
            # If we can't import LlamaAttention, create a minimal dummy
            warnings.warn(
                f"Could not import LlamaAttention: {e}. Creating minimal fallback.",
                UserWarning
            )
            
            import torch.nn as nn
            
            class LlamaFlashAttention2(nn.Module):
                """Minimal dummy fallback for LlamaFlashAttention2"""
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    warnings.warn(
                        "Using minimal dummy LlamaFlashAttention2. "
                        "This may cause issues. Please use transformers==4.46.3",
                        UserWarning
                    )
            
            llama_module.LlamaFlashAttention2 = LlamaFlashAttention2
        
        print("✓ Transformers compatibility patch applied successfully")
        return True
        
    except Exception as e:
        error_msg = (
            f"Failed to apply transformers compatibility patch: {e}\n"
            "Please install the compatible version: pip install transformers==4.46.3"
        )
        warnings.warn(error_msg, UserWarning)
        return False


def check_transformers_version():
    """
    Check the installed transformers version and provide recommendations.
    """
    try:
        import transformers
        version = transformers.__version__
        
        print(f"Installed transformers version: {version}")
        
        # Parse version
        major, minor, patch = map(int, version.split('.')[:3])
        
        if (major, minor) == (4, 46):
            print("✓ You are using the recommended transformers version (4.46.x)")
            return True
        elif major == 4 and minor >= 47:
            print("⚠ You are using a newer transformers version that may have compatibility issues")
            print("  Recommended: pip install transformers==4.46.3")
            return False
        else:
            print("⚠ You are using an older transformers version")
            print("  Recommended: pip install transformers==4.46.3")
            return False
            
    except Exception as e:
        print(f"Could not check transformers version: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek-OCR Transformers Compatibility Checker")
    print("=" * 60)
    
    # Check version
    is_compatible = check_transformers_version()
    
    print("\n" + "=" * 60)
    
    if not is_compatible:
        print("Applying compatibility patch...")
        success = apply_transformers_compatibility_patch()
        
        if success:
            print("\n✓ Patch applied successfully!")
            print("You can now use DeepSeek-OCR with your current transformers version.")
        else:
            print("\n✗ Patch failed!")
            print("Please install the recommended version:")
            print("  pip install transformers==4.46.3 tokenizers==0.20.3")
    else:
        print("\n✓ Your transformers version is compatible!")
        print("No patch needed.")
    
    print("=" * 60)
