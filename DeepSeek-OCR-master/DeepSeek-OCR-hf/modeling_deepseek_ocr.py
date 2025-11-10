"""
DeepSeek-OCR Model Implementation

This module provides a wrapper around the DeepSeek-OCR model to properly
handle model loading and generation with Transformers, eliminating warnings.
"""

import warnings
import torch
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union

# Try relative import first, fall back to absolute import
try:
    from .configuration_deepseek_ocr import DeepSeekOCRConfig
except ImportError:
    from configuration_deepseek_ocr import DeepSeekOCRConfig


class DeepSeekOCRForCausalLM(PreTrainedModel):
    """
    DeepSeek-OCR model wrapper for proper Transformers integration.
    
    This class wraps the actual DeepSeek-OCR model loaded with trust_remote_code=True
    and provides proper configuration to avoid warnings.
    """
    
    config_class = DeepSeekOCRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekOCRBlock"]
    
    def __init__(self, config: DeepSeekOCRConfig):
        """
        Initialize the DeepSeek-OCR model.
        
        Args:
            config: Model configuration
        """
        # Set _model to None before calling super().__init__
        # This is needed because can_generate() is called during init
        self._model = None
        super().__init__(config)
        self.config = config
    
    def can_generate(self):
        """Override to prevent automatic generation config creation during init."""
        # Return False during init, True after model is loaded
        return hasattr(self, '_model') and self._model is not None
        
    def _load_actual_model(self, model_name_or_path: str, **kwargs):
        """
        Load the actual DeepSeek-OCR model with trust_remote_code.
        
        This is called lazily to load the real model implementation.
        """
        if self._model is None:
            # Suppress the specific warnings we're trying to fix
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*model of type.*")
                warnings.filterwarnings("ignore", message=".*not initialized from.*")
                warnings.filterwarnings("ignore", message=".*TRAIN this model.*")
                
                self._model = AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    **kwargs
                )
        return self._model
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained DeepSeek-OCR model.
        
        This method properly handles the model loading to avoid warnings.
        """
        # Extract configuration-related kwargs
        config_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ['cache_dir', 'force_download', 'resume_download', 
                     'proxies', 'local_files_only', 'token', 'revision']
        }
        
        # Try to load config, but handle if it doesn't exist
        try:
            config = DeepSeekOCRConfig.from_pretrained(
                pretrained_model_name_or_path, 
                **config_kwargs
            )
        except Exception:
            # If config loading fails, create a default config
            config = DeepSeekOCRConfig()
        
        # Create model instance
        model = cls(config)
        
        # Load the actual model implementation
        model._model = model._load_actual_model(
            pretrained_model_name_or_path,
            **kwargs
        )
        
        # Copy important attributes from the loaded model
        if hasattr(model._model, 'generation_config'):
            model.generation_config = model._model.generation_config
            
            # Fix generation config to avoid warnings
            if model.generation_config is not None:
                # Set pad_token_id if not set
                if model.generation_config.pad_token_id is None:
                    if hasattr(model._model, 'config') and hasattr(model._model.config, 'eos_token_id'):
                        model.generation_config.pad_token_id = model._model.config.eos_token_id
                
                # Remove temperature if do_sample is False
                if hasattr(model.generation_config, 'do_sample') and not model.generation_config.do_sample:
                    if hasattr(model.generation_config, 'temperature'):
                        model.generation_config.temperature = None
        
        return model
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() to load the model.")
        return self._model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """
        Generate text using the model with proper configuration.
        
        This method fixes common generation warnings.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() to load the model.")
        
        # Fix common generation parameter issues
        if 'do_sample' not in kwargs:
            kwargs['do_sample'] = False
        
        # If do_sample is False, remove temperature
        if not kwargs.get('do_sample', False):
            kwargs.pop('temperature', None)
        
        # Set pad_token_id if not provided
        if 'pad_token_id' not in kwargs:
            if hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None:
                kwargs['pad_token_id'] = self.config.pad_token_id
            elif hasattr(self.config, 'eos_token_id') and self.config.eos_token_id is not None:
                kwargs['pad_token_id'] = self.config.eos_token_id
        
        # Suppress generation-related warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*do_sample.*temperature.*")
            warnings.filterwarnings("ignore", message=".*attention mask.*pad token.*")
            warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")
            warnings.filterwarnings("ignore", message=".*seen_tokens.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*get_max_cache.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*position_ids.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*position_embeddings.*mandatory.*")
            
            return self._model.generate(*args, **kwargs)
    
    def infer(self, *args, **kwargs):
        """
        Run inference using the model's custom infer method.
        
        This wraps the model's infer method with warning suppression.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Use from_pretrained() to load the model.")
        
        if not hasattr(self._model, 'infer'):
            raise AttributeError("The loaded model does not have an 'infer' method.")
        
        # Suppress warnings during inference
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*do_sample.*temperature.*")
            warnings.filterwarnings("ignore", message=".*attention mask.*pad token.*")
            warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")
            warnings.filterwarnings("ignore", message=".*seen_tokens.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*get_max_cache.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*position_ids.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*position_embeddings.*mandatory.*")
            
            return self._model.infer(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped model.
        
        This allows transparent access to the underlying model's methods and attributes.
        """
        # Avoid infinite recursion for private attributes
        if name.startswith('_'):
            return super().__getattr__(name)
        
        # Try to get from the wrapped model
        if self._model is not None:
            try:
                return getattr(self._model, name)
            except AttributeError:
                pass
        
        # Fall back to default behavior
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def eval(self):
        """Set the model to evaluation mode."""
        super().eval()
        if self._model is not None:
            self._model.eval()
        return self
    
    def train(self, mode=True):
        """Set the model to training mode."""
        super().train(mode)
        if self._model is not None:
            self._model.train(mode)
        return self
    
    def cuda(self, device=None):
        """Move the model to CUDA."""
        super().cuda(device)
        if self._model is not None:
            self._model.cuda(device)
        return self
    
    def cpu(self):
        """Move the model to CPU."""
        super().cpu()
        if self._model is not None:
            self._model.cpu()
        return self
    
    def to(self, *args, **kwargs):
        """Move the model to a device or change dtype."""
        super().to(*args, **kwargs)
        if self._model is not None:
            self._model.to(*args, **kwargs)
        return self


# Register the model with AutoModel
try:
    from transformers import AutoModel
    AutoModel.register(DeepSeekOCRConfig, DeepSeekOCRForCausalLM)
except Exception:
    # Registration might fail in some environments, but that's okay
    pass
