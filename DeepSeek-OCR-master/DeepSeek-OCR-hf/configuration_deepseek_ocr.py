"""
DeepSeek-OCR Model Configuration

This configuration class properly registers the DeepSeek-OCR model type
to avoid model type mismatch warnings when using Transformers.
"""

from transformers import PretrainedConfig
from typing import Optional, Dict, Any


class DeepSeekOCRConfig(PretrainedConfig):
    """
    Configuration class for DeepSeek-OCR model.
    
    This class wraps the DeepSeek-VL-V2 configuration and properly sets
    the model_type to avoid warnings about model type mismatches.
    """
    
    model_type = "deepseek_ocr"
    is_composition = False  # Disable composition to avoid text_config.to_dict() issues
    
    def __init__(
        self,
        # Vision encoder config
        vision_config: Optional[Dict[str, Any]] = None,
        # Projector config
        projector_config: Optional[Dict[str, Any]] = None,
        # Text/Language model config
        text_config: Optional[Dict[str, Any]] = None,
        # Special tokens
        image_token_id: int = 100015,
        # Tile configuration
        tile_tag: str = "2D",
        global_view_pos: str = "tail",
        # Generation config
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        # Other configs
        **kwargs
    ):
        """
        Initialize DeepSeek-OCR configuration.
        
        Args:
            vision_config: Configuration for vision encoders (SAM + CLIP)
            projector_config: Configuration for the MLP projector
            text_config: Configuration for the language model
            image_token_id: Token ID for image tokens
            tile_tag: Tile tagging method (default: "2D")
            global_view_pos: Position of global view (default: "tail")
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            bos_token_id: Beginning of sequence token ID
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs
        )
        
        # Store configuration components
        # Keep them as dicts to avoid issues with nested config objects
        if isinstance(vision_config, dict):
            self.vision_config = vision_config
        else:
            self.vision_config = vision_config or {}
            
        if isinstance(projector_config, dict):
            self.projector_config = projector_config
        else:
            self.projector_config = projector_config or {}
            
        if isinstance(text_config, dict):
            self.text_config = text_config
        else:
            self.text_config = text_config or {}
        
        # Special tokens and settings
        self.image_token_id = image_token_id
        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        
        # Set default pad_token_id to eos_token_id if not provided
        if self.pad_token_id is None and self.eos_token_id is not None:
            self.pad_token_id = self.eos_token_id
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load configuration from a pretrained model.
        
        This method handles loading from models that may have different
        model_type values in their config.json.
        """
        # Try to load the config
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        
        # Handle model_type conversion
        if config_dict.get("model_type") == "deepseek_vl_v2":
            config_dict["model_type"] = "deepseek_ocr"
        
        return cls.from_dict(config_dict, **kwargs)
    
    def to_dict(self):
        """
        Serialize configuration to dictionary.
        """
        output = super().to_dict()
        output["vision_config"] = self.vision_config
        output["projector_config"] = self.projector_config
        output["text_config"] = self.text_config
        output["image_token_id"] = self.image_token_id
        output["tile_tag"] = self.tile_tag
        output["global_view_pos"] = self.global_view_pos
        return output


# Register the configuration with AutoConfig
try:
    from transformers import AutoConfig
    AutoConfig.register("deepseek_ocr", DeepSeekOCRConfig)
except ImportError:
    pass
