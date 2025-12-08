"""
Standalone DeepEncoder Reproduction Script
==========================================

This script extracts and reproduces the DeepEncoder functionality from DeepSeek-OCR.
It allows you to input an image and get the encoded embeddings without running the full OCR pipeline.

Usage:
    python reproduce_deepencoder.py --image_path your_image.jpg --output_path output.pt

Author: Solution for GitHub Issue #289
"""

import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple
import sys
import os

# Add the DeepSeek-OCR paths to system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-vllm'))

from deepencoder.sam_vary_sdpa import build_sam_vit_b
from deepencoder.clip_sdpa import build_clip_l
from deepencoder.build_linear import MlpProjector
from addict import Dict


class DeepEncoderStandalone:
    """
    Standalone DeepEncoder for extracting vision embeddings from images.
    
    This class encapsulates the core vision encoding functionality of DeepSeek-OCR,
    allowing independent usage without the full OCR pipeline.
    
    Architecture:
        1. SAM Encoder: Extracts spatial features (1024-dim)
        2. CLIP Encoder: Extracts semantic features (1024-dim)
        3. Feature Concatenation: Combines SAM + CLIP (2048-dim)
        4. MLP Projector: Projects to embedding space (1280-dim)
        5. Special Tokens: Adds layout tokens (newline, separator)
    """
    
    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize the DeepEncoder.
        
        Args:
            model_path: Path to DeepSeek-OCR model weights
            base_size: Base resolution (512/640/1024/1280)
            image_size: Crop resolution for dynamic mode (640/1024)
            crop_mode: Whether to use dynamic cropping
            device: Device to run on ('cuda' or 'cpu')
            dtype: Data type for computation
        """
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.device = device
        self.dtype = dtype
        self.patch_size = 16
        self.downsample_ratio = 4
        
        print(f"Initializing DeepEncoder on {device}...")
        print(f"Configuration: base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")
        
        # Initialize the three core components
        self._init_models()
        
        # Load weights if model_path is provided
        if model_path:
            self._load_weights(model_path)
        
        print("DeepEncoder initialized successfully!")
    
    def _init_models(self):
        """Initialize SAM encoder, CLIP encoder, and MLP projector."""
        # 1. SAM Encoder - for spatial features
        print("Loading SAM encoder...")
        self.sam_model = build_sam_vit_b().to(self.device).to(self.dtype)
        
        # 2. CLIP Encoder - for semantic features
        print("Loading CLIP encoder...")
        self.vision_model = build_clip_l().to(self.device).to(self.dtype)
        
        # 3. MLP Projector - projects concatenated features to embedding space
        print("Loading MLP projector...")
        n_embed = 1280
        projector_config = Dict(
            projector_type="linear",
            input_dim=2048,  # SAM (1024) + CLIP (1024)
            n_embed=n_embed
        )
        self.projector = MlpProjector(projector_config).to(self.device).to(self.dtype)
        
        # 4. Special tokens for layout
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std).to(self.device).to(self.dtype)
        self.view_separator = nn.Parameter(torch.randn(n_embed) * embed_std).to(self.device).to(self.dtype)
    
    def _load_weights(self, model_path: str):
        """Load pretrained weights from HuggingFace or local path."""
        try:
            from transformers import AutoModel
            print(f"Loading weights from {model_path}...")
            
            # Load the full model to extract vision encoder weights
            full_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype
            )
            
            # Extract and load weights for each component
            state_dict = full_model.state_dict()
            
            # Load SAM model weights
            sam_weights = {k.replace('sam_model.', ''): v 
                          for k, v in state_dict.items() if 'sam_model' in k}
            if sam_weights:
                self.sam_model.load_state_dict(sam_weights, strict=False)
                print(f"Loaded {len(sam_weights)} SAM weights")
            
            # Load CLIP model weights
            clip_weights = {k.replace('vision_model.', ''): v 
                           for k, v in state_dict.items() if 'vision_model' in k}
            if clip_weights:
                self.vision_model.load_state_dict(clip_weights, strict=False)
                print(f"Loaded {len(clip_weights)} CLIP weights")
            
            # Load projector weights
            proj_weights = {k.replace('projector.', ''): v 
                           for k, v in state_dict.items() if 'projector' in k}
            if proj_weights:
                self.projector.load_state_dict(proj_weights, strict=False)
                print(f"Loaded {len(proj_weights)} projector weights")
            
            # Load special tokens
            if 'image_newline' in state_dict:
                self.image_newline.data = state_dict['image_newline'].to(self.device).to(self.dtype)
            if 'view_seperator' in state_dict:
                self.view_separator.data = state_dict['view_seperator'].to(self.device).to(self.dtype)
            
            print("Weights loaded successfully!")
            
        except Exception as e:
            print(f"Warning: Could not load weights from {model_path}: {e}")
            print("Using randomly initialized weights.")
    
    def preprocess_image(
        self, 
        image: Image.Image,
        target_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess PIL image to tensor.
        
        Args:
            image: PIL Image
            target_size: Target size for resizing (if None, uses base_size)
            
        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        if target_size is None:
            target_size = self.base_size
        
        # Resize image maintaining aspect ratio
        image = image.convert('RGB')
        w, h = image.size
        
        # Calculate new size maintaining aspect ratio
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Pad to square
        padded = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        padded.paste(image, (0, 0))
        
        # Convert to tensor and normalize
        img_array = np.array(padded).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device).to(self.dtype)
    
    def count_tiles(self, width: int, height: int) -> Tuple[int, int]:
        """
        Calculate number of tiles for dynamic cropping.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            (num_width_tiles, num_height_tiles)
        """
        aspect_ratio = width / height
        
        # Define possible tile configurations
        tile_configs = [
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 1), (2, 2), (2, 3), (2, 4),
            (3, 1), (3, 2), (3, 3), (3, 4),
            (4, 1), (4, 2), (4, 3), (4, 4),
        ]
        
        # Find closest aspect ratio match
        best_config = (1, 1)
        min_diff = float('inf')
        
        for w_tiles, h_tiles in tile_configs:
            config_ratio = w_tiles / h_tiles
            diff = abs(config_ratio - aspect_ratio)
            if diff < min_diff:
                min_diff = diff
                best_config = (w_tiles, h_tiles)
        
        return best_config
    
    @torch.no_grad()
    def encode_image(
        self, 
        image: Image.Image,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Encode a single image to vision embeddings.
        
        This is the CORE DEEPENCODER FUNCTIONALITY that corresponds to
        the `_pixel_values_to_embedding()` method in deepseek_ocr.py.
        
        Args:
            image: PIL Image to encode
            return_intermediate: If True, return intermediate features
            
        Returns:
            Vision embeddings tensor [num_tokens, 1280]
            If return_intermediate=True, returns dict with all intermediate features
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image, self.base_size)
        
        # Step 1: SAM Encoder - Extract spatial features
        print("Running SAM encoder...")
        sam_features = self.sam_model(image_tensor)  # [1, 1024, H/16, W/16]
        
        # Step 2: CLIP Encoder - Extract semantic features
        print("Running CLIP encoder...")
        clip_features = self.vision_model(image_tensor, sam_features)  # [1, seq_len, 1024]
        
        # Step 3: Concatenate features (SAM + CLIP)
        print("Concatenating features...")
        combined_features = torch.cat([
            clip_features[:, 1:],  # Remove CLS token
            sam_features.flatten(2).permute(0, 2, 1)  # Flatten spatial dimensions
        ], dim=-1)  # [1, seq_len, 2048]
        
        # Step 4: Project to embedding space
        print("Projecting to embedding space...")
        embeddings = self.projector(combined_features)  # [1, seq_len, 1280]
        
        # Step 5: Add special layout tokens
        print("Adding layout tokens...")
        _, hw, n_dim = embeddings.shape
        h = w = int(hw ** 0.5)
        
        # Reshape to 2D grid
        embeddings = embeddings.view(h, w, n_dim)
        
        # Add newline token at end of each row
        embeddings = torch.cat([
            embeddings,
            self.image_newline[None, None, :].expand(h, 1, n_dim)
        ], dim=1)
        
        # Flatten back
        embeddings = embeddings.view(-1, n_dim)
        
        # Add view separator at the end
        embeddings = torch.cat([
            embeddings,
            self.view_separator[None, :]
        ], dim=0)
        
        print(f"Encoding complete! Output shape: {embeddings.shape}")
        
        if return_intermediate:
            return {
                'embeddings': embeddings,
                'sam_features': sam_features,
                'clip_features': clip_features,
                'combined_features': combined_features,
                'num_tokens': embeddings.shape[0]
            }
        
        return embeddings
    
    def encode_batch(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Encode a batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of vision embedding tensors
        """
        embeddings_list = []
        for i, image in enumerate(images):
            print(f"\nEncoding image {i+1}/{len(images)}...")
            embeddings = self.encode_image(image)
            embeddings_list.append(embeddings)
        return embeddings_list
    
    def get_num_tokens(self, image: Image.Image) -> int:
        """
        Calculate the number of tokens that will be generated for an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Number of tokens
        """
        w, h = image.size
        
        if self.crop_mode and (w > 640 or h > 640):
            # Dynamic cropping mode
            num_width_tiles, num_height_tiles = self.count_tiles(w, h)
            
            # Global view tokens
            h_global = w_global = int((self.base_size // self.patch_size) / self.downsample_ratio)
            global_tokens = h_global * (w_global + 1)  # +1 for newline tokens
            
            # Local view tokens
            h_local = w_local = int((self.image_size // self.patch_size) / self.downsample_ratio)
            local_tokens = (num_height_tiles * h_local) * (num_width_tiles * w_local + 1)
            
            total_tokens = global_tokens + local_tokens + 1  # +1 for separator
        else:
            # Single view mode
            h_tokens = w_tokens = int((self.base_size // self.patch_size) / self.downsample_ratio)
            total_tokens = h_tokens * (w_tokens + 1) + 1  # +1 for newlines, +1 for separator
        
        return total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce DeepEncoder functionality from DeepSeek-OCR"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="deepencoder_output.pt",
        help="Path to save output embeddings"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Path to DeepSeek-OCR model"
    )
    parser.add_argument(
        "--base_size",
        type=int,
        default=1024,
        choices=[512, 640, 1024, 1280],
        help="Base resolution (Tiny:512, Small:640, Base:1024, Large:1280)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        choices=[640, 1024],
        help="Crop resolution for dynamic mode"
    )
    parser.add_argument(
        "--crop_mode",
        action="store_true",
        help="Enable dynamic cropping (Gundam mode)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate features"
    )
    
    args = parser.parse_args()
    
    # Initialize DeepEncoder
    encoder = DeepEncoderStandalone(
        model_path=args.model_path,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        device=args.device
    )
    
    # Load image
    print(f"\nLoading image from {args.image_path}...")
    image = Image.open(args.image_path)
    print(f"Image size: {image.size}")
    
    # Calculate expected tokens
    num_tokens = encoder.get_num_tokens(image)
    print(f"Expected number of tokens: {num_tokens}")
    
    # Encode image
    print("\n" + "="*50)
    print("Starting encoding process...")
    print("="*50)
    
    result = encoder.encode_image(image, return_intermediate=args.save_intermediate)
    
    if args.save_intermediate:
        embeddings = result['embeddings']
        print("\n" + "="*50)
        print("Encoding Results:")
        print("="*50)
        print(f"SAM features shape: {result['sam_features'].shape}")
        print(f"CLIP features shape: {result['clip_features'].shape}")
        print(f"Combined features shape: {result['combined_features'].shape}")
        print(f"Final embeddings shape: {embeddings.shape}")
        print(f"Number of tokens: {result['num_tokens']}")
        
        # Save all results
        torch.save(result, args.output_path)
        print(f"\nAll results saved to {args.output_path}")
    else:
        embeddings = result
        print("\n" + "="*50)
        print("Encoding Results:")
        print("="*50)
        print(f"Final embeddings shape: {embeddings.shape}")
        print(f"Number of tokens: {embeddings.shape[0]}")
        
        # Save embeddings
        torch.save(embeddings, args.output_path)
        print(f"\nEmbeddings saved to {args.output_path}")
    
    # Print statistics
    print("\n" + "="*50)
    print("Embedding Statistics:")
    print("="*50)
    print(f"Mean: {embeddings.mean().item():.4f}")
    print(f"Std: {embeddings.std().item():.4f}")
    print(f"Min: {embeddings.min().item():.4f}")
    print(f"Max: {embeddings.max().item():.4f}")
    
    print("\n✓ DeepEncoder reproduction complete!")


if __name__ == "__main__":
    main()
