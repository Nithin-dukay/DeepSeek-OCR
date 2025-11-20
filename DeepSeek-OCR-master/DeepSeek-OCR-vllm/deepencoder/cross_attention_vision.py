import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CrossAttentionVision(nn.Module):
    """
    Cross-attention module for vision token compression.
    Uses learnable queries to extract fixed number of vision tokens from variable-length vision features.
    Inspired by DETR's object detection approach.
    """

    def __init__(self, embed_dim: int = 1280, num_queries: int = 6, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Learnable query vectors
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Cross-attention layers
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed-forward network
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, seq_len, embed_dim] - flattened vision features

        Returns:
            compressed_tokens: [batch_size, num_queries, embed_dim] - fixed number of tokens
        """
        batch_size = vision_features.size(0)

        # Expand queries for batch processing
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_queries, embed_dim]

        # Cross-attention: queries attend to vision features
        attn_output, _ = self.cross_attn(
            query=queries,
            key=vision_features,
            value=vision_features
        )

        # Add & norm
        attn_output = self.norm1(attn_output + queries)

        # Feed-forward
        ffn_output = self.ffn(attn_output)
        output = self.norm2(ffn_output + attn_output)

        return output


class VisionTokenCompressor(nn.Module):
    """
    Complete vision token compression module that combines SAM and CLIP features
    and compresses them to fixed number of tokens using cross-attention.
    """

    def __init__(self, embed_dim: int = 1280, num_queries: int = 6, num_heads: int = 8):
        super().__init__()
        self.cross_attention = CrossAttentionVision(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_heads=num_heads
        )

    def forward(self, sam_features: torch.Tensor, clip_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sam_features: [batch_size, seq_len, sam_dim] - SAM vision features
            clip_features: [batch_size, seq_len, clip_dim] - CLIP vision features

        Returns:
            compressed_tokens: [batch_size, num_queries, embed_dim] - fixed compressed tokens
        """
        # Concatenate SAM and CLIP features as in original implementation
        combined_features = torch.cat([
            clip_features[:, 1:],  # Remove CLS token from CLIP
            sam_features.flatten(2).permute(0, 2, 1)  # Flatten spatial dims
        ], dim=-1)

        # Apply cross-attention to get fixed number of tokens
        compressed_tokens = self.cross_attention(combined_features)

        return compressed_tokens