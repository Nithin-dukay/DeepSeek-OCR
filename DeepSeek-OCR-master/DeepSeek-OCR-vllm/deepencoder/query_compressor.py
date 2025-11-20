"""
Query-based Vision Token Compressor inspired by DETR decoder architecture.

This module uses learnable queries with cross-attention to compress variable-length
vision tokens into a fixed number of tokens, addressing GitHub Issue #271.

Key features:
1. Fixed number of output tokens (resolution-independent)
2. Controllable computational cost
3. Multi-scale information aggregation through iterative cross-attention
4. Support for different query counts (64, 128, 256, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from flash_attn import flash_attn_func


class QueryBasedCompressor(nn.Module):
    """
    DETR-style query-based compressor that uses learnable queries to extract
    a fixed number of vision tokens from variable-length vision features.
    
    Args:
        num_queries (int): Fixed number of output tokens (e.g., 64, 128, 256)
        hidden_dim (int): Hidden dimension of the model
        num_layers (int): Number of cross-attention layers (default: 6)
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim
        dropout (float): Dropout rate
        use_flash_attn (bool): Whether to use flash attention
    """
    
    def __init__(
        self,
        num_queries: int = 128,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        
        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        nn.init.normal_(self.query_embed, std=0.02)
        
        # Positional encoding for queries (optional, helps with ordering)
        self.query_pos_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        nn.init.normal_(self.query_pos_embed, std=0.02)
        
        # Stack of transformer decoder layers
        self.layers = nn.ModuleList([
            QueryCompressorLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_flash_attn=use_flash_attn,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: Vision features from encoder [B, N, C] where N is variable
            
        Returns:
            compressed_features: Fixed-size features [B, num_queries, C]
        """
        B = vision_features.shape[0]
        
        # Expand queries for batch
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, C]
        query_pos = self.query_pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, C]
        
        # Apply transformer decoder layers
        for layer in self.layers:
            queries = layer(
                queries=queries,
                query_pos=query_pos,
                vision_features=vision_features,
            )
        
        # Final normalization
        queries = self.norm(queries)
        
        return queries


class QueryCompressorLayer(nn.Module):
    """
    Single layer of the query-based compressor.
    
    Architecture:
    1. Self-attention on queries (with positional encoding)
    2. Cross-attention: queries attend to vision features
    3. Feed-forward network
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_flash_attn = use_flash_attn
        
        # Self-attention on queries
        self.self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-attention: queries attend to vision features
        self.cross_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward network
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        queries: torch.Tensor,
        query_pos: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries: Query embeddings [B, num_queries, C]
            query_pos: Positional encoding for queries [B, num_queries, C]
            vision_features: Vision features from encoder [B, N, C]
            
        Returns:
            Updated queries [B, num_queries, C]
        """
        # Self-attention with positional encoding
        q = queries + query_pos
        queries = queries + self.dropout1(self.self_attn(self.norm1(q)))
        
        # Cross-attention: queries attend to vision features
        queries = queries + self.dropout2(
            self.cross_attn(
                query=self.norm2(queries + query_pos),
                key_value=vision_features,
            )
        )
        
        # Feed-forward network
        queries = queries + self.mlp(self.norm3(queries))
        
        return queries


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_flash_attn = use_flash_attn
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C]
            
        Returns:
            Output tensor [B, N, C]
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        if self.use_flash_attn and x.is_cuda:
            # Use flash attention for efficiency
            try:
                # flash_attn expects [B, N, 3, num_heads, head_dim]
                out = flash_attn_func(
                    qkv[:, :, 0],  # Q
                    qkv[:, :, 1],  # K
                    qkv[:, :, 2],  # V
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                )
                out = out.reshape(B, N, C)
            except:
                # Fallback to standard attention
                q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
                out = out.transpose(1, 2).reshape(B, N, C)
        else:
            # Standard scaled dot-product attention
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
            out = out.transpose(1, 2).reshape(B, N, C)
        
        out = self.out_proj(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention module where queries attend to key-value pairs."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_flash_attn = use_flash_attn
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Query tensor [B, N_q, C]
            key_value: Key-value tensor [B, N_kv, C]
            
        Returns:
            Output tensor [B, N_q, C]
        """
        B, N_q, C = query.shape
        N_kv = key_value.shape[1]
        
        # Compute Q from queries
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim)
        
        # Compute K, V from key_value
        kv = self.kv_proj(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]
        
        if self.use_flash_attn and query.is_cuda:
            # Use flash attention for efficiency
            try:
                out = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                )
                out = out.reshape(B, N_q, C)
            except:
                # Fallback to standard attention
                q = q.transpose(1, 2)  # [B, num_heads, N_q, head_dim]
                k = k.transpose(1, 2)  # [B, num_heads, N_kv, head_dim]
                v = v.transpose(1, 2)  # [B, num_heads, N_kv, head_dim]
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
                out = out.transpose(1, 2).reshape(B, N_q, C)
        else:
            # Standard scaled dot-product attention
            q = q.transpose(1, 2)  # [B, num_heads, N_q, head_dim]
            k = k.transpose(1, 2)  # [B, num_heads, N_kv, head_dim]
            v = v.transpose(1, 2)  # [B, num_heads, N_kv, head_dim]
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
            out = out.transpose(1, 2).reshape(B, N_q, C)
        
        out = self.out_proj(out)
        return out


def build_query_compressor(
    num_queries: int = 128,
    hidden_dim: int = 1024,
    num_layers: int = 6,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    use_flash_attn: bool = True,
) -> QueryBasedCompressor:
    """
    Factory function to build a query-based compressor.
    
    Args:
        num_queries: Number of output tokens (64, 128, 256, etc.)
        hidden_dim: Hidden dimension
        num_layers: Number of cross-attention layers (default: 6 as in DETR)
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
        use_flash_attn: Whether to use flash attention
        
    Returns:
        QueryBasedCompressor module
    """
    return QueryBasedCompressor(
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        use_flash_attn=use_flash_attn,
    )


if __name__ == "__main__":
    # Test the query compressor
    print("Testing Query-Based Compressor...")
    
    # Create compressor with 128 queries
    compressor = build_query_compressor(
        num_queries=128,
        hidden_dim=1024,
        num_layers=6,
        num_heads=16,
    )
    
    # Test with different input sizes (simulating different resolutions)
    test_cases = [
        (2, 256, 1024),   # Small resolution: 512x512 -> 16x16 patches
        (2, 1024, 1024),  # Medium resolution: 1024x1024 -> 32x32 patches
        (2, 4096, 1024),  # Large resolution: 2048x2048 -> 64x64 patches
    ]
    
    for B, N, C in test_cases:
        vision_features = torch.randn(B, N, C)
        output = compressor(vision_features)
        print(f"Input: [{B}, {N}, {C}] -> Output: {list(output.shape)}")
        assert output.shape == (B, 128, C), f"Expected shape ({B}, 128, {C}), got {output.shape}"
    
    print("\n✅ All tests passed! Query compressor produces fixed-size output regardless of input resolution.")
    
    # Calculate parameters
    total_params = sum(p.numel() for p in compressor.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
