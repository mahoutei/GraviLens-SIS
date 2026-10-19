# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch
import torch.nn as nn

from .transformer import TransformerBlock

class HierarchicalTransformerStage(nn.Module):
    """Stack of TransformerBlocks with cross-attention key/value injection."""
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4., dropout: float = 0.):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, window_size, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, H: int, W: int,
                key=None, value=None) -> torch.Tensor:
        if H % self.window_size != 0 or W % self.window_size != 0:
            raise ValueError(
                f"H={H}, W={W} must be divisible by window_size={self.window_size}.")
        for blk in self.blocks:
            x = blk(x, H, W, key, value)
        return x
