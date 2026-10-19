# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class SPT(nn.Module):
    """
    Shifted Patch Tokenization for ViT and ViTSD.
    Uses einops rearrange + padding shifts — lighter than Conv2d-based SPT.
    Reference: Lee et al., 2021 (ViT for Small Datasets).
    """
    def __init__(self, dim: int, patch_size: int, channels: int = 1):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        return self.to_patch_tokens(x_with_shifts)
