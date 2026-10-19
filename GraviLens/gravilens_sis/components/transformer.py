# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch
import torch.nn as nn

from .wsma import WindowAttention

class TransformerBlock(nn.Module):
    """Single Swin-style block: window attention + MLP + CLS-pass-through."""
    def __init__(self, dim: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 4., dropout: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim), nn.Dropout(dropout)
        )

    @staticmethod
    def _partition(x: torch.Tensor, ws: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)

    @staticmethod
    def _reverse(windows: torch.Tensor, ws: int,
                 H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / ws / ws))
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def forward(self, x: torch.Tensor, H: int, W: int,
                key=None, value=None) -> torch.Tensor:
        B, L, C = x.shape
        cls_token  = x[:, 0:1, :]
        x_patches  = x[:, 1:, :]
        shortcut   = x_patches
        x_patches  = self.norm1(x_patches).view(B, H, W, C)

        windows = self._partition(x_patches, self.window_size)
        attn_win = self.attn(windows)
        x_patches = self._reverse(attn_win, self.window_size, H, W)
        x_patches = x_patches.view(B, H * W, C)
        x_patches = shortcut + x_patches
        x = torch.cat([cls_token, x_patches], dim=1)
        return x + self.mlp(self.norm2(x))
