# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch
import torch.nn as nn
from einops import repeat

from ..components.lsa import LSA
from ..components.spt_vit import SPT

class ViTSDModel(nn.Module):
    """
    Vision Transformer for Small Datasets (ViTSD) with LSA + SPT.
    Reference: Lee et al., 2021.
    Data-driven baseline — no physics prior.
    """
    def __init__(self, image_size: int = 64, patch_size: int = 32,
                 num_classes: int = 3, dim: int = 1024,
                 depth: int = 2, heads: int = 16, mlp_dim: int = 16,
                 channels: int = 1, dropout: float = 0.1,
                 emb_dropout: float = 0.1):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2

        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size,
                                      channels=channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(nn.ModuleList([
                LSA(dim, heads=heads, dim_head=64, dropout=dropout),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(1)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        for attn, ff in self.transformer:
            x = attn(x) + x
            x = ff(x) + x

        return self.mlp_head(x[:, 0])
