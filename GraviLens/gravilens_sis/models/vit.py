# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class _ViTFeedForward(nn.Module):
    """Pre-norm MLP block for the ViT encoder."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ViTAttention(nn.Module):
    """Standard multi-head self-attention with internal pre-norm."""
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim   = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads  = heads
        self.scale  = dim_head ** -0.5
        self.norm   = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out  = (nn.Sequential(nn.Linear(inner_dim, dim),
                                      nn.Dropout(dropout))
                        if project_out else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = self.norm(x)
        qkv  = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, v)
        out  = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class _ViTEncoder(nn.Module):
    """
    Stack of (_ViTAttention, _ViTFeedForward) blocks with a terminal LayerNorm.
    Residual connections are applied outside each sub-module so that internal
    norms implement pre-norm (rather than post-norm) transformer semantics.
    """
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                _ViTAttention(dim, heads=heads, dim_head=dim_head,
                              dropout=dropout),
                _ViTFeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTModel(nn.Module):
    """
    Vision Transformer (ViT) with dual-LayerNorm patch embedding.
    Reference: Dosovitskiy et al., 2020; updated attention stack from
    Lee et al. (ViT for Small Datasets), 2021.
    Data-driven baseline — no physics prior.
    Supports cls-token ('cls') or mean-pooling ('mean') classification.
    """
    def __init__(self, image_size: int = 64, patch_size: int = 32,
                 num_classes: int = 3, dim: int = 1024,
                 depth: int = 2, heads: int = 16, mlp_dim: int = 16,
                 pool: str = 'cls', channels: int = 1, dim_head: int = 64,
                 dropout: float = 0.1, emb_dropout: float = 0.1):
        super().__init__()
        image_h, image_w = image_size, image_size
        patch_h, patch_w = patch_size, patch_size

        assert image_h % patch_h == 0 and image_w % patch_w == 0, \
            'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, \
            "pool must be 'cls' or 'mean'."

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_dim   = channels * patch_h * patch_w

        # Dual-LayerNorm embedding: normalise raw patches, project, then
        # normalise projected tokens — stabilises early training on small data.
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_h, p2=patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout       = nn.Dropout(emb_dropout)
        self.transformer   = _ViTEncoder(dim, depth, heads, dim_head,
                                         mlp_dim, dropout)
        self.pool          = pool
        self.to_latent     = nn.Identity()
        self.mlp_head      = nn.Linear(dim, num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(1)

        x       = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
