# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import random
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def _cait_dropout_layers(layers: nn.ModuleList,
                          dropout: float) -> list:
    """
    Stochastic depth (Huang et al., 2016): randomly skip transformer layers
    during training. At least one layer is always kept.
    """
    if dropout == 0:
        return list(layers)
    num_layers = len(layers)
    to_drop    = torch.zeros(num_layers).uniform_(0.0, 1.0) < dropout
    if all(to_drop):
        to_drop[random.randrange(num_layers)] = False
    return [layer for layer, drop in zip(layers, to_drop) if not drop]


class _CaiTLayerScale(nn.Module):
    """
    Layer-scale (Touvron et al., 2021): element-wise learnable gain applied
    after each sub-module.  Initial value follows the depth schedule in §2
    of the paper: ε = 0.1 (≤18 layers), 1e-5 (19–24), 1e-6 (>24).
    """
    def __init__(self, dim: int, fn: nn.Module, depth: int):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6
        self.scale = nn.Parameter(torch.zeros(1, 1, dim).fill_(init_eps))
        self.fn    = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) * self.scale


class _CaiTPreNorm(nn.Module):
    """Apply LayerNorm before the wrapped module (pre-norm convention)."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class _CaiTFeedForward(nn.Module):
    """MLP block for CaiT.  No internal norm — norm lives in _CaiTPreNorm."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CaiTAttention(nn.Module):
    """
    CaiT attention with talking-heads mixing (Shazeer et al., 2020).
    Accepts an optional `context` tensor for the CLS cross-attention stage:
    query comes from CLS tokens, key/value from patch tokens.
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q  = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.mix_heads_pre_attn  = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _, h = *x.shape, self.heads
        kv_src = x if context is None else torch.cat((x, context), dim=1)

        q    = self.to_q(x)
        k, v = self.to_kv(kv_src).chunk(2, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # Talking-heads: mix across head dimension pre- and post-softmax.
        dots = torch.einsum('b h i j, h g -> b g i j',
                            dots, self.mix_heads_pre_attn)
        attn = self.attend(dots)
        attn = torch.einsum('b h i j, h g -> b g i j',
                            attn, self.mix_heads_post_attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class _CaiTEncoder(nn.Module):
    """
    CaiT transformer stage: LayerScale-wrapped (PreNorm(Attention) +
    PreNorm(FFN)) blocks with optional stochastic depth.
    Used for both patch-to-patch and CLS cross-attention passes.
    """
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, dropout: float = 0.,
                 layer_dropout: float = 0.):
        super().__init__()
        self.layer_dropout = layer_dropout
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                _CaiTLayerScale(
                    dim,
                    _CaiTPreNorm(dim, _CaiTAttention(dim, heads=heads,
                                                     dim_head=dim_head,
                                                     dropout=dropout)),
                    depth=ind + 1),
                _CaiTLayerScale(
                    dim,
                    _CaiTPreNorm(dim, _CaiTFeedForward(dim, mlp_dim,
                                                        dropout=dropout)),
                    depth=ind + 1),
            ]))

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        layers = (_cait_dropout_layers(self.layers, self.layer_dropout)
                  if self.training and self.layer_dropout > 0
                  else list(self.layers))
        for attn, ff in layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class CaiTModel(nn.Module):
    """
    Class-Attention in Image Transformers (CaiT).
    Reference: Touvron et al., 2021.
    Data-driven baseline — no physics prior.

    Two-stage design:
      1. Patch transformer  — self-attention among patch tokens.
      2. CLS transformer    — cross-attention from CLS token(s) to patches,
                              with LayerScale + stochastic depth.

    Talking-heads mixing and layer-scale stabilise training without warmup.
    """
    def __init__(self, image_size: int = 64, patch_size: int = 32,
                 num_classes: int = 3, dim: int = 1024,
                 depth: int = 2, cls_depth: int = 1, heads: int = 16,
                 mlp_dim: int = 16, dim_head: int = 64,
                 dropout: float = 0.1, emb_dropout: float = 0.1,
                 layer_dropout: float = 0.0, channels: int = 1):
        super().__init__()
        assert image_size % patch_size == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim   = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding     = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token         = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout           = nn.Dropout(emb_dropout)

        self.patch_transformer = _CaiTEncoder(dim, depth, heads, dim_head,
                                              mlp_dim, dropout, layer_dropout)
        self.cls_transformer   = _CaiTEncoder(dim, cls_depth, heads, dim_head,
                                              mlp_dim, dropout, layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(1)

        x       = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        # Stage 1: patch-to-patch self-attention.
        x = self.patch_transformer(x)

        # Stage 2: CLS cross-attention over frozen patch representations.
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = self.cls_transformer(cls_tokens, context=x)

        return self.mlp_head(x[:, 0])
