# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange

class MultiLocallySelfAttention(nn.Module):
    """
    Locally Self-Attention (LSA): standard MHA with diagonal attention mask.
    The identity mask (True on diagonal) blocks intra-token self-attention,
    forcing each token to attend only to other tokens.
    """
    def __init__(self, embedding_dim: int, num_heads: int, num_patches: int,
                 device: Union[str, torch.device], dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)
        attn_mask = torch.eye(1 + num_patches, 1 + num_patches,
                              dtype=torch.bool, requires_grad=False)
        self.register_buffer('attention_mask', attn_mask)
        self.mha = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, key: torch.Tensor, query: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        x, _ = self.mha(query, key, value,
                        attn_mask=self.attention_mask)
        return x


class FeedForwardBlockLSA(nn.Module):
    """
    FeedForward block used inside Lensformer/LensPINN transformer layers.
    Emits raw logits for multiclass (Identity final activation) to avoid
    the double-softmax error present in the original Velôso et al. code.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int,
                 num_hidden_layers: int, activation_function,
                 task_type: str, dropout: float = 0.1):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            activation_function(),
            nn.Dropout(dropout)
        ]
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       activation_function(), nn.Dropout(dropout)]

        layers.append(nn.Linear(hidden_dim, out_dim))

        if task_type == "regression":
            layers.append(nn.Identity())
        elif task_type == "binary":
            if out_dim != 1:
                raise ValueError("Binary classification requires out_dim=1.")
            layers.append(nn.Sigmoid())
        elif task_type == "multiclass":
            layers.append(nn.Identity())
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerLSABlock(nn.Module):
    """
    Transformer block using MultiLocallySelfAttention.
    Supports cross-attention: key/query/value can differ (used by Lensformer).
    """
    def __init__(self, embedding_dim: int, num_head: int, num_patches: int,
                 num_hidden_neurons: int, num_hidden_layers: int,
                 activation_function, device: Union[str, torch.device],
                 dropout: float = 0.1):
        super().__init__()
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)
        self.mlsa = MultiLocallySelfAttention(
            embedding_dim, num_head, num_patches, device, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForwardBlockLSA(
            in_dim=embedding_dim, out_dim=embedding_dim,
            hidden_dim=num_hidden_neurons,
            num_hidden_layers=num_hidden_layers,
            activation_function=activation_function,
            task_type="regression", dropout=dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1))
        self.query_T = nn.Linear(embedding_dim, embedding_dim)
        self.value_T = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, key: torch.Tensor,
                query: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None) -> torch.Tensor:
        key = key.to(self.device)
        if query is None:
            query = (self.query_T(key) / self.temperature).to(self.device)
        if value is None:
            value = self.value_T(key)

        value = value + self.mlsa(key, query, value)
        value = self.norm1(value)
        value = value + self.feedforward(value)
        value = self.norm2(value)
        return self.dropout(value)


class LSA(nn.Module):
    """
    Locally Self-Attention for ViTSD (Lee et al., 2021).
    Learnable temperature + masked diagonal to suppress self-attention bias.
    Pre-norm is applied internally before the QKV projection, consistent
    with the updated baseline (Lee et al., 2021, revised).
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(
            torch.log(torch.tensor(dim_head ** -0.5)))
        self.norm    = nn.LayerNorm(dim)
        self.attend  = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out  = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
