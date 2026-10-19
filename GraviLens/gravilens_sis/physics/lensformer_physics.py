# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
from typing import Tuple, Union

import torch
import torch.nn as nn

from ..components.lsa import TransformerLSABlock

def _torch_image_to_source(image: torch.Tensor, E_r: torch.Tensor,
                            image_size: int, device: torch.device,
                            min_angle: float = -3.323,
                            max_angle: float = 3.232) -> torch.Tensor:
    """Differentiable per-image source reconstruction (scatter_add)."""
    pw = (max_angle - min_angle) / image_size
    cx, cy = image_size // 2, image_size // 2
    imones_1d = torch.ones(image_size * image_size, device=device)

    rx = torch.arange(-(cx - 1), image_size - (cx - 1), device=device)
    ry = torch.arange(-(cy - 1), image_size - (cy - 1), device=device)
    x, y = torch.meshgrid(rx, ry, indexing='ij')
    x, y = x * pw, y * pw
    r = torch.sqrt(x ** 2 + y ** 2)
    r[r == 0] = 1.0

    k = torch.ones((image_size, image_size), device=device) * E_r
    xdef = k * x / r
    ydef = k * y / r

    bx = torch.clamp(((x - xdef) / pw + cx), 0, image_size - 1).long()
    by = torch.clamp(((y - ydef) / pw + cy), 0, image_size - 1).long()

    flat_image = image.view(-1)
    flat_bx    = bx.view(-1)
    flat_by    = by.view(-1)
    indices    = flat_bx * image_size + flat_by

    src_flat = torch.zeros(image_size * image_size, device=device)
    cnt_flat = torch.zeros(image_size * image_size, device=device)
    src_flat = src_flat.scatter_add(0, indices, flat_image)
    cnt_flat = cnt_flat.scatter_add(0, indices, imones_1d)
    mask = cnt_flat != 0
    src_flat[mask] /= cnt_flat[mask]
    return src_flat.view(image_size, image_size)


class PhysicsInformedEncoder(nn.Module):
    """Lensformer encoder: pixel-wise Einstein angle map via transformer."""
    def __init__(self, image_size: int, patch_size: int, embedding_dim: int,
                 num_patches: int, num_heads: int, hidden_dim: int,
                 transformer_block_activation_function,
                 num_transformer_block: int,
                 device: Union[str, torch.device], k_max: float, k_min: float,
                 num_hidden_layer: int = 1, dropout: float = 0.1,
                 min_angle: float = -3.323, max_angle: float = 3.232,
                 mag: int = 1):
        super().__init__()
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.k_max, self.k_min = k_max, k_min
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)
        self.min_angle, self.max_angle = min_angle, max_angle

        self.transformer = nn.ModuleList()
        for _ in range(num_transformer_block):
            self.transformer.append(
                TransformerLSABlock(
                    embedding_dim=embedding_dim, num_head=num_heads,
                    num_patches=num_patches,
                    num_hidden_neurons=hidden_dim,
                    num_hidden_layers=num_hidden_layer,
                    activation_function=transformer_block_activation_function,
                    device=device, dropout=dropout))

        total_features = (num_patches + 1) * embedding_dim
        self.transformer.append(nn.Flatten())
        self.transformer.append(
            nn.Linear(total_features, image_size * image_size))

    def forward(self, input_images: torch.Tensor,
                patches: torch.Tensor,
                distortion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_images = input_images.to(self.device)
        distortion   = distortion.to(self.device)
        patches      = patches.to(self.device)
        B = input_images.shape[0]

        k_sigmoid = patches
        for layer in self.transformer:
            k_sigmoid = layer(k_sigmoid)

        k_sigmoid     = k_sigmoid.view(B, self.image_size, self.image_size)
        distortion    = distortion.view(B, self.image_size, self.image_size)
        einstein_angle = k_sigmoid * distortion

        input_images = input_images.view(B, self.image_size, self.image_size)
        source_image = torch.zeros_like(input_images)
        for i in range(B):
            source_image[i] = _torch_image_to_source(
                input_images[i], einstein_angle[i],
                self.image_size, self.device,
                self.min_angle, self.max_angle)

        return k_sigmoid, source_image


class ChangedPhysicsInformedEncoder(nn.Module):
    """LensPINN encoder: scalar Einstein radius prediction."""
    def __init__(self, image_size: int, patch_size: int, embedding_dim: int,
                 num_patches: int, num_heads: int, hidden_dim: int,
                 transformer_block_activation_function,
                 num_transformer_block: int,
                 device: Union[str, torch.device], k_max: float, k_min: float,
                 num_hidden_layer: int = 1, dropout: float = 0.1,
                 min_angle: float = -3.323, max_angle: float = 3.232,
                 mag: int = 1):
        super().__init__()
        self.image_size = image_size
        self.k_max, self.k_min = k_max, k_min
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)
        self.min_angle, self.max_angle = min_angle, max_angle

        self.transformer = nn.ModuleList()
        for _ in range(num_transformer_block):
            self.transformer.append(
                TransformerLSABlock(
                    embedding_dim=embedding_dim, num_head=num_heads,
                    num_patches=num_patches,
                    num_hidden_neurons=hidden_dim,
                    num_hidden_layers=num_hidden_layer,
                    activation_function=transformer_block_activation_function,
                    device=device, dropout=dropout))

        total_features = (num_patches + 1) * embedding_dim
        self.transformer.append(nn.Flatten())
        self.transformer.append(nn.Linear(total_features, 1))

    def forward(self, input_images: torch.Tensor,
                patches: torch.Tensor,
                distortion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_images = input_images.to(self.device)
        patches      = patches.to(self.device)
        B = input_images.shape[0]

        theta_E = patches
        for layer in self.transformer:
            theta_E = layer(theta_E)

        input_images = input_images.view(B, self.image_size, self.image_size)
        source_image = torch.zeros_like(input_images)
        for i in range(B):
            source_image[i] = _torch_image_to_source(
                input_images[i], theta_E[i],
                self.image_size, self.device,
                self.min_angle, self.max_angle)

        return theta_E, source_image
