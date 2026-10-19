# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
from typing import Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ShiftedPatchTokenization(nn.Module):
    """
    Shifted Patch Tokenization with diagonal 4-shift augmentation.
    Used exclusively by GraviLens. Disables Conv2d bias per Swin-SPT convention.
    """
    def __init__(self, image_size: Union[int, Tuple[int, int]],
                 patch_size: Union[int, Tuple[int, int]],
                 embed_dim: int, in_channels: int, device: torch.device):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.embed_dim = embed_dim
        self.image_width, self.image_height = image_size
        self.patch_width, self.patch_height = patch_size
        self.device = device

        if self.image_width % self.patch_width != 0:
            raise ValueError("Image size must be divisible by patch size.")

        self.num_patches = int((self.image_width // self.patch_width)
                               * (self.image_height // self.patch_height))
        self.delta_x = self.patch_width // 2
        self.delta_y = self.patch_height // 2
        total_channels = in_channels * 5

        self.projection = nn.Conv2d(
            in_channels=total_channels,
            out_channels=embed_dim,
            kernel_size=(self.patch_width, self.patch_height),
            stride=(self.patch_width, self.patch_height)
        ).to(device)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim))

        if self.projection.bias is not None:
            self.register_buffer('projection_bias', self.projection.bias.data)
            self.projection.bias = None

    def _translate(self, image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        return TF.affine(image, angle=0, translate=(dx, dy),
                         fill=0, scale=1, shear=0).to(self.device)

    def get_num_patches(self) -> int:
        return self.num_patches

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.size(0)
        dx, dy = self.delta_x, self.delta_y
        shifted = torch.cat([
            image,
            self._translate(image, -dx, -dy),
            self._translate(image,  dx, -dy),
            self._translate(image, -dx,  dy),
            self._translate(image,  dx,  dy),
        ], dim=1)

        patches = self.projection(shifted).flatten(2).transpose(1, 2)
        patches = self.layer_norm(patches)
        cls = self.cls_token.expand(B, -1, -1)
        patches = torch.cat([cls, patches], dim=1)
        return patches + self.positional_encoding
