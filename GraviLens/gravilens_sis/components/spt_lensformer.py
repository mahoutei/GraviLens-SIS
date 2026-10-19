# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
from typing import Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ShiftPatchTokenizer(nn.Module):
    """
    Shifted Patch Tokenizer from Lensformer (Velôso et al., 2023).
    Supports 'diagonal' and 'rectangular' translation modes.
    """
    def __init__(self, image_size: int, embedding_size: int, patch_size: int,
                 num_channels: int, translation_mode: str,
                 device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.translation_mode = translation_mode
        self.device = (torch.device(device)
                       if isinstance(device, str) else device)

        self.num_patches = (image_size // patch_size) ** 2
        self.total_channels = num_channels * 5

        self.tokenizer = nn.Conv2d(
            in_channels=self.total_channels,
            out_channels=embedding_size,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size)
        )

        self.shift_x = patch_size // 2
        self.shift_y = patch_size // 2

        self.class_embedding = nn.Parameter(
            torch.zeros(1, 1, embedding_size))
        self.positional = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embedding_size))

    def _translate(self, image: torch.Tensor,
                   dx: int, dy: int) -> torch.Tensor:
        return TF.affine(img=image, angle=0, translate=(dx, dy),
                         scale=1, shear=0).to(self.device)

    def get_num_patches(self) -> int:
        return self.num_patches

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device)
        B = image.shape[0]
        sx, sy = self.shift_x, self.shift_y

        if self.translation_mode == "diagonal":
            patches_cat = torch.cat([
                image,
                self._translate(image, -sx, -sy),
                self._translate(image,  sx, -sy),
                self._translate(image, -sx,  sy),
                self._translate(image,  sx,  sy),
            ], dim=1)
        elif self.translation_mode == "rectangular":
            patches_cat = torch.cat([
                image,
                self._translate(image,   0,  sy),
                self._translate(image,   0, -sy),
                self._translate(image, -sx,   0),
                self._translate(image,  sx,   0),
            ], dim=1)
        else:
            raise ValueError(f"Unknown translation_mode: {self.translation_mode}")

        embeddings = self.tokenizer(patches_cat).flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(B, -1, -1)
        patches = torch.cat([cls, embeddings], dim=1)
        return patches + self.positional
