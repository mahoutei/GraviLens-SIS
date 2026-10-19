# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import math

import torch
import torch.nn as nn

from ..components.htransformer import HierarchicalTransformerStage
from .sis_deflection import SimpleSISDeflection
from .sis_predictor import SimpleSISPredictor

class SISPhysicsInformedEncoder(nn.Module):
    """GraviLens encoder: computes inverse-warp source reconstruction."""
    def __init__(self, image_size: int, patch_size: int, embed_dim: int,
                 num_patches: int, num_heads: int, num_hidden_neurons: int,
                 activation_function, num_transformer_blocks: int,
                 device: torch.device, window_size: int = 4):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.pixel_scale = 0.101
        self.H = self.W = int(math.sqrt(num_patches))

        if self.H % window_size != 0:
            for ws in [2, 4, 8]:
                if self.H % ws == 0:
                    window_size = ws
                    break

        half = (image_size - 1) / 2
        coords_1d = torch.linspace(-half, half, image_size,
                                   device=device) * self.pixel_scale
        gx, gy = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
        self.flat_grid_x = gx.flatten()
        self.flat_grid_y = gy.flatten()

        self.sis_deflection = SimpleSISDeflection(eps=1e-6, core_radius=0.01)
        self.sis_predictor  = SimpleSISPredictor(
            embed_dim, image_size=image_size, device=device)
        self.transformer = HierarchicalTransformerStage(
            dim=embed_dim, depth=num_transformer_blocks,
            num_heads=num_heads, window_size=window_size,
            mlp_ratio=4., dropout=0.1)

    def forward(self, input_images: torch.Tensor,
                patches: torch.Tensor) -> torch.Tensor:
        B = input_images.shape[0]

        transformer_out = self.transformer(patches, self.H, self.W)
        k_map   = self.sis_predictor(transformer_out)
        theta_E = self.sis_predictor._theta_E
        k_flat  = theta_E.expand(-1, self.flat_grid_x.shape[0])

        alpha_x, alpha_y = self.sis_deflection.calculate_deflection(
            self.flat_grid_x.unsqueeze(0).expand(B, -1),
            self.flat_grid_y.unsqueeze(0).expand(B, -1),
            k_flat)

        source_x = self.flat_grid_x.unsqueeze(0) - alpha_x
        source_y = self.flat_grid_y.unsqueeze(0) - alpha_y

        max_coord = ((self.image_size - 1) / 2) * self.pixel_scale
        src_px = torch.clamp(
            ((source_x + max_coord) / (2 * max_coord)
             * (self.image_size - 1)).round().long(),
            0, self.image_size - 1)
        src_py = torch.clamp(
            ((source_y + max_coord) / (2 * max_coord)
             * (self.image_size - 1)).round().long(),
            0, self.image_size - 1)

        flat_imgs = input_images.view(B, -1)
        flat_idx  = src_px * self.image_size + src_py

        source_flat  = torch.zeros(B, self.image_size ** 2, device=self.device)
        count_flat   = torch.zeros(B, self.image_size ** 2, device=self.device)
        ones         = torch.ones_like(flat_idx, dtype=torch.float32)

        source_flat = source_flat.scatter_add(1, flat_idx, flat_imgs)
        count_flat  = count_flat.scatter_add(1, flat_idx, ones)

        mask = count_flat > 0
        source_flat[mask] /= count_flat[mask]

        max_val = source_flat.amax(dim=1, keepdim=True).clamp(min=1e-8)
        source_image = (source_flat / max_val).view(
            B, 1, self.image_size, self.image_size)
        return source_image
