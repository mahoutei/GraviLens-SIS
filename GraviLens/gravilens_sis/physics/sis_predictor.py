# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
from typing import Optional

import torch
import torch.nn as nn

class SimpleSISPredictor(nn.Module):
    """
    Physics-correct SIS kappa-map predictor.
    Predicts scalar Einstein radius theta_E and optional lens-center offset
    from the CLS token, then computes the radially-symmetric convergence map.
    """
    def __init__(self, embed_dim: int, image_size: int = 64,
                 k_min: float = 0.7, k_max: float = 1.4,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.image_size = image_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.k_min, self.k_max = k_min, k_max

        self.global_branch = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1), nn.Sigmoid()
        )
        self.center_branch = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(),
            nn.Linear(128, 2), nn.Tanh()
        )

        half = (image_size - 1) / 2.0
        coords = torch.linspace(-half, half, image_size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        # (consolidation) materialize meshgrid views: torch>=2.x rejects
        # in-place ops (e.g. EMA's lerp_) on stride-0 buffers.
        self.register_buffer('grid_x', grid_x.contiguous())
        self.register_buffer('grid_y', grid_y.contiguous())
        self._theta_E = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        cls = x[:, 0, :]

        theta_E_raw = self.global_branch(cls)
        theta_E = self.k_min + (self.k_max - self.k_min) * theta_E_raw
        self._theta_E = theta_E

        max_offset = (self.image_size - 1) / 2.0 * 0.20
        center = self.center_branch(cls) * max_offset
        x_c, y_c = center[:, 0], center[:, 1]

        dx = self.grid_x.unsqueeze(0) - x_c.view(B, 1, 1)
        dy = self.grid_y.unsqueeze(0) - y_c.view(B, 1, 1)
        r = torch.sqrt(dx ** 2 + dy ** 2 + 1.0)

        k_map = theta_E.view(B, 1, 1) / (2.0 * r)
        return torch.clamp(k_map, self.k_min, self.k_max).unsqueeze(1)
