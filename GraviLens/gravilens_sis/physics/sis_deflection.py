# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
from typing import Tuple

import torch

class SimpleSISDeflection:
    """SIS deflection via potential gradient."""
    def __init__(self, eps: float = 1e-8, core_radius: float = 0.01):
        self.eps = eps
        self.core_radius = core_radius

    def calculate_deflection(
            self, x: torch.Tensor, y: torch.Tensor,
            k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if k.ndim == 1:
            k = k.unsqueeze(-1).expand(-1, x.shape[-1])
        r = torch.sqrt(x ** 2 + y ** 2 + self.core_radius ** 2)
        return k * x / r, k * y / r
