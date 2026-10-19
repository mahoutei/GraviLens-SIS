# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch
import torch.nn as nn

class Flatten(nn.Module):
    """Flatten (B, *, *) to (B, num_neurons)."""
    def __init__(self, num_neurons_flatten: int):
        super().__init__()
        self.num_neurons_flatten = num_neurons_flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous().view(-1, self.num_neurons_flatten)
