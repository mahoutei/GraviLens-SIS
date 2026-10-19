# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """MLP used as GraviLens classifier head."""
    def __init__(self, input_dim: int, output_dim: int, activation,
                 hidden_dim: int, num_layers: int,
                 task_type: str = 'multi_classification',
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        current = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(current, hidden_dim),
                       activation(), nn.Dropout(dropout)]
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
