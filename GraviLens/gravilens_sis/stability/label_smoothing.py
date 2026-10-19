# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch
import torch.nn as nn

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing (Szegedy et al., 2016)."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))
