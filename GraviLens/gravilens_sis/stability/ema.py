# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import copy

import torch

class ModelEMA:
    """Exponential Moving Average of model weights (Tan & Le, 2019)."""
    def __init__(self, model, decay=0.9999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.module.parameters(),
                                      model.parameters()):
                ema_p.lerp_(model_p.detach(), 1.0 - self.decay)
            for ema_b, model_b in zip(self.module.buffers(),
                                      model.buffers()):
                if ema_b.is_floating_point():
                    ema_b.lerp_(model_b.detach().float(), 1.0 - self.decay)
                else:
                    ema_b.copy_(model_b)
