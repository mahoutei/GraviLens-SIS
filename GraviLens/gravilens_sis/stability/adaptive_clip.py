# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch

def adaptive_gradient_clip(model, clip_factor=0.01):
    parameters = [p for p in model.parameters()
                  if p.grad is not None and p.requires_grad]
    if not parameters:
        return 0.0
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach()) for p in parameters])).item()
    param_norm = torch.norm(
        torch.stack([torch.norm(p.detach()) for p in parameters])).item()
    max_norm = clip_factor * param_norm
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return total_norm
