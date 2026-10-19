# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import random

import numpy as np
import torch
import torch.nn as nn

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total:,} ({total / 1e6:.2f}M)")
    return total / 1e6
