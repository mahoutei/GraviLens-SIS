import numpy as np
import torch
import torch.nn as nn

from gravilens_sis.utils import count_parameters, seed_everything


def test_seed_everything_reproducible():
    seed_everything(7)
    a, t = np.random.rand(3), torch.rand(3)
    seed_everything(7)
    b, u = np.random.rand(3), torch.rand(3)
    assert np.allclose(a, b) and torch.allclose(t, u)


def test_count_parameters_counts_linear():
    assert count_parameters(nn.Linear(4, 3)) == 15 / 1e6  # returns millions
