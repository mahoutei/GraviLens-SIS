import torch

from gravilens_sis.components.lsa import LSA
from gravilens_sis.components.spt_vit import SPT
from gravilens_sis.components.wsma import WindowAttention


def test_spt_forward():
    spt = SPT(dim=32, patch_size=8, channels=1)
    out = spt(torch.randn(2, 1, 64, 64))
    assert out.shape[0] == 2 and out.shape[-1] == 32


def test_window_attention_forward():
    att = WindowAttention(dim=32, num_heads=4, window_size=4)
    x = torch.randn(2 * 16, 16, 32)          # (B*num_windows, window**2, dim)
    out = att(x)
    assert out.shape == x.shape


def test_lsa_forward():
    lsa = LSA(dim=32, heads=4, dim_head=8)
    x = torch.randn(2, 16, 32)
    out = lsa(x)
    assert out.shape == x.shape
