import inspect

import pytest
import torch
import torch.nn as nn

from gravilens_sis import configs
from gravilens_sis.models.cait import CaiTModel
from gravilens_sis.models.gravilens import GraviLens
from gravilens_sis.models.lensformer import Lensformer
from gravilens_sis.models.lenspinn import LensPINN
from gravilens_sis.models.vit import ViTModel
from gravilens_sis.models.vitsd import ViTSDModel

N_CLASSES = 2  # the released v9 benchmark models were trained with 2 output classes

CASES = [
    (GraviLens, configs.GRAVILENS_CONFIG, False),
    (Lensformer, configs.LENSFORMER_CONFIG, True),
    (LensPINN, configs.LENSPINN_SMALL_CONFIG, True),
    (ViTModel, configs.VIT_CONFIG, False),
    (CaiTModel, configs.CAIT_CONFIG, False),
    (ViTSDModel, configs.VITSD_CONFIG, False),
]


def build(cls, cfg):
    sig = inspect.signature(cls.__init__).parameters
    kw = {k: v for k, v in cfg.items() if k in sig}
    extra = dict(device=torch.device('cpu'), num_classes=N_CLASSES,
                 transformer_activation=nn.GELU, feedforward_activation=nn.GELU)
    kw.update({k: v for k, v in extra.items() if k in sig})
    return cls(**kw).eval()


@pytest.mark.parametrize('cls,cfg,needs_d', CASES, ids=lambda v: getattr(v, '__name__', ''))
def test_forward(cls, cfg, needs_d):
    torch.manual_seed(0)
    model = build(cls, cfg)
    img = torch.randn(2, 1, 64, 64)
    dist = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        if cls is GraviLens:
            out = model(img.squeeze(1))          # dispatch: GraviLens(images.squeeze(1))
        elif needs_d:
            out = model(img, dist)[2]            # PINN forwards return (pred, aux, logits)
        else:
            out = model(img)
    assert out.shape == (2, N_CLASSES)
    assert torch.isfinite(out).all()
