import torch
import torch.nn as nn

from gravilens_sis.physics.sis_deflection import SimpleSISDeflection
from gravilens_sis.physics.sis_encoder import SISPhysicsInformedEncoder
from gravilens_sis.physics.sis_predictor import SimpleSISPredictor


def test_sis_deflection_finite():
    dev = SimpleSISDeflection()
    g = torch.linspace(-1, 1, 32)
    x, y = torch.meshgrid(g, g, indexing='xy')
    for k in (0.7, 1.4):
        dx, dy = dev.calculate_deflection(x, y, torch.tensor(k))
        for d in (dx, dy):
            assert d.shape == x.shape and torch.isfinite(d).all()


def test_physics_modules_instantiate():
    p = SimpleSISPredictor(embed_dim=16, image_size=64, device=torch.device('cpu'))
    assert isinstance(p, nn.Module)
    e = SISPhysicsInformedEncoder(image_size=64, patch_size=8, embed_dim=16,
                                  num_patches=64, num_heads=4, num_hidden_neurons=32,
                                  activation_function=nn.GELU, num_transformer_blocks=1,
                                  device=torch.device('cpu'))
    assert isinstance(e, nn.Module)
