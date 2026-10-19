import numpy as np
import torch
import torch.nn as nn

from gravilens_sis import configs
from gravilens_sis.data.distortion import precompute_distortions
from gravilens_sis.models.gravilens import GraviLens
from gravilens_sis.models.vit import ViTModel
from gravilens_sis.training.loops import train_benchmark_model, train_gravilens
from gravilens_sis.training.training_manager import SimpleTrialManager


def _synthetic(n_train=12, n_val=3, n_test=3, classes=2, seed=1):
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    X = rng.random((n, 64, 64), dtype=np.float32)
    y = np.eye(classes, dtype=np.float32)[rng.integers(0, classes, n)]
    D = precompute_distortions(X, verbose=False)
    return (X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:],
            D[:n_train], D[n_train:n_train + n_val], D[n_train + n_val:],
            y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:])


def test_smoke_benchmark_loop(tmp_path):
    Xtr, Xva, Xte, Dtr, Dva, Dte, ytr, yva, yte = _synthetic()
    model = ViTModel(image_size=64, patch_size=32, num_classes=2, dim=32, depth=1,
                     heads=4, mlp_dim=8, channels=1, dropout=0.0, emb_dropout=0.0)
    cfg = dict(n_epochs=1, batch_size=4, lr=1e-3, weight_decay=0.0, patience=5,
               save_every=99)
    mgr = SimpleTrialManager(str(tmp_path))
    trial = mgr.create_trial('smoke-vit', config={**cfg, 'model': 'ViT'})
    res = train_benchmark_model(model, 'vit', trial, mgr, Xtr, Xva, Xte,
                                Dtr, Dva, Dte, ytr, yva, yte, cfg,
                                torch.device('cpu'))
    assert isinstance(res, dict) and res


def test_smoke_gravilens_loop(tmp_path):
    Xtr, Xva, Xte, Dtr, Dva, Dte, ytr, yva, yte = _synthetic()
    model = GraviLens(image_size=64, patch_size=8, embed_dim=32, in_channels=1,
                      num_classes=2, num_heads=4, num_hidden_neurons=16,
                      num_hidden_layers=1, transformer_activation=nn.GELU,
                      feedforward_activation=nn.GELU, num_transformer_blocks=1,
                      device=torch.device('cpu'), dropout_rate=0.0, window_size=4)
    cfg = dict(configs.GRAVILENS_CONFIG, n_epochs=1, batch_size=4, save_every=99,
               patience=5)
    mgr = SimpleTrialManager(str(tmp_path))
    trial = mgr.create_trial('smoke-gravilens', config={**cfg, 'model': 'GraviLens'})
    res = train_gravilens(model, trial, mgr, Xtr, Xva, Xte, Dtr, Dva, Dte,
                          ytr, yva, yte, cfg, torch.device('cpu'))
    assert isinstance(res, dict) and res
