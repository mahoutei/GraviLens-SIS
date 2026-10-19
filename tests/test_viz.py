import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.nn as nn

from gravilens_sis.data.dataset import get_unified_loader
from gravilens_sis.models.gravilens import GraviLens
from gravilens_sis.viz.metrics import plot_confusion_matrix, plot_roc_curves


def test_metric_plots_write_files(tmp_path):
    torch.manual_seed(0)
    # plot_roc_curves / plot_confusion_matrix dispatch with model_type='gravilens'
    # internally, so the model under test must be a GraviLens.
    model = GraviLens(image_size=64, patch_size=8, embed_dim=32, in_channels=1,
                      num_classes=2, num_heads=4, num_hidden_neurons=16,
                      num_hidden_layers=1, transformer_activation=nn.GELU,
                      feedforward_activation=nn.GELU, num_transformer_blocks=1,
                      device=torch.device('cpu'), dropout_rate=0.0, window_size=4)
    rng = np.random.default_rng(0)
    X = rng.random((8, 64, 64), dtype=np.float32)
    D = rng.random((8, 64, 64), dtype=np.float32)
    y = np.eye(2, dtype=np.float32)[rng.integers(0, 2, 8)]
    loader = get_unified_loader(X, D, y, batch_size=4, shuffle=False)
    roc_png, cm_png = tmp_path / 'roc.png', tmp_path / 'cm.png'
    plot_roc_curves(model, loader, torch.device('cpu'), save_path=str(roc_png))
    plot_confusion_matrix(model, loader, torch.device('cpu'),
                          class_names=['Axion', 'CDM'], save_path=str(cm_png))
    assert roc_png.stat().st_size > 0 and cm_png.stat().st_size > 0
