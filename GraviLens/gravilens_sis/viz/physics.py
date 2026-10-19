# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import matplotlib.pyplot as plt
import numpy as np
import torch

from .helpers import _ms_rc, _ms_save

def visualize_lens_correction(model, X_test, device,
                               num_samples=4, save_path=None):
    """n×3 grid: original | source-reconstructed | kappa-map."""
    model.eval()
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    row_h = 2.2
    with _ms_rc():
        fig, axes = plt.subplots(num_samples, 3,
                                 figsize=(6.5, row_h * num_samples),
                                 gridspec_kw={'hspace': 0.08, 'wspace': 0.06})
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        col_titles = ['Lensed image', 'Source reconstruction', 'Kappa map']
        for col, t in enumerate(col_titles):
            axes[0, col].set_title(t, fontsize=8, pad=3)
        with torch.no_grad():
            for row, si in enumerate(indices):
                img = torch.tensor(X_test[si], dtype=torch.float32).unsqueeze(0).to(device)
                corrected = model.invert_lens(img)
                k_map     = model.predict_lens_parameters(img)
                for col, (data, cmap) in enumerate([
                    (img.squeeze().cpu().numpy(),       'viridis'),
                    (corrected.squeeze().cpu().numpy(), 'viridis'),
                    (k_map.squeeze().cpu().numpy(),     'hot'),
                ]):
                    im = axes[row, col].imshow(data, cmap=cmap,
                                               origin='lower',
                                               interpolation='nearest')
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    plt.colorbar(im, ax=axes[row, col],
                                 fraction=0.044, pad=0.03,
                                 format='%.2f').ax.tick_params(labelsize=5)
        _ms_save(fig, save_path, tight=False)


def visualize_delensing_comparison(model, X_test, device,
                                    num_samples=6, save_path=None):
    """n×4 grid: lensed | delensed | residual | kappa-map."""
    model.eval()
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    row_h = 2.0
    with _ms_rc():
        fig, axes = plt.subplots(num_samples, 4,
                                 figsize=(6.5, row_h * num_samples),
                                 gridspec_kw={'hspace': 0.08, 'wspace': 0.06})
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        col_titles = ['Lensed', 'Delensed', 'Residual', 'Kappa map']
        col_cmaps  = ['viridis', 'viridis', 'RdBu_r', 'hot']
        for col, t in enumerate(col_titles):
            axes[0, col].set_title(t, fontsize=8, pad=3)
        with torch.no_grad():
            for row, si in enumerate(indices):
                img      = torch.tensor(X_test[si], dtype=torch.float32).unsqueeze(0).to(device)
                delensed = model.invert_lens(img)
                k_map    = model.predict_lens_parameters(img)
                residual = img - delensed
                for col, (data, cmap) in enumerate([
                    (img.squeeze().cpu().numpy(),      col_cmaps[0]),
                    (delensed.squeeze().cpu().numpy(), col_cmaps[1]),
                    (residual.squeeze().cpu().numpy(), col_cmaps[2]),
                    (k_map.squeeze().cpu().numpy(),    col_cmaps[3]),
                ]):
                    im = axes[row, col].imshow(data, cmap=cmap,
                                               origin='lower',
                                               interpolation='nearest')
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    plt.colorbar(im, ax=axes[row, col],
                                 fraction=0.044, pad=0.03,
                                 format='%.2f').ax.tick_params(labelsize=5)
        _ms_save(fig, save_path, tight=False)
