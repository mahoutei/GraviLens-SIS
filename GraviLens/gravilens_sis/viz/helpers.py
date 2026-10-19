# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import contextlib as _contextlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from ..training.loops import _dispatch_forward

try:  # optional plotting style used in the original notebook environment
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'ieee', 'no-latex'])
except ImportError:
    pass

_MANUSCRIPT_RC = {
    'font.size':              8,
    'axes.titlesize':         9,
    'axes.labelsize':         8,
    'xtick.labelsize':        7,
    'ytick.labelsize':        7,
    'legend.fontsize':        7,
    'legend.framealpha':      0.85,
    'legend.borderpad':       0.35,
    'legend.handlelength':    1.4,
    'legend.handletextpad':   0.4,
    'legend.columnspacing':   1.0,
    'lines.linewidth':        1.2,
    'lines.markersize':       3.5,
    'axes.linewidth':         0.6,
    'xtick.major.width':      0.5,
    'ytick.major.width':      0.5,
    'xtick.major.size':       3.0,
    'ytick.major.size':       3.0,
    'grid.linewidth':         0.4,
    'grid.alpha':             0.3,
    'figure.dpi':             150,
    'savefig.dpi':            600,
    'savefig.bbox':           'tight',
    'savefig.pad_inches':     0.04,
}


@_contextlib.contextmanager
def _ms_rc():
    """Apply manuscript RC params and restore on exit."""
    with plt.rc_context(_MANUSCRIPT_RC):
        yield


def _ms_save(fig, save_path, tight=True):
    """tight_layout + save + show + close."""
    if tight:
        fig.tight_layout(pad=0.4)
    if save_path:
        fig.savefig(save_path)
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


def _ms_collect_probs(model, model_type, test_loader, device):
    """Return (probs [N×C], labels [N]) from a test loader."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, dists, lbls in test_loader:
            imgs  = imgs.to(device)
            dists = dists.to(device)
            logits = _dispatch_forward(model, model_type, imgs, dists)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(lbls.numpy())
    return np.vstack(all_probs), np.array(all_labels)


def _ms_binarize(labels, n_classes):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    return y_bin


def _ms_conf_matrix(model, model_type, test_loader, device, class_names):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, dists, lbls in test_loader:
            imgs  = imgs.to(device)
            dists = dists.to(device)
            out   = _dispatch_forward(model, model_type, imgs, dists)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
    return confusion_matrix(labels, preds, normalize='true')


def _ms_auc_bars(ax, models_dict, auc_table, class_names, annot_size=5.5):
    """Grouped bar chart for per-class AUC, shared by ROC functions."""
    plotted  = [cn for cn in auc_table if auc_table[cn]]
    if not plotted:
        return
    all_vals = [v for vals in auc_table.values() for v in vals if vals]
    y_min    = max(0.0, min(all_vals) - 0.02) if all_vals else 0.9
    y_max    = min(1.0, max(all_vals) + 0.01) if all_vals else 1.0
    x        = np.arange(len(plotted))
    n        = len(models_dict)
    bar_w    = min(0.14, 0.8 / n)
    for mi, (mkey, _) in enumerate(models_dict.items()):
        heights = [auc_table[cn][mi] if mi < len(auc_table[cn]) else 0.0
                   for cn in plotted]
        offset  = (mi - n / 2 + 0.5) * bar_w
        bars    = ax.bar(x + offset, heights, bar_w,
                         label=DISPLAY_NAMES.get(mkey, mkey),
                         color=MODEL_COLORS.get(mkey),
                         alpha=0.85, edgecolor='white', linewidth=0.4)
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + (y_max - y_min) * 0.006,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=annot_size, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(plotted)
    ax.set_ylabel('AUC')
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.22)
    ax.legend(fontsize=5.5, ncol=2)
    ax.grid(True, axis='y')


DISPLAY_NAMES = {
    'gravilens':      'GraviLens',
    'lensformer':     'Lensformer',
    'lenspinn_small': 'LensPINN-Small',
    'vit':            'ViT',
    'cait':           'CaiT',
    'vitsd':          'ViTSD',
}


MODEL_COLORS = {
    'gravilens':      '#1f77b4',
    'lensformer':     '#ff7f0e',
    'lenspinn_small': '#9467bd',
    'vit':            '#2ca02c',
    'cait':           '#d62728',
    'vitsd':          '#8c564b',
}


MODEL_CATEGORIES = {
    'gravilens':      'PINN',
    'lensformer':     'PINN',
    'lenspinn_small': 'PINN',
    'vit':            'Data-Driven',
    'cait':           'Data-Driven',
    'vitsd':          'Data-Driven',
}


MODEL_LINESTYLES = {
    'gravilens':      '-',
    'lensformer':     '-',
    'lenspinn_small': '-',
    'vit':            '--',
    'cait':           '--',
    'vitsd':          '--',
}
