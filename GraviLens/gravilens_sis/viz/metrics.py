# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve

from ..training.loops import _dispatch_forward
from .helpers import (_ms_auc_bars, _ms_binarize, _ms_collect_probs,
                      _ms_conf_matrix, _ms_rc, _ms_save,
                      DISPLAY_NAMES, MODEL_CATEGORIES, MODEL_COLORS)

def plot_training_metrics(train_loss, train_f1, train_acc,
                          val_loss, val_f1, val_acc,
                          test_loss, test_f1, test_acc,
                          learning_rates, model_name='GraviLens',
                          save_path=None):
    """2×2 training-curves panel sized for the thesis text column."""
    with _ms_rc():
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2))
        fig.suptitle(f'{model_name} — training curves',
                     fontsize=9, fontweight='bold', y=1.01)
        panels = [
            (train_loss, val_loss,       'Loss'),
            (train_f1,   val_f1,         'F1'),
            (train_acc,  val_acc,         'Accuracy'),
            (learning_rates, None,        'Learning rate'),
        ]
        for ax, (tr, vl, title) in zip(axes.flat, panels):
            ax.plot(tr, linewidth=1.2,
                    label='Train' if vl is not None else 'LR')
            if vl is not None:
                ax.plot(vl, linewidth=1.2, label='Val')
                ax.legend(fontsize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title, fontsize=8)
            ax.grid(True)
        _ms_save(fig, save_path)


def plot_roc_curves(model, test_loader, device, save_path=None):
    """Single-model per-class ROC; square 3.2-in panel."""
    model.eval()
    all_probs, all_labels_raw = [], []
    _all_names = ['Axion', 'CDM', 'No-sub']
    with torch.no_grad():
        for imgs, dists, lbls in test_loader:
            imgs = imgs.to(device)
            dists = dists.to(device)
            out  = _dispatch_forward(model, 'gravilens', imgs, dists)
            all_probs.append(F.softmax(out, dim=1).cpu().numpy())
            all_labels_raw.extend(lbls.numpy())
    all_probs  = np.vstack(all_probs)
    all_labels = np.array(all_labels_raw)
    n_classes  = all_probs.shape[1]
    y_bin      = _ms_binarize(all_labels, n_classes)
    cls_names  = _all_names[:n_classes]

    with _ms_rc():
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
            ax.plot(fpr, tpr, linewidth=1.2,
                    label=f'{cls_names[i]} ({auc(fpr, tpr):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('GraviLens — ROC', fontsize=9, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(True)
        _ms_save(fig, save_path)


def plot_confusion_matrix(model, test_loader, device,
                          class_names=None, save_path=None):
    """Single-model normalised confusion matrix; compact 3.0-in square."""
    import seaborn as sns
    if class_names is None:
        class_names = ['Axion', 'CDM', 'No-sub']
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, dists, lbls in test_loader:
            imgs  = imgs.to(device)
            dists = dists.to(device)
            out   = _dispatch_forward(model, 'gravilens', imgs, dists)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(lbls.numpy())
    cm = confusion_matrix(labels, preds, normalize='true')
    with _ms_rc():
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0.0, vmax=1.0, ax=ax,
                    annot_kws={'size': 7},
                    linewidths=0.3, linecolor='#cccccc',
                    cbar_kws={'shrink': 0.88, 'pad': 0.02})
        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_ylabel('True', fontsize=7)
        ax.set_title('GraviLens — confusion matrix',
                     fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        _ms_save(fig, save_path, tight=False)
        fig.tight_layout(pad=0.3)
        if save_path:
            fig.savefig(save_path)


def calculate_kappa_statistics(kappa_map):
    """Compute summary statistics from a GraviLens kappa-map (unchanged)."""
    flat    = kappa_map.flatten()
    grad_y, grad_x = np.gradient(kappa_map)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    fft   = np.fft.fft2(kappa_map)
    power = np.abs(np.fft.fftshift(fft)) ** 2
    H, W  = kappa_map.shape
    cy, cx = H // 2, W // 2
    r_max  = min(cy, cx)
    low  = power[cy - r_max // 4:cy + r_max // 4,
                 cx - r_max // 4:cx + r_max // 4].mean()
    high = (power.mean() * power.size - low * (r_max // 2) ** 2) / (
        power.size - (r_max // 2) ** 2 + 1e-8)
    ys, xs = np.indices(kappa_map.shape)
    total  = flat.sum() + 1e-8
    cx_c   = (xs.flatten() * flat).sum() / total
    cy_c   = (ys.flatten() * flat).sum() / total
    r_vals = np.sqrt((xs - cx_c) ** 2 + (ys - cy_c) ** 2).flatten()
    return {
        'mean':               float(flat.mean()),
        'std':                float(flat.std()),
        'max':                float(flat.max()),
        'min':                float(flat.min()),
        'mean_gradient':      float(grad_mag.mean()),
        'gradient_variance':  float(grad_mag.var()),
        'rms_variation':      float(np.sqrt(np.mean((flat - flat.mean()) ** 2))),
        'low_freq_power':     float(low),
        'high_freq_power':    float(high),
        'high_to_low_ratio':  float(high / (low + 1e-8)),
        'correlation_length': float((r_vals * flat).sum() / total),
    }


def plot_benchmark_learning_curves(all_results, save_path=None):
    """Overlay val loss / F1 / accuracy for PINNs only. (6.5 × 2.5 in)"""
    pinn_keys = [k for k in all_results
                 if MODEL_CATEGORIES.get(k, '') == 'PINN']
    metrics = [('val_loss', 'Validation loss'),
               ('val_f1',   'Validation F1'),
               ('val_acc',  'Validation accuracy')]
    with _ms_rc():
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))
        fig.suptitle('Validation curves — PINNs',
                     fontsize=9, fontweight='bold', y=1.01)
        for ax, (key, ylabel) in zip(axes, metrics):
            for mkey in pinn_keys:
                res = all_results[mkey]
                if key in res:
                    ax.plot(res[key],
                            label=DISPLAY_NAMES.get(mkey, mkey),
                            color=MODEL_COLORS.get(mkey),
                            linewidth=1.2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontsize=8)
            ax.legend(fontsize=6, loc='best')
            ax.grid(True)
        _ms_save(fig, save_path)


def plot_benchmark_roc(models_dict, test_loader, device,
                       class_names=None, save_path=None):
    """Macro-average ROC + per-class AUC bars for PINNs. (6.5 × 3.2 in)"""
    if class_names is None:
        class_names = ['Axion', 'CDM', 'No-sub']
    pinn_dict = {k: v for k, v in models_dict.items()
                 if MODEL_CATEGORIES.get(k, '') == 'PINN'}
    auc_table = {n: [] for n in class_names}
    with _ms_rc():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
        axes[0].set_title('Macro-average ROC — PINNs',
                          fontsize=9, fontweight='bold')
        axes[1].set_title('Per-class AUC — PINNs',
                          fontsize=9, fontweight='bold')
        for mkey, (model, mtype) in pinn_dict.items():
            probs, labels = _ms_collect_probs(model, mtype, test_loader, device)
            n_cls   = probs.shape[1]
            names_m = class_names[:n_cls]
            y_bin   = _ms_binarize(labels, n_cls)
            all_fpr = np.unique(np.concatenate(
                [roc_curve(y_bin[:, i], probs[:, i])[0] for i in range(n_cls)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_cls):
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i])
                mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
                auc_table[names_m[i]].append(auc(fpr_i, tpr_i))
            mean_tpr /= n_cls
            macro_auc = roc_auc_score(y_bin, probs, average='macro')
            axes[0].plot(all_fpr, mean_tpr,
                         color=MODEL_COLORS.get(mkey), linewidth=1.2,
                         label=f"{DISPLAY_NAMES.get(mkey, mkey)} "
                               f"(AUC={macro_auc:.3f})")
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
        axes[0].set_xlabel('False positive rate')
        axes[0].set_ylabel('True positive rate')
        axes[0].legend(fontsize=6, loc='lower right')
        axes[0].grid(True)
        _ms_auc_bars(axes[1], pinn_dict, auc_table, class_names)
        _ms_save(fig, save_path)


def plot_benchmark_confusion_matrices(models_dict, test_loader, device,
                                       class_names=None, save_path=None):
    """Normalised confusion matrices for PINNs side-by-side. (6.5 × 2.8 in)"""
    import seaborn as sns
    if class_names is None:
        class_names = ['Axion', 'CDM', 'No-sub']
    pinn_dict = {k: v for k, v in models_dict.items()
                 if MODEL_CATEGORIES.get(k, '') == 'PINN'}
    n = len(pinn_dict)
    with _ms_rc():
        fig, axes = plt.subplots(1, n, figsize=(6.5, 2.8))
        if n == 1:
            axes = [axes]
        fig.suptitle('Confusion matrices — PINNs (normalised)',
                     fontsize=9, fontweight='bold', y=1.01)
        for ax, (mkey, (model, mtype)) in zip(axes, pinn_dict.items()):
            cm = _ms_conf_matrix(model, mtype, test_loader, device, class_names)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        vmin=0.0, vmax=1.0, ax=ax,
                        annot_kws={'size': 7},
                        linewidths=0.3, linecolor='#cccccc',
                        cbar_kws={'shrink': 0.85, 'pad': 0.02})
            ax.set_xlabel('Predicted', fontsize=7)
            ax.set_ylabel('True', fontsize=7)
            ax.set_title(DISPLAY_NAMES.get(mkey, mkey),
                         fontsize=8, fontweight='bold')
            ax.tick_params(labelsize=7)
        _ms_save(fig, save_path)
