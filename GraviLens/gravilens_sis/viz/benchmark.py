# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import time as _time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve

from ..training.loops import _dispatch_forward
from .helpers import (_ms_auc_bars, _ms_binarize, _ms_collect_probs,
                      _ms_conf_matrix, _ms_rc, _ms_save,
                      DISPLAY_NAMES, MODEL_COLORS, MODEL_CATEGORIES,
                      MODEL_LINESTYLES)

def plot_all_models_learning_curves(all_results, save_path=None):
    """
    Figure 4.2 — 1×3 overlay of val loss / F1 / accuracy for all six models.
    PINNs solid, data-driven dashed. (6.5 × 3.0 in)
    """
    metrics = [('val_loss', 'Validation loss'),
               ('val_f1',   'Validation F1'),
               ('val_acc',  'Validation accuracy')]
    with _ms_rc():
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.0))
        fig.suptitle(
            'All models — validation curves: PINNs (solid) vs '
            'data-driven (dashed)',
            fontsize=9, fontweight='bold', y=1.01)
        for ax, (key, ylabel) in zip(axes, metrics):
            for mkey, res in all_results.items():
                if key not in res:
                    continue
                ax.plot(res[key],
                        label=DISPLAY_NAMES.get(mkey, mkey),
                        color=MODEL_COLORS.get(mkey),
                        linestyle=MODEL_LINESTYLES.get(mkey, '-'),
                        linewidth=1.2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontsize=8)
            ax.legend(fontsize=6, loc='best')
            ax.grid(True)
        # Category legend on rightmost axis
        pinn_h = mpatches.Patch(facecolor='none', edgecolor='gray',
                                 linewidth=1.0, label='PINNs (solid)')
        dd_h   = mpatches.Patch(facecolor='none', edgecolor='gray',
                                 linewidth=1.0, linestyle='--',
                                 label='Data-driven (dashed)')
        existing = axes[2].get_legend().legend_handles if axes[2].get_legend() else []
        axes[2].legend(handles=existing + [pinn_h, dd_h],
                       fontsize=6, loc='best')
        _ms_save(fig, save_path)


def plot_all_models_training_curves_grid(all_results, save_path=None):
    """
    Figure 4.3 — 2×3 per-model training-dynamics grid.
    Row 0: PINNs | Row 1: Data-driven. (6.5 × 4.5 in)
    """
    pinn_keys = [k for k in all_results
                 if MODEL_CATEGORIES.get(k, '') == 'PINN']
    dd_keys   = [k for k in all_results
                 if MODEL_CATEGORIES.get(k, '') == 'Data-Driven']
    n_cols    = max(len(pinn_keys), len(dd_keys))
    with _ms_rc():
        fig, axes = plt.subplots(2, n_cols, figsize=(6.5, 4.5))
        fig.suptitle(
            'All models — individual training dynamics\n'
            'Row 1: PINNs  |  Row 2: Data-driven',
            fontsize=9, fontweight='bold', y=1.01)

        def _panel(ax, mkey):
            res    = all_results[mkey]
            color  = MODEL_COLORS.get(mkey)
            epochs = np.arange(1, len(res['val_loss']) + 1)
            ax2    = ax.twinx()
            ax.plot(epochs, res['train_loss'], color=color, linewidth=0.9,
                    linestyle='--', alpha=0.55, label='Train loss')
            ax.plot(epochs, res['val_loss'],   color=color, linewidth=1.2,
                    linestyle='-',  alpha=0.85, label='Val loss')
            ax2.plot(epochs, res['val_f1'],    color=color, linewidth=1.2,
                     linestyle='-.', alpha=0.85, label='Val F1')
            ax.set_xlabel('Epoch', fontsize=7)
            ax.set_ylabel('Loss', fontsize=7, color=color)
            ax2.set_ylabel('Val F1', fontsize=7, color=color)
            cat = MODEL_CATEGORIES.get(mkey, '')
            ax.set_title(f"{DISPLAY_NAMES.get(mkey, mkey)}\n[{cat}]",
                         fontsize=8, fontweight='bold', pad=2)
            ax.grid(True)
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, fontsize=5.5, loc='upper right',
                      handlelength=1.2)
            ax.tick_params(labelsize=7)
            ax2.tick_params(labelsize=7)

        for col, mkey in enumerate(pinn_keys):
            _panel(axes[0, col], mkey)
        for col, mkey in enumerate(dd_keys):
            _panel(axes[1, col], mkey)
        for col in range(len(pinn_keys), n_cols):
            axes[0, col].set_visible(False)
        for col in range(len(dd_keys), n_cols):
            axes[1, col].set_visible(False)
        _ms_save(fig, save_path)


def plot_all_models_roc(models_dict, test_loader, device,
                         class_names=None, save_path=None):
    """
    Figure 4.4 — macro-average ROC + per-class AUC bar chart for all
    six models. PINNs solid, data-driven dashed. (6.5 × 3.5 in)
    """
    if class_names is None:
        class_names = ['Axion', 'CDM', 'No-sub']
    auc_table = {n: [] for n in class_names}
    with _ms_rc():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
        axes[0].set_title('Macro-average ROC — all models',
                          fontsize=9, fontweight='bold')
        axes[1].set_title('Per-class AUC — all models',
                          fontsize=9, fontweight='bold')
        for mkey, (model, mtype) in models_dict.items():
            probs, labels = _ms_collect_probs(model, mtype, test_loader, device)
            n_cls   = probs.shape[1]
            names_m = (class_names[:n_cls] if n_cls <= len(class_names)
                       else [str(i) for i in range(n_cls)])
            for nm in names_m:
                if nm not in auc_table:
                    auc_table[nm] = []
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
            cat = MODEL_CATEGORIES.get(mkey, '')
            axes[0].plot(all_fpr, mean_tpr,
                         color=MODEL_COLORS.get(mkey),
                         linestyle=MODEL_LINESTYLES.get(mkey, '-'),
                         linewidth=1.3,
                         label=f"{DISPLAY_NAMES.get(mkey, mkey)} "
                               f"[{cat}] ({macro_auc:.3f})")
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
        axes[0].set_xlabel('False positive rate')
        axes[0].set_ylabel('True positive rate')
        axes[0].legend(fontsize=5.5, loc='lower right')
        axes[0].grid(True)
        _ms_auc_bars(axes[1], models_dict, auc_table, class_names,
                     annot_size=5.5)
        _ms_save(fig, save_path)


def plot_all_models_confusion_matrices(models_dict, test_loader, device,
                                        class_names=None, save_path=None):
    """
    Figure 4.5 — 2×3 normalised confusion matrices.
    Row 0: PINNs | Row 1: Data-driven. (6.5 × 4.6 in)
    """
    import seaborn as sns
    if class_names is None:
        class_names = ['Axion', 'CDM', 'No-sub']
    pinn_keys = [k for k in models_dict
                 if MODEL_CATEGORIES.get(k, '') == 'PINN']
    dd_keys   = [k for k in models_dict
                 if MODEL_CATEGORIES.get(k, '') == 'Data-Driven']
    n_cols    = max(len(pinn_keys), len(dd_keys))
    with _ms_rc():
        fig, axes = plt.subplots(2, n_cols, figsize=(6.5, 4.6))
        fig.suptitle(
            'All models — confusion matrices (normalised)\n'
            'Row 1: PINNs  |  Row 2: Data-driven',
            fontsize=9, fontweight='bold', y=1.01)

        def _draw(ax, mkey, model, mtype):
            cm = _ms_conf_matrix(model, mtype, test_loader, device, class_names)
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        vmin=0.0, vmax=1.0, ax=ax,
                        annot_kws={'size': 7},
                        linewidths=0.3, linecolor='#cccccc',
                        cbar_kws={'shrink': 0.88, 'pad': 0.02})
            cat = MODEL_CATEGORIES.get(mkey, '')
            ax.set_title(f"{DISPLAY_NAMES.get(mkey, mkey)}\n[{cat}]",
                         fontsize=8, fontweight='bold', pad=2)
            ax.set_xlabel('Predicted', fontsize=7)
            ax.set_ylabel('True', fontsize=7)
            ax.tick_params(labelsize=7)

        for col, mkey in enumerate(pinn_keys):
            m, mt = models_dict[mkey]
            _draw(axes[0, col], mkey, m, mt)
        for col, mkey in enumerate(dd_keys):
            m, mt = models_dict[mkey]
            _draw(axes[1, col], mkey, m, mt)
        for col in range(len(pinn_keys), n_cols):
            axes[0, col].set_visible(False)
        for col in range(len(dd_keys), n_cols):
            axes[1, col].set_visible(False)
        _ms_save(fig, save_path)


def plot_pinn_vs_datadriven_summary(models_dict, all_results,
                                     test_loader, device,
                                     save_path=None):
    """
    1×3 summary: (a) best val F1 bar, (b) test accuracy bar,
    (c) complexity-vs-performance scatter. (6.5 × 3.0 in)
    """
    model_keys    = list(models_dict.keys())
    display_names = [DISPLAY_NAMES.get(k, k) for k in model_keys]
    colors        = [MODEL_COLORS.get(k) for k in model_keys]
    categories    = [MODEL_CATEGORIES.get(k, '') for k in model_keys]

    best_val_f1s, test_accs, params_M = [], [], []
    for mkey, (model, mtype) in models_dict.items():
        res = all_results.get(mkey, {})
        best_val_f1s.append(max(res.get('val_f1', [0.0])))
        preds_all, labels_all = [], []
        model.eval()
        with torch.no_grad():
            for imgs, dists, lbls in test_loader:
                imgs  = imgs.to(device)
                dists = dists.to(device)
                out   = _dispatch_forward(model, mtype, imgs, dists)
                preds_all.extend(out.argmax(1).cpu().numpy())
                labels_all.extend(lbls.numpy())
        from sklearn.metrics import accuracy_score as _acc
        test_accs.append(_acc(labels_all, preds_all))
        params_M.append(sum(p.numel() for p in model.parameters()
                            if p.requires_grad) / 1e6)

    n_pinns   = sum(1 for c in categories if c == 'PINN')
    divider_x = n_pinns - 0.5
    x         = np.arange(len(model_keys))

    with _ms_rc():
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.0))
        fig.suptitle('PINNs vs data-driven — performance summary',
                     fontsize=9, fontweight='bold', y=1.01)
        for ax, values, ylabel, title, letter in zip(
                axes,
                [best_val_f1s, test_accs, None],
                ['Best validation F1', 'Test accuracy', ''],
                ['Best val F1 by model', 'Test accuracy by model',
                 'Complexity vs performance'],
                ['a', 'b', 'c']):
            if letter == 'c':
                for i, mkey in enumerate(model_keys):
                    ax.scatter(params_M[i], best_val_f1s[i],
                               color=colors[i], s=40, zorder=3,
                               marker='o' if categories[i] == 'PINN' else 's',
                               edgecolors='black', linewidth=0.5)
                    ax.annotate(display_names[i],
                                (params_M[i], best_val_f1s[i]),
                                textcoords='offset points', xytext=(4, 3),
                                fontsize=5.5)
                ax.set_xlabel('Parameters (M)')
                ax.set_ylabel('Best validation F1')
                ax.set_xscale('log')
                ax.set_title('(c) Complexity vs performance',
                             fontsize=8, fontweight='bold')
                ax.grid(True, which='both')
            else:
                bars = ax.bar(x, values, color=colors, alpha=0.85,
                              edgecolor='black', linewidth=0.5)
                for bar, val, cat in zip(bars, values, categories):
                    if cat == 'PINN':
                        bar.set_hatch('//')
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f'{val:.4f}', ha='center', va='bottom',
                            fontsize=5.5)
                ax.axvline(x=divider_x, color='gray',
                           linestyle=':', linewidth=1.0)
                ylim = ax.get_ylim()
                ax.text(divider_x - 0.5, ylim[0] + 0.01,
                        'PINNs', ha='center', fontsize=6, color='gray')
                ax.text(divider_x + 0.5, ylim[0] + 0.01,
                        'Data-driven', ha='center', fontsize=6, color='gray')
                ax.set_xticks(x)
                ax.set_xticklabels(display_names, rotation=30, ha='right',
                                   fontsize=6.5)
                ax.set_ylabel(ylabel)
                ax.set_title(f'({letter}) {title}',
                             fontsize=8, fontweight='bold')
                ax.set_ylim(0, 1.12)
                ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.7)
                ax.grid(True, axis='y')
        pinn_p = mpatches.Patch(facecolor='gray', hatch='//',
                                 edgecolor='black', label='PINN')
        dd_p   = mpatches.Patch(facecolor='gray', edgecolor='black',
                                 label='Data-driven')
        axes[0].legend(handles=[pinn_p, dd_p], fontsize=6, loc='upper right')
        _ms_save(fig, save_path)


def plot_inference_benchmark(models_dict, device,
                              image_size=64,
                              batch_sizes=None,
                              n_warmup=10,
                              n_runs=100,
                              save_path=None):
    """
    Figure 4.6 — latency (ms/image) and throughput (images/s) vs batch size
    for all six models. PINNs solid, data-driven dashed. (6.5 × 3.0 in)
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    use_cuda = (device.type == 'cuda')
    bench = {}
    for mkey, (model, mtype) in models_dict.items():
        model.eval()
        lat_ms, tput = [], []
        for bs in batch_sizes:
            d_img  = torch.randn(bs, 1, image_size, image_size, device=device)
            d_dist = torch.randn(bs, 1, image_size, image_size, device=device)
            with torch.no_grad():
                for _ in range(n_warmup):
                    _dispatch_forward(model, mtype, d_img, d_dist)
            if use_cuda:
                torch.cuda.synchronize(device)
            if use_cuda:
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                with torch.no_grad():
                    t0.record()
                    for _ in range(n_runs):
                        _dispatch_forward(model, mtype, d_img, d_dist)
                    t1.record()
                torch.cuda.synchronize(device)
                elapsed = t0.elapsed_time(t1)
            else:
                t_s = _time.perf_counter()
                with torch.no_grad():
                    for _ in range(n_runs):
                        _dispatch_forward(model, mtype, d_img, d_dist)
                elapsed = (_time.perf_counter() - t_s) * 1e3
            ms_img = elapsed / (n_runs * bs)
            lat_ms.append(ms_img)
            tput.append(1000.0 / ms_img)
            print(f"    {DISPLAY_NAMES.get(mkey, mkey):<18} "
                  f"bs={bs:>2}  {ms_img:6.3f} ms  "
                  f"{1000.0/ms_img:8.1f} img/s")
        bench[mkey] = {'latency': lat_ms, 'throughput': tput}

    dev_label = 'GPU' if use_cuda else 'CPU'
    with _ms_rc():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
        fig.suptitle(
            f'Inference benchmark  ({dev_label}, '
            f'{n_runs} runs/point, {image_size}×{image_size} px)',
            fontsize=9, fontweight='bold', y=1.01)
        for mkey, data in bench.items():
            color = MODEL_COLORS.get(mkey)
            ls    = MODEL_LINESTYLES.get(mkey, '-')
            dname = DISPLAY_NAMES.get(mkey, mkey)
            axes[0].plot(batch_sizes, data['latency'],
                         marker='o', markersize=3.5, linewidth=1.2,
                         color=color, linestyle=ls, label=dname)
            axes[1].plot(batch_sizes, data['throughput'],
                         marker='s', markersize=3.5, linewidth=1.2,
                         color=color, linestyle=ls, label=dname)
        for ax, ylabel, title in zip(
                axes,
                ['Latency (ms / image)', 'Throughput (images / s)'],
                ['Latency vs batch size', 'Throughput vs batch size']):
            ax.set_xscale('log', base=2)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([str(b) for b in batch_sizes])
            ax.set_xlabel('Batch size')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=8, fontweight='bold')
            ax.legend(fontsize=6, loc='best')
            ax.grid(True, which='both')
        _ms_save(fig, save_path)
    return bench
