# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import torch.nn as nn

from ..training.loops import evaluate_model_unified
from .helpers import DISPLAY_NAMES, MODEL_CATEGORIES

def print_benchmark_table(models_dict, all_results, test_loader, device):
    """Print a summary table for PINNs vs data-driven (unchanged)."""
    criterion = nn.CrossEntropyLoss()
    print("\n" + "=" * 100)
    print(f"{'BENCHMARK SUMMARY':^100}")
    print("=" * 100)
    print(f"{'Model':<22} {'Category':<14} {'Params (M)':<12} "
          f"{'Best Val F1':<14} {'Test F1':<12} {'Test Acc':<12} {'Test Loss':<12}")
    print("-" * 100)
    for mkey, (model, mtype) in models_dict.items():
        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad) / 1e6
        tm   = evaluate_model_unified(model, mtype, test_loader, criterion, device)
        res  = all_results.get(mkey, {})
        bvf1 = max(res.get('val_f1', [0.0]))
        print(f"{DISPLAY_NAMES.get(mkey, mkey):<22} "
              f"{MODEL_CATEGORIES.get(mkey, 'Unknown'):<14} "
              f"{n_params:<12.2f} {bvf1:<14.4f} "
              f"{tm['f1']:<12.4f} {tm['acc']:<12.4f} {tm['loss']:<12.4f}")
    print("=" * 100)
    print("Metrics: weighted F1 | top-1 accuracy | cross-entropy loss\n")


def print_all_models_table(models_dict, all_results, test_loader, device):
    """Extended benchmark table covering all six models (unchanged)."""
    criterion = nn.CrossEntropyLoss()
    print("\n" + "=" * 110)
    print(f"{'FULL BENCHMARK SUMMARY — ALL MODELS':^110}")
    print("=" * 110)
    print(f"{'Model':<22} {'Category':<14} {'Params (M)':<12} "
          f"{'Best Val F1':<14} {'Test F1':<12} {'Test Acc':<12} "
          f"{'Test Loss':<12}")
    print("-" * 110)
    for category in ('PINN', 'Data-Driven'):
        for mkey, (model, mtype) in models_dict.items():
            if MODEL_CATEGORIES.get(mkey, '') != category:
                continue
            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad) / 1e6
            tm   = evaluate_model_unified(model, mtype, test_loader,
                                          criterion, device)
            res  = all_results.get(mkey, {})
            bvf1 = max(res.get('val_f1', [0.0]))
            print(f"{DISPLAY_NAMES.get(mkey, mkey):<22} {category:<14} "
                  f"{n_params:<12.2f} {bvf1:<14.4f} "
                  f"{tm['f1']:<12.4f} {tm['acc']:<12.4f} {tm['loss']:<12.4f}")
        print("-" * 110)
    print("=" * 110)
    print("Metrics: weighted F1 | top-1 accuracy | cross-entropy loss")
    print("PINNs: GraviLens | Lensformer | LensPINN-Small")
    print("Data-Driven: ViT | CaiT | ViTSD\n")
