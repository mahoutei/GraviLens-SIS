"""GraviLens command-line interface.

Argparse wrapper around the training functions from the final v9 notebook
(Section 17). Paths that were Colab-drive constants there are arguments here.
"""
import argparse
import inspect

import numpy as np
import torch
import torch.nn as nn

from .configs import (CAIT_CONFIG, GRAVILENS_CONFIG, LENSFORMER_CONFIG,
                      LENSPINN_SMALL_CONFIG, VIT_CONFIG, VITSD_CONFIG)
from .data.dataset import load_numpy_array_from_path
from .data.distortion import precompute_distortions
from .models.cait import CaiTModel
from .models.gravilens import GraviLens
from .models.lensformer import Lensformer
from .models.vit import ViTModel
from .models.vitsd import ViTSDModel
from .training.loops import train_benchmark_model, train_gravilens
from .training.training_manager import SimpleTrialManager
from .utils import count_parameters, seed_everything

# name -> (class, config, model_type key used by the training/eval dispatch,
# transformer+feedforward activation class or None).
# Mirrors the six trials of the recorded v9 benchmark, including the accepted
# checkpoints. Note: the "LensPINN-Small" ablation in the run was implemented
# as a smaller Lensformer (LENSPINN_SMALL_CONFIG, patch=16, embed=384, 2
# blocks) — not the LensPINN class, which the run did not train.
REGISTRY = {
    'gravilens': (GraviLens, GRAVILENS_CONFIG, 'gravilens', nn.GELU),
    'lensformer': (Lensformer, LENSFORMER_CONFIG, 'lensformer', nn.ELU),
    'lenspinn-small': (Lensformer, LENSPINN_SMALL_CONFIG, 'lensformer', nn.ELU),
    'vit': (ViTModel, VIT_CONFIG, 'vit', None),
    'cait': (CaiTModel, CAIT_CONFIG, 'cait', None),
    'vitsd': (ViTSDModel, VITSD_CONFIG, 'vitsd', None),
}


def build_model(cls, cfg, n_classes, device, activation):
    """Instantiate a model from a v9 config, passing only accepted arguments."""
    sig = inspect.signature(cls.__init__).parameters
    kw = {k: v for k, v in cfg.items() if k in sig}
    if 'dropout' in cfg and 'dropout_rate' in sig:
        kw['dropout_rate'] = cfg['dropout']
    extra = dict(device=device, num_classes=n_classes)
    if activation is not None:
        extra['transformer_activation'] = activation
        extra['feedforward_activation'] = activation
    kw.update({k: v for k, v in extra.items() if k in sig})
    return cls(**kw).to(device)


def build_argparser():
    p = argparse.ArgumentParser(prog='gravilens_sis', description=__doc__)
    p.add_argument('--data-dir', required=True,
                   help='directory with data_{train,val,test}.npy and labels_*.npy')
    p.add_argument('--checkpoint-dir', default='./model_checkpoints')
    p.add_argument('--models', nargs='+', choices=sorted(REGISTRY),
                   default=sorted(REGISTRY))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--seed', type=int, default=42)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    seed_everything(args.seed)
    device = torch.device('cuda' if (args.device != 'cpu'
                                     and torch.cuda.is_available()) else 'cpu')
    print(f'  Device: {device}')

    X_train = load_numpy_array_from_path(f'{args.data_dir}/data_train.npy').astype(np.float32)
    y_train = load_numpy_array_from_path(f'{args.data_dir}/labels_train.npy')
    X_val = load_numpy_array_from_path(f'{args.data_dir}/data_val.npy').astype(np.float32)
    y_val = load_numpy_array_from_path(f'{args.data_dir}/labels_val.npy')
    X_test = load_numpy_array_from_path(f'{args.data_dir}/data_test.npy').astype(np.float32)
    y_test = load_numpy_array_from_path(f'{args.data_dir}/labels_test.npy')
    n_classes = y_train.shape[1] if y_train.ndim == 2 else int(y_train.max()) + 1
    print(f'  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} '
          f'| Classes: {n_classes}')

    print('  Precomputing distortion maps (Lensformer / LensPINN)...')
    D_train = precompute_distortions(X_train)
    D_val = precompute_distortions(X_val, verbose=False)
    D_test = precompute_distortions(X_test, verbose=False)

    manager = SimpleTrialManager(args.checkpoint_dir)
    for name in args.models:
        cls, cfg, model_type, activation = REGISTRY[name]
        print(f'  Training {name} ...')
        model = build_model(cls, cfg, n_classes, device, activation)
        count_parameters(model)
        trial = manager.create_trial(f'{name}-v9-benchmark',
                                     config={**cfg, 'model': name,
                                             'n_classes': n_classes})
        if name == 'gravilens':
            results = train_gravilens(model, trial, manager,
                                      X_train, X_val, X_test,
                                      D_train, D_val, D_test,
                                      y_train, y_val, y_test, cfg, device)
        else:
            results = train_benchmark_model(model, model_type, trial, manager,
                                            X_train, X_val, X_test,
                                            D_train, D_val, D_test,
                                            y_train, y_val, y_test, cfg, device)
        print(f'  {name}: test_f1={results.get("test_f1")}')


if __name__ == '__main__':
    main()
