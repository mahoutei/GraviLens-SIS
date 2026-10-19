# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from ..data.dataset import (get_progressive_unified_loader,
                            get_unified_loader)
from ..stability.adaptive_clip import adaptive_gradient_clip
from ..stability.early_stopping import EarlyStopping
from ..stability.ema import ModelEMA
from ..stability.grad_accum import GradientAccumulator
from ..stability.label_smoothing import LabelSmoothingCrossEntropy
from ..stability.lookahead import Lookahead
from ..stability.warmup_cosine import WarmupCosineScheduler
from .training_manager import SimpleTrialManager

def _dispatch_forward(model, model_type: str,
                      images: torch.Tensor,
                      distortions: torch.Tensor) -> torch.Tensor:
    """
    Route forward pass based on model family.

    PINNs:
      - 'gravilens':      GraviLens(images.squeeze(1))
      - 'lensformer':     Lensformer(images, distortions)[2]
      - 'lenspinn_small': LensPINN(images, distortions)[2]

    Data-driven:
      - 'vit', 'cait', 'vitsd': model(images)  [distortions ignored]
    """
    if model_type == 'gravilens':
        return model(images.squeeze(1))
    elif model_type in ('vit', 'cait', 'vitsd'):
        return model(images)
    else:  # lensformer / lenspinn_small
        _, _, logits = model(images, distortions)
        return logits


def evaluate_model_unified(model, model_type: str, loader: DataLoader,
                            criterion: nn.Module,
                            device: torch.device) -> Dict[str, float]:
    """Unified evaluation for all model families."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, distortions, labels in loader:
            images      = images.to(device)
            distortions = distortions.to(device)
            labels      = labels.to(device)

            outputs = _dispatch_forward(model, model_type, images, distortions)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return {
        'loss': total_loss / max(1, len(loader)),
        'f1':   f1_score(all_labels, all_preds, average='weighted'),
        'acc':  accuracy_score(all_labels, all_preds)
    }


def train_gravilens(model, trial_name: str, manager: SimpleTrialManager,
                    X_train, X_val, X_test,
                    D_train, D_val, D_test,
                    y_train, y_val, y_test,
                    config: Dict, device: torch.device):
    """
    GraviLens training loop — v9 convergence-optimised.

    Fixes applied vs v8:
      [FIX 1] scheduler.step(epoch=0) called BEFORE the loop so epoch 0
              sees min_lr, not the raw optimizer default (inverted warmup).
      [FIX 2] warmup_epochs reduced (config: 20 -> 5).
      [FIX 3] initial_lr increased (config: 1e-4 -> 3e-4).
      [FIX 4] accumulation_steps reduced (config: 4 -> 1).
      [FIX 5] Lookahead k read from config (default 3, was hardcoded 5).
      [FIX 6] adaptive_clip_factor relaxed (config: 0.01 -> 0.05).
      [FIX 7] label_smoothing reduced (config: 0.1 -> 0.05).
      [FIX 8] aug_start delays augmentation onset (config default: 10).
      [FIX 9] patience tightened (config: 30 -> 25).
    """
    n_epochs           = config['n_epochs']
    batch_size         = config['batch_size']
    initial_lr         = config['initial_lr']
    warmup_epochs      = config['warmup_epochs']
    accumulation_steps = config['accumulation_steps']
    weight_decay       = config['weight_decay']
    use_ema            = config['use_ema']
    ema_decay          = config['ema_decay']
    use_lookahead      = config['use_lookahead']
    lookahead_k        = config.get('lookahead_k', 3)
    lookahead_alpha    = config.get('lookahead_alpha', 0.5)
    label_smoothing    = config['label_smoothing']
    clip_factor        = config['adaptive_clip_factor']
    patience           = config['patience']
    aug_start          = config.get('aug_start', 10)
    save_every         = config.get('save_every', 5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr,
                                  weight_decay=weight_decay)
    if use_lookahead:
        optimizer = Lookahead(optimizer, k=lookahead_k, alpha=lookahead_alpha)

    # [FIX 1] Initialise scheduler and immediately prime it for epoch 0.
    # Without this call, epoch 0 trains at the raw AdamW lr (initial_lr),
    # then the scheduler plunges to min_lr on the first step() call —
    # an inverted warmup that wastes ~19 epochs of near-zero LR.
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, n_epochs,
                                      min_lr=1e-6, max_lr=initial_lr)
    scheduler.step(epoch=0)   # <-- FIX 1: set LR to min_lr before epoch 0

    criterion      = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    ema        = ModelEMA(model, decay=ema_decay) if use_ema else None
    grad_acc   = GradientAccumulator(accumulation_steps)
    stopper    = EarlyStopping(patience=patience, min_delta=1e-4, mode='min')

    val_loader  = get_unified_loader(X_val,  D_val,  y_val,
                                     batch_size, shuffle=False)
    test_loader = get_unified_loader(X_test, D_test, y_test,
                                     batch_size, shuffle=False)

    train_losses, train_f1s, train_accs = [], [], []
    val_losses,   val_f1s,   val_accs   = [], [], []
    test_losses,  test_f1s,  test_accs  = [], [], []
    learning_rates = []

    # Log starting LR so it's visible that FIX 1 is active.
    _start_lr = optimizer.param_groups[0]['lr']
    print(f"  [v9] Epoch-0 starting LR: {_start_lr:.2e}  "
          f"(warmup target: {initial_lr:.2e} at epoch {warmup_epochs})")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        # [FIX 8] Delayed augmentation onset via aug_start.
        train_loader = get_progressive_unified_loader(
            X_train, D_train, y_train, batch_size,
            epoch, n_epochs, aug_start=aug_start)

        optimizer.zero_grad()
        for batch_idx, (images, distortions, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images.squeeze(1))
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()

            # [FIX 4] With accumulation_steps=1 this fires every batch —
            # same cadence as the baselines (Lensformer etc.).
            if grad_acc.should_step():
                # [FIX 6] Relaxed clip_factor (0.05 vs 0.01) preserves
                # more gradient signal in early training.
                adaptive_gradient_clip(model, clip_factor)
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            with torch.no_grad():
                epoch_loss += eval_criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        train_f1   = f1_score(all_labels, all_preds, average='weighted')
        train_acc  = accuracy_score(all_labels, all_preds)
        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        train_accs.append(train_acc)

        val_metrics  = evaluate_model_unified(
            model, 'gravilens', val_loader, eval_criterion, device)
        test_metrics = evaluate_model_unified(
            model, 'gravilens', test_loader, eval_criterion, device)

        val_losses.append(val_metrics['loss'])
        val_f1s.append(val_metrics['f1'])
        val_accs.append(val_metrics['acc'])
        test_losses.append(test_metrics['loss'])
        test_f1s.append(test_metrics['f1'])
        test_accs.append(test_metrics['acc'])

        # Advance scheduler AFTER the epoch (epoch+1 feeds the warmup formula).
        lr = scheduler.step()
        learning_rates.append(lr)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            aug_note = ("no aug" if epoch < aug_start
                        else f"aug {min(1.0,(epoch-aug_start)/max(1,n_epochs*0.7-aug_start)):.2f}")
            print(f"Epoch {epoch+1:>3}/{n_epochs} | "
                  f"Train {train_loss:.4f} | "
                  f"Val {val_metrics['loss']:.4f} | "
                  f"Val F1 {val_metrics['f1']:.4f} | "
                  f"LR {lr:.2e} | {aug_note}")

        if (epoch + 1) % save_every == 0:
            manager.save_checkpoint(trial_name, model, optimizer, epoch, {
                'train_loss': train_loss, 'train_f1': train_f1,
                'train_acc': train_acc,
                'val_loss': val_metrics['loss'], 'val_f1': val_metrics['f1'],
                'val_acc': val_metrics['acc']
            })

        if stopper(val_metrics['loss']):
            print(f"\n  Early stopping at epoch {epoch + 1}")
            break

    manager.mark_complete(trial_name)

    return {
        'train_loss': train_losses, 'train_f1': train_f1s,
        'train_acc': train_accs,
        'val_loss': val_losses,   'val_f1': val_f1s,
        'val_acc': val_accs,
        'test_loss': test_losses,  'test_f1': test_f1s,
        'test_acc': test_accs,
        'lr': learning_rates
    }


def train_benchmark_model(model, model_type: str, trial_name: str,
                          manager: SimpleTrialManager,
                          X_train, X_val, X_test,
                          D_train, D_val, D_test,
                          y_train, y_val, y_test,
                          config: Dict, device: torch.device):
    """
    Training loop for Lensformer, LensPINN-Small, ViT, CaiT, and ViTSD.

    Uses standard AdamW + CrossEntropyLoss. EarlyStopping is added for
    consistency with the GraviLens evaluation protocol.

    The _dispatch_forward router handles the model-specific forward pass,
    so ViT/CaiT/ViTSD simply ignore the distortion tensors in the batch.
    """
    n_epochs     = config['n_epochs']
    batch_size   = config['batch_size']
    lr           = config['lr']
    weight_decay = config['weight_decay']
    patience     = config.get('patience', 15)
    save_every   = config.get('save_every', 5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    # Differential LR for LensPINN's pretrained CNN backbones
    if hasattr(model, 'decoder1'):
        backbone_params = (list(model.decoder1.parameters())
                           + list(model.decoder2.parameters()))
        backbone_ids = {id(p) for p in backbone_params}
        new_params   = [p for p in model.parameters()
                        if id(p) not in backbone_ids]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': new_params,      'lr': lr},
        ], weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    stopper   = EarlyStopping(patience=patience, min_delta=1e-4, mode='min')

    train_loader = get_unified_loader(X_train, D_train, y_train, batch_size,
                                      shuffle=True, augment=False)
    val_loader   = get_unified_loader(X_val,   D_val,   y_val,  batch_size,
                                      shuffle=False)
    test_loader  = get_unified_loader(X_test,  D_test,  y_test, batch_size,
                                      shuffle=False)

    train_losses, train_f1s, train_accs = [], [], []
    val_losses,   val_f1s,   val_accs   = [], [], []
    test_losses,  test_f1s,  test_accs  = [], [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        for images, distortions, labels in train_loader:
            images      = images.to(device)
            distortions = distortions.to(device)
            labels      = labels.to(device)

            optimizer.zero_grad()
            outputs = _dispatch_forward(model, model_type, images, distortions)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        train_f1   = f1_score(all_labels, all_preds, average='weighted')
        train_acc  = accuracy_score(all_labels, all_preds)
        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        train_accs.append(train_acc)

        val_metrics  = evaluate_model_unified(
            model, model_type, val_loader, criterion, device)
        test_metrics = evaluate_model_unified(
            model, model_type, test_loader, criterion, device)

        val_losses.append(val_metrics['loss'])
        val_f1s.append(val_metrics['f1'])
        val_accs.append(val_metrics['acc'])
        test_losses.append(test_metrics['loss'])
        test_f1s.append(test_metrics['f1'])
        test_accs.append(test_metrics['acc'])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")

        if (epoch + 1) % save_every == 0:
            manager.save_checkpoint(trial_name, model, optimizer, epoch, {
                'train_loss': train_loss, 'train_f1': train_f1,
                'train_acc': train_acc,
                'val_loss': val_metrics['loss'], 'val_f1': val_metrics['f1'],
                'val_acc': val_metrics['acc']
            })

        if stopper(val_metrics['loss']):
            print(f"\n  Early stopping at epoch {epoch + 1}")
            break

    manager.mark_complete(trial_name)

    return {
        'train_loss': train_losses, 'train_f1': train_f1s,
        'train_acc': train_accs,
        'val_loss': val_losses,   'val_f1': val_f1s,
        'val_acc': val_accs,
        'test_loss': test_losses,  'test_f1': test_f1s,
        'test_acc': test_accs,
    }
