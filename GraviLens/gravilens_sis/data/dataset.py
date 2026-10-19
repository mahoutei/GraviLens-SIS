# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from ..utils import DEFAULT_SEED

class GravLensDataset(Dataset):
    """
    Unified dataset for all model families.

    Returns (image [1,H,W], distortion [1,H,W], label_int) triples.

    GraviLens/ViT/CaiT/ViTSD ignore the distortion channel;
    Lensformer/LensPINN use it as a second input stream encoding the
    intensity-gradient feature map.
    """
    def __init__(self, X: np.ndarray, D: np.ndarray, y: np.ndarray,
                 augment: bool = False, aug_intensity: float = 0.0):
        self.X = X.astype(np.float32)
        self.D = D.astype(np.float32)
        if y.ndim == 2:
            self.y = np.argmax(y, axis=1).astype(np.int64)
        else:
            self.y = y.astype(np.int64)
        self.augment = augment
        self.aug_intensity = aug_intensity
        self.H, self.W = X.shape[-2], X.shape[-1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img  = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        dist = torch.tensor(self.D[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.y[idx], dtype=torch.long)

        if self.augment and self.aug_intensity > 0:
            if random.random() < 0.5:
                img  = TF.hflip(img)
                dist = TF.hflip(dist)
            if random.random() < 0.5:
                img  = TF.vflip(img)
                dist = TF.vflip(dist)
            max_angle = 15.0 * self.aug_intensity
            angle = random.uniform(-max_angle, max_angle)
            img  = TF.rotate(img,  angle)
            dist = TF.rotate(dist, angle)

        return img, dist, label


def get_unified_loader(X: np.ndarray, D: np.ndarray, y: np.ndarray,
                       batch_size: int, shuffle: bool = True,
                       augment: bool = False, aug_intensity: float = 0.0,
                       seed: int = DEFAULT_SEED) -> DataLoader:
    dataset = GravLensDataset(X, D, y, augment=augment,
                              aug_intensity=aug_intensity)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=g, num_workers=0, pin_memory=True)


def get_progressive_unified_loader(X: np.ndarray, D: np.ndarray, y: np.ndarray,
                                   batch_size: int, current_epoch: int,
                                   total_epochs: int,
                                   aug_start: int = 10,
                                   seed: int = DEFAULT_SEED) -> DataLoader:
    """
    Progressive augmentation loader for GraviLens training.

    [FIX 8] Augmentation is held at zero intensity for the first ``aug_start``
    epochs so the model sees clean gradients during early convergence.
    After that, intensity ramps linearly over 70% of the remaining budget.
    """
    if current_epoch < aug_start:
        intensity = 0.0
    else:
        intensity = min(1.0, (current_epoch - aug_start)
                        / max(1, (total_epochs * 0.7 - aug_start)))
    return get_unified_loader(X, D, y, batch_size, shuffle=True,
                              augment=True, aug_intensity=intensity, seed=seed)


def load_numpy_array_from_path(path: str) -> np.ndarray:
    return np.load(path)
