import numpy as np
import torch

from gravilens_sis.data.dataset import GravLensDataset, get_unified_loader
from gravilens_sis.data.distortion import precompute_distortions, preprocess_distortion

rng = np.random.default_rng(0)
X = rng.random((6, 64, 64), dtype=np.float32)
y = np.eye(2, dtype=np.float32)[rng.integers(0, 2, 6)]


def test_preprocess_distortion_shape():
    d = np.asarray(preprocess_distortion(X[0]))
    assert d.shape[-2:] == (64, 64) and np.isfinite(d).all()


def test_dataset_returns_triples():
    D = precompute_distortions(X, verbose=False)
    ds = GravLensDataset(X, D, y)
    assert len(ds) == 6
    item = ds[0]
    assert isinstance(item, (tuple, list)) and len(item) == 3
    for t in item:
        assert torch.isfinite(torch.as_tensor(t).float()).all()


def test_loader_batches():
    D = precompute_distortions(X, verbose=False)
    loader = get_unified_loader(X, D, y, batch_size=2, shuffle=False)
    xb, db, yb = next(iter(loader))
    assert xb.shape[0] == 2 and torch.isfinite(xb).all()
