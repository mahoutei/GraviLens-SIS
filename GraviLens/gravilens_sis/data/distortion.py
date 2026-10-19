# Extracted verbatim from gravilens_finals_poster.py (final v9 Colab export).
import numpy as np

def preprocess_distortion(image: np.ndarray) -> np.ndarray:
    """
    Compute the physics-informed distortion map used by Lensformer/LensPINN.

    Transformation:
        log_img  = [log(I_max / I)]^2
        gradient = d²(log_img) / dx dy   (cross second-order gradient)
        distortion = |tanh(gradient)|
    """
    imax = np.max(image)
    if imax <= 0:
        return np.zeros_like(image, dtype=np.float32)
    log_img  = np.square(np.log(imax / np.clip(image, 1e-12, None)))
    gradient = np.gradient(np.gradient(log_img, axis=0), axis=1)
    return np.abs(np.tanh(gradient)).astype(np.float32)


def precompute_distortions(X: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Batch-compute distortion maps for an entire image array."""
    D = np.zeros_like(X, dtype=np.float32)
    for i in range(len(X)):
        D[i] = preprocess_distortion(X[i])
    if verbose:
        print(f"  Distortion maps: {D.shape}  "
              f"[mean={D.mean():.4f}, std={D.std():.4f}]")
    return D
