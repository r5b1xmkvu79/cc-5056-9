"""Bokeh PSF kernels and (FFT-accelerated) disk convolution.

The physical blur of an ideal thin lens with a circular aperture is a
uniform ("top-hat") disk whose diameter equals the local circle of
confusion.  Kernels are normalised to sum to 1 so that a unit-energy
source deposits exactly one unit of energy after convolution.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

_EPS_RADIUS = 1e-3


def disk_kernel(radius_px: float, intensity: float = 1.0) -> np.ndarray:
    """Normalised top-hat disk kernel for the given radius in pixels."""
    r = max(float(radius_px), _EPS_RADIUS)
    k = int(np.ceil(r))
    yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
    disk = (xx * xx + yy * yy) <= r * r
    kernel = disk.astype(np.float64)
    total = kernel.sum()
    if total == 0.0:
        kernel[k, k] = 1.0
        total = 1.0
    return kernel * (intensity / total)


def convolve_disk(image: np.ndarray, radius_px: float) -> np.ndarray:
    """Convolve an image with the normalised bokeh disk (energy preserving).

    Multi-channel images are convolved channel by channel with the same
    kernel.  A radius below one pixel degrades gracefully toward identity
    via the fractional coverage of the sub-pixel disk.
    """
    if radius_px <= _EPS_RADIUS:
        return image.copy()
    kernel = disk_kernel(radius_px)
    if image.ndim == 2:
        return fftconvolve(image, kernel, mode="same")
    out = np.empty_like(image, dtype=np.float64)
    for c in range(image.shape[2]):
        out[..., c] = fftconvolve(image[..., c], kernel, mode="same")
    return out


def quantize_radii(
    radius_px: np.ndarray, n_layers: int, max_radius_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map continuous per-pixel radii onto n_layers discrete radii.

    Returns (layer_index, layer_radius).  Layer 0 is the sharp layer
    (radius 0).  Every other layer takes the *upper* edge of its bin, so
    a pixel is never blurred less than its physical CoC -- under-blurring
    would erase bokeh energy, whereas a slight over-blur only softens it.
    The final layer is capped at the actual observed maximum.
    """
    r = np.clip(radius_px, 0.0, max_radius_px)
    if n_layers < 2:
        raise ValueError("need at least two layers")
    edges = np.linspace(0.0, max(max_radius_px, 1e-6), n_layers)
    idx = np.clip(np.digitize(r, edges[1:-1], right=False), 0, n_layers - 1)
    radii = edges.copy()
    radii[0] = 0.0
    return idx, radii
