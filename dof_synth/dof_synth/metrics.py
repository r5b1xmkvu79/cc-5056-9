"""Quantitative validation against the optical sampling reference.

Four failure modes are measured explicitly instead of judged by eye:

energy conservation
    Total flux of emissive point lights must survive defocus -- a bokeh
    disk spreads the same energy over more pixels, it must not gain or
    lose it.  Reported as the ratio output/input flux.
halo
    Energy from a bright background point leaking *around* the opaque
    thin foreground (measured on the occluded side of the slab).
penetration
    Background colour visible through a foreground bokeh footprint
    where the foreground coverage should block it.
rmse / structure agreement
    Difference from the Monte-Carlo thin-lens reference, plus agreement
    between the scatter and gather solvers.
"""

from __future__ import annotations

import numpy as np


def total_energy(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        return float(image.sum())
    return float(image[mask].sum())


def energy_ratio(
    sharp: np.ndarray,
    defocused: np.ndarray,
    source_mask: np.ndarray,
    search_radius_px: int,
) -> float:
    """Ratio of bokeh-flux around each point light to its sharp flux."""
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(2, 1)
    region = ndimage.binary_dilation(
        source_mask, structure=structure, iterations=search_radius_px
    )
    sharp_e = total_energy(sharp, region)
    out_e = total_energy(defocused, region)
    if sharp_e <= 0:
        return float("nan")
    return out_e / sharp_e


def halo_energy(
    image: np.ndarray,
    point_support: np.ndarray,
    foreground_silhouette: np.ndarray,
    dilation_px: int,
) -> float:
    """Flux from background point lights landing on the foreground side.

    The measurement region is the slab silhouette dilated by its bokeh
    footprint.  A physically correct lens does deposit *some* light there
    (the disk wraps around the occluder's edges), so this value is
    meaningful as a solver-vs-reference comparison, not as a zero target.
    """
    from scipy import ndimage

    covered = ndimage.binary_dilation(
        foreground_silhouette, iterations=dilation_px
    )
    leaked = covered & point_support
    return total_energy(image, leaked)


def halo_excess(
    defocused: np.ndarray,
    reference: np.ndarray,
    point_support: np.ndarray,
    foreground_silhouette: np.ndarray,
    dilation_px: int,
) -> float:
    """Non-physical halo: flux above what the optical reference allows."""
    from scipy import ndimage

    covered = ndimage.binary_dilation(
        foreground_silhouette, iterations=dilation_px
    )
    region = covered & point_support
    excess = np.maximum(
        defocused[region].sum(axis=1) - reference[region].sum(axis=1), 0.0
    )
    return float(excess.sum())


def penetration_error(
    defocused: np.ndarray,
    reference: np.ndarray,
    foreground_footprint: np.ndarray,
) -> float:
    """Mean absolute deviation from the reference inside the foreground
    bokeh footprint.  Background leaking through an opaque foreground
    shows up as extra energy the reference does not contain; normalised
    by the reference brightness of the same region.
    """
    if not foreground_footprint.any():
        return 0.0
    a = defocused[foreground_footprint].mean(axis=1)
    b = reference[foreground_footprint].mean(axis=1)
    return float(np.abs(a - b).mean() / max(b.mean(), 1e-6))


def rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def per_pixel_error_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b) ** 2, axis=-1))


def naive_radius_blur(
    color: np.ndarray, radius_px: np.ndarray, n_bins: int = 16
) -> np.ndarray:
    """The deliberately naive baseline: one blur radius per pixel,
    ignoring occlusion.  Kept only to demonstrate its failure modes.
    """
    from .kernels import convolve_disk, quantize_radii

    idx, radii = quantize_radii(radius_px, n_bins, float(radius_px.max()) + 1e-3)
    out = np.zeros_like(color)
    weight = np.zeros(color.shape[:2])
    for k, r in enumerate(radii):
        sel = idx == k
        if not sel.any():
            continue
        layer = color * sel[..., None]
        blurred = convolve_disk(layer, r)
        a = convolve_disk(sel.astype(float), r)
        out += blurred
        weight += a
    return out / np.maximum(weight, 1e-6)[..., None]
