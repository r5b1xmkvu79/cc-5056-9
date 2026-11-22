"""Geometry helpers: edge depth-pollution detection and depth repair.

A rendered depth map is only valid at surfaces that are actually visible.
At silhouette edges the depth of a thin foreground element is mixed with
background samples (or, with hard edge rendering, the boundary pixel
ownership is ambiguous).  Feeding those boundary depths straight into the
CoC formula either hides the foreground (background-like radius) or leaks
foreground blur into the background.  We therefore make the polluted set
*explicit and detectable* instead of silently trusting it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def edge_contamination_mask(
    depth_m: np.ndarray,
    silhouette_m: np.ndarray | None = None,
    depth_step_m: float = 0.05,
    edge_half_width_px: int = 1,
) -> np.ndarray:
    """Return a boolean mask of pixels whose depth is unreliable.

    Two detectors are OR-combined:

    1. Depth discontinuity: neighbours separated by more than
       ``depth_step_m`` -- the classic silhouette between two surfaces.
    2. Declared silhouette: an external mask of known thin structure
       (for example the analytic coverage of a procedural scene), dilated
       by ``edge_half_width_px`` so rasterised boundary pixels are caught.

    The result is dilated by ``edge_half_width_px`` because a hard edge
    pollutes roughly one pixel on each side.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    bad = np.zeros(depth.shape, dtype=bool)

    if depth_step_m > 0:
        # Plain differences (not np.roll): wrap-around at the image
        # border would otherwise mark the whole first row/column.
        bad[1:, :] |= np.abs(depth[1:, :] - depth[:-1, :]) > depth_step_m
        bad[:, 1:] |= np.abs(depth[:, 1:] - depth[:, :-1]) > depth_step_m

    if silhouette_m is not None:
        silhouette = np.asarray(silhouette_m, dtype=bool)
        # A silhouette mask covers the whole thin surface, but only its
        # boundary pixels have polluted depth.  Erode and subtract so the
        # surface interior keeps its valid measured depth.
        interior = ndimage.binary_erosion(silhouette)
        bad |= silhouette & ~interior

    if edge_half_width_px > 0:
        bad = ndimage.binary_dilation(
            bad, iterations=edge_half_width_px
        )
    return bad


def nearest_valid_depth(
    depth_m: np.ndarray, invalid_mask: np.ndarray
) -> np.ndarray:
    """Fill invalid depth pixels with the nearest valid depth.

    Boundary pixels inherit the *foreground* depth when both sides are
    valid; this is the conservative choice for DoF because an
    over-estimated foreground CoC never makes thin structure disappear.
    Where the neighbourhood is entirely invalid we fall back to the
    nearest valid value regardless of ordering.
    """
    depth = np.array(depth_m, dtype=np.float64)
    invalid = np.asarray(invalid_mask, dtype=bool)
    valid = ~invalid
    if valid.all():
        return depth

    filled = depth.copy()
    if valid.any():
        _, (iy, ix) = ndimage.distance_transform_edt(
            ~valid, return_indices=True
        )
        nearest = depth[iy, ix]
        filled[invalid] = nearest[invalid]

    # Prefer the foreground (minimum depth) inside each invalid island.
    labels, n = ndimage.label(invalid)
    if n:
        for label_id in range(1, n + 1):
            island = labels == label_id
            dilated = ndimage.binary_dilation(island, iterations=2) & valid
            if dilated.any():
                filled[island] = depth[dilated].min()
    return filled
