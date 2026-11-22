"""Disocclusion accounting: what the pinhole camera could not see.

When a defocused *foreground* surface spreads over the image it also
covers pixels whose pinhole ray lands behind it, and -- conversely --
bending rays around the foreground can expose background that was never
recorded in the colour/depth inputs.  Those regions cannot be
reconstructed from the input and must be declared rather than faked:

  unknown mask
      pixels for which no colour/depth evidence exists (occluded by a
      foreground surface within its foreground CoC footprint)
  unrecoverable mask
      subset of the unknown mask that lies further than
      ``declaration_radius_px`` from any visible background sample;
      the value there is a *declared approximation*, not data

The fill itself is nearest-neighbour extrapolation from the closest
visible background -- the most conservative declared approximation,
which never invents structure and makes its own footprint visible.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def disocclusion_mask(
    depth_m: np.ndarray,
    fg_mask: np.ndarray,
    fg_radius_px: np.ndarray,
    edge_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the unknown-background and reachable-background masks.

    Parameters
    ----------
    depth_m:
        Cleaned per-pixel depth.
    fg_mask:
        True where the surface is in front of the focus plane
        (signed CoC > 0).
    fg_radius_px:
        Foreground bokeh radius (zero outside ``fg_mask``).
    edge_mask:
        Detectable polluted boundary pixels -- these are treated as
        known foreground evidence, not unknown background.

    Returns
    -------
    unknown_mask:
        Pixels covered by the dilated foreground footprint for which no
        independent background sample exists.
    reachable_mask:
        Pixels within the declared extrapolation radius of at least one
        visible (non-foreground, non-edge) background sample.
    """
    h, w = depth_m.shape
    fg = np.asarray(fg_mask, dtype=bool)
    edge = np.asarray(edge_mask, dtype=bool)

    if fg.any():
        footprint = np.zeros((h, w), dtype=bool)
        # Process in descending radius buckets so the union is exact
        # without one huge global dilation.
        buckets = np.unique(np.round(fg_radius_px[fg], decimals=1))
        for r in buckets[::-1]:
            sel = fg & (np.round(fg_radius_px, decimals=1) == r)
            d = ndimage.distance_transform_edt(~sel)
            footprint |= d <= float(r)
    else:
        footprint = np.zeros((h, w), dtype=bool)

    visible_bg = ~(fg | edge)
    unknown = footprint & ~fg & ~edge

    # A background sample is reachable at a pixel when the distance to
    # the nearest visible background does not exceed the foreground
    # radius that hides that pixel.
    dist_to_bg, _ = ndimage.distance_transform_edt(
        ~visible_bg, return_indices=True
    )
    limit_field = np.zeros((h, w), dtype=np.float64)
    if fg.any():
        _, (iy, ix) = ndimage.distance_transform_edt(
            ~fg, return_indices=True
        )
        limit_field = fg_radius_px[iy, ix]
    reachable = unknown & (dist_to_bg <= np.maximum(limit_field, 1.0))

    return unknown, reachable


def fill_unknown_background(
    color: np.ndarray,
    depth_m: np.ndarray,
    unknown_mask: np.ndarray,
    reachable_mask: np.ndarray,
    declaration_radius_px: float,
    foreground_mask: np.ndarray,
    edge_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill unknown regions and flag which fill values are recoverable.

    Fill values are extrapolated only from *visible background* samples.
    Pulling depth from the foreground that hides a pixel would assign it
    the occluder's huge foreground CoC and smear it across the frame, so
    foreground (and edge-polluted) pixels are explicitly excluded as
    donors.  Returns ``(filled_color, filled_depth, unrecoverable_mask)``.
    Unreachable pixels receive the same nearest-neighbour fill so the
    compositor stays defined, but are flagged as unrecoverable: the
    viewer is shown the exact range no input evidence covers.
    """
    unknown = np.asarray(unknown_mask, dtype=bool)
    reachable = np.asarray(reachable_mask, dtype=bool)
    donor = ~(unknown | foreground_mask | edge_mask)
    if not donor.any():
        donor = ~unknown  # pathological frame: fall back to any evidence

    _, (iy, ix) = ndimage.distance_transform_edt(
        ~donor, return_indices=True
    )
    filled_color = color.copy()
    filled_depth = depth_m.copy()
    if unknown.any():
        filled_color[unknown] = color[iy[unknown], ix[unknown]]
        filled_depth[unknown] = depth_m[iy[unknown], ix[unknown]]

    dist_to_known = ndimage.distance_transform_edt(unknown)
    unrecoverable = unknown & (
        (~reachable)
        | (dist_to_known > declaration_radius_px)
    )
    return filled_color, filled_depth, unrecoverable
