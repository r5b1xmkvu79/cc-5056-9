"""Shared DoF pipeline: data structures and input preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraParams
from .geometry import edge_contamination_mask
from .inpaint import disocclusion_mask, fill_unknown_background


@dataclass
class DoFMasks:
    edge_contaminated: np.ndarray       # depth untrustworthy (detectable)
    foreground: np.ndarray               # signed CoC > 0
    in_focus: np.ndarray                 # |CoC| <= limit
    unknown_background: np.ndarray       # no colour/depth evidence
    unrecoverable: np.ndarray            # declared approximation range
    signed_coc_px: np.ndarray
    radius_px: np.ndarray
    # Effective provenance of the sample feeding each pixel's CoC:
    # 0 measured background, 1 measured foreground, 2 edge boundary,
    # 3 unknown-fill (approximation), 4 unrecoverable (approximation).
    provenance: np.ndarray


@dataclass
class DoFResult:
    image: np.ndarray                    # final linear RGB, float in [0, 1]
    method: str
    masks: DoFMasks
    camera: CameraParams
    declaration: dict                    # numeric summary of approximations
    trace: object | None = None          # LayerTrace when tracing was requested


def prepare_inputs(
    color: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraParams,
    silhouette_m: np.ndarray | None = None,
    depth_step_m: float = 0.15,
    edge_half_width_px: int = 1,
    declaration_radius_px: float = 8.0,
    coc_limit_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, DoFMasks, dict]:
    """Validate inputs and produce cleaned colour/depth plus full masks.

    The original ``color`` and ``depth_m`` arrays are never modified;
    every downstream stage works on copies, so recomputation with a new
    focus plane always starts from the raw measurements.
    """
    color = np.asarray(color, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color must be an HxWx3 RGB array")
    if depth_m.shape != color.shape[:2]:
        raise ValueError("depth/color spatial dimensions must match")
    if not np.all(np.isfinite(depth_m)) or np.any(depth_m <= 0):
        raise ValueError("depth must contain finite positive distances (m)")
    h, w = depth_m.shape
    if (w, h) != (camera.image_width_px, camera.image_height_px):
        raise ValueError("camera pixel dimensions must match the inputs")

    edge = edge_contamination_mask(
        depth_m, silhouette_m, depth_step_m, edge_half_width_px
    )
    # Keep the measured depth at boundary pixels: overwriting it with the
    # foreground depth would erase the background samples the lens can
    # still see around the occluder.  Pollution is declared through the
    # edge mask instead; only non-finite values (which cannot occur for
    # valid inputs) would be repaired.
    clean_depth = depth_m.copy()

    signed = camera.signed_coc_px(clean_depth)
    radius = np.abs(signed) * 0.5  # signed_coc is a diameter
    fg = signed > 0.0

    unknown, reachable = disocclusion_mask(
        clean_depth, fg, np.where(fg, radius, 0.0), edge
    )
    filled_color, filled_depth, unrecoverable = fill_unknown_background(
        color, clean_depth, unknown, reachable,
        declaration_radius_px, fg, edge,
    )
    # CoC for the blur estimate uses the repaired field, but surface
    # ownership (foreground/background, edge, unknown) stays tied to the
    # original measurements so a filled pixel never flips identity.
    signed = camera.signed_coc_px(filled_depth)
    radius = np.abs(signed) * 0.5

    # Provenance is tied to the *original* measurement so a filled pixel
    # never masquerades as directly observed data.
    provenance = np.zeros(clean_depth.shape, dtype=np.int8)
    provenance[fg] = 1
    provenance[edge] = 2
    provenance[unknown] = 3
    provenance[unrecoverable] = 4

    masks = DoFMasks(
        edge_contaminated=edge,
        foreground=fg,
        in_focus=np.abs(camera.signed_coc_px(clean_depth)) <= coc_limit_px,
        unknown_background=unknown,
        unrecoverable=unrecoverable,
        signed_coc_px=signed,
        radius_px=radius,
        provenance=provenance,
    )
    declaration = {
        "edge_pixels": int(edge.sum()),
        "unknown_pixels": int(unknown.sum()),
        "unrecoverable_pixels": int(unrecoverable.sum()),
        "unknown_fraction": float(unknown.mean()),
        "unrecoverable_fraction": float(unrecoverable.mean()),
        "declaration_radius_px": float(declaration_radius_px),
        "coc_limit_px": float(coc_limit_px),
    }
    return filled_color, filled_depth, masks, declaration
