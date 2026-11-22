"""Method 1: forward scattering of bokeh disks, near layer to far layer.

Every surface pixel emits the light it received onto the sensor footprint
of its own circle of confusion.  Layers are processed strictly from the
closest depth to the farthest; each blurred layer is *over-composited* on
the accumulated image using its own blurred coverage.

Why this handles occlusion correctly where "blur by per-pixel radius"
fails:

  * A thin foreground element still contributes a full disk at every
    output pixel the disk covers -- its light is not cancelled by the
    background depth stored at that output pixel.
  * Background light reaching a pixel covered by foreground energy is
    attenuated by the foreground coverage (the ``1 - a_fg`` term), which
    is the physically correct partial occlusion of a defocused edge.
  * Disoccluded pixels (see :mod:`inpaint`) start from declared fill
    values and are explicitly flagged; no background is invented silently.
"""

from __future__ import annotations

import numpy as np

from .camera import CameraParams
from .common import DoFMasks, DoFResult, prepare_inputs
from .kernels import convolve_disk, quantize_radii
from .trace import TraceRecorder, TraceRequest


def render_scatter(
    color: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraParams,
    n_layers: int = 64,
    max_radius_px: float | None = None,
    silhouette_m: np.ndarray | None = None,
    declaration_radius_px: float = 8.0,
    coc_limit_px: float = 1.0,
    depth_step_m: float = 0.15,
    edge_half_width_px: int = 1,
    trace: TraceRequest | None = None,
    layer_order: str = "depth",
    layer_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> DoFResult:
    filled_color, filled_depth, masks, declaration = prepare_inputs(
        color,
        depth_m,
        camera,
        silhouette_m=silhouette_m,
        declaration_radius_px=declaration_radius_px,
        coc_limit_px=coc_limit_px,
        depth_step_m=depth_step_m,
        edge_half_width_px=edge_half_width_px,
    )
    h, w = filled_depth.shape
    radius = masks.radius_px
    r_cap = float(radius.max()) if max_radius_px is None else max_radius_px
    r_cap = max(r_cap, 1e-3)

    if layer_override is not None:
        # Diagnostic seam: callers cannot otherwise force a chosen layer
        # partition, and the whole point of the equal-depth order test is
        # to put two co-planar regions in two layers of equal radius.
        layer_idx, layer_radius = layer_override
        n_layers = int(np.asarray(layer_radius).size)
    else:
        layer_idx, layer_radius = quantize_radii(radius, n_layers, r_cap)

    acc_color = np.zeros((h, w, 3), dtype=np.float64)
    acc_alpha = np.zeros((h, w), dtype=np.float64)

    # One blur layer per quantised CoC, convolved as a whole so a thin
    # surface keeps a full, continuous bokeh disk.  Layers are visited
    # from the smallest depth (nearest surface) to the largest; within a
    # layer the disk PSF already averages co-located surfaces correctly.
    layer_min_depth = np.full(n_layers, np.inf)
    for k in range(n_layers):
        sel = layer_idx == k
        if sel.any():
            layer_min_depth[k] = filled_depth[sel].min()

    if layer_order not in ("depth", "reverse"):
        raise ValueError("layer_order must be 'depth' or 'reverse'")
    visit = np.argsort(layer_min_depth)
    if layer_order == "reverse":
        # Diagnostic only: visiting far-to-near in a front-to-back
        # over-composite is physically wrong, so the caller can measure
        # how much the result depends on compositing order alone.
        visit = visit[::-1]

    recorder = None
    if trace is not None:
        recorder = TraceRecorder(
            trace,
            provenance=masks.provenance,
            signed_coc_px=masks.signed_coc_px,
            depth_m=filled_depth,
            layer_index=layer_idx,
            layer_radius=layer_radius,
            layer_min_depth=layer_min_depth,
            method="forward_scatter",
            n_layers=n_layers,
        )

    for order_rank, k in enumerate(visit):
        if not np.isfinite(layer_min_depth[k]):
            continue
        sel = layer_idx == k
        r = layer_radius[k]
        coverage = sel.astype(np.float64)
        premult = np.zeros((h, w, 3), dtype=np.float64)
        premult[sel] = filled_color[sel]

        if recorder is not None:
            recorder.begin_layer(k, r, order_rank, sel)

        if r > 1e-3:
            coverage = convolve_disk(coverage, r)
            for c in range(3):
                premult[..., c] = convolve_disk(premult[..., c], r)
        # Premultiplied-alpha over: the convolved premult already carries
        # colour weighted by disk coverage, so dividing it back out would
        # destroy the energy the lens spreads into neighbouring pixels.
        alpha = np.clip(coverage, 0.0, 1.0)
        weight = 1.0 - acc_alpha
        if recorder is not None:
            recorder.note(
                k, r, order_rank, alpha, weight, acc_alpha,
                premult * weight[..., None],
            )
        acc_color += premult * weight[..., None]
        acc_alpha += alpha * weight

    # Pixels without any coverage (should only happen on an empty frame)
    # fall back to the input colour so output always matches input size.
    hole = acc_alpha <= 1e-8
    if hole.any():
        acc_color[hole] = filled_color[hole]
        acc_alpha[hole] = 1.0

    # Preserve total energy against clipping introduced by compositing.
    out = np.clip(acc_color, 0.0, None)
    declaration["method"] = "forward_scatter"
    declaration["n_layers"] = n_layers
    declaration["max_radius_px"] = r_cap
    layer_trace = None
    if recorder is not None:
        recorder.finalize(out)
        layer_trace = recorder.layer_trace()
    return DoFResult(
        out, "forward_scatter", masks, camera, declaration, trace=layer_trace
    )
