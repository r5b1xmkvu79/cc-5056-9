"""Method 2: reverse per-pixel sampling through quantised depth layers.

For every output pixel the lens ray bundle is simulated by gathering
samples inside the local circle of confusion.  A plain per-pixel disk
blur of the input image would pull background colour through a sharp
foreground edge ("see-through") because the stored colour at sampled
neighbours ignores occlusion.  We instead:

  1. blur each quantised depth layer separately with that layer's CoC,
  2. at every output pixel composite the blurred layers front-to-back,
  3. split the foreground (signed CoC > 0) and background (CoC <= 0)
     contributions: foreground energy occludes the background with its
     blurred coverage, while the background can never overwrite a
     foreground footprint;
  4. weight every gather by the disk PSF so energy is conserved, and
  5. keep the unknown/unrecoverable masks visible so newly exposed
     regions are declared rather than hidden.

This is the "reverse layered sampling" counterpart of
:mod:`scatter`: both share the same optical kernel and masks and must
agree up to layer quantisation, which the test suite checks.
"""

from __future__ import annotations

import numpy as np

from .camera import CameraParams
from .common import DoFResult, prepare_inputs
from .kernels import convolve_disk, quantize_radii
from .trace import TraceRecorder, TraceRequest


def render_gather(
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
        layer_idx, layer_radius = layer_override
        n_layers = int(np.asarray(layer_radius).size)
    else:
        layer_idx, layer_radius = quantize_radii(radius, n_layers, r_cap)

    # Layer depth: representative depth for ordering = minimum member
    # depth, so front-to-back compositing is conservative.
    layer_min_depth = np.full(n_layers, np.inf)
    for k in range(n_layers):
        sel = layer_idx == k
        if sel.any():
            layer_min_depth[k] = filled_depth[sel].min()

    acc_color = np.zeros((h, w, 3), dtype=np.float64)
    acc_alpha = np.zeros((h, w), dtype=np.float64)
    fg_accum_alpha = np.zeros((h, w), dtype=np.float64)

    if layer_order not in ("depth", "reverse"):
        raise ValueError("layer_order must be 'depth' or 'reverse'")
    visit = np.argsort(layer_min_depth)
    if layer_order == "reverse":
        # Diagnostic only: the reverse-gather compositor is also
        # front-to-back, so visiting far-to-near is intentionally wrong
        # and only used to measure order sensitivity.
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
            method="reverse_gather",
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
            coverage_b = np.clip(convolve_disk(coverage, r), 0.0, 1.0)
            premult_b = np.stack(
                [convolve_disk(premult[..., c], r) for c in range(3)],
                axis=-1,
            )
        else:
            coverage_b = coverage
            premult_b = premult

        is_fg_layer = float(camera.signed_coc_mm(layer_min_depth[k])) > 0.0
        if is_fg_layer:
            # Foreground light spreads over background pixels; once its
            # coverage is accumulated it blocks further background light
            # proportionally -- this is the occlusion-side bokeh energy.
            # premult_b already equals colour * disk coverage, so it is
            # composited directly without renormalising energy away.
            weight = 1.0 - fg_accum_alpha
            if recorder is not None:
                recorder.note(
                    k, r, order_rank, coverage_b, weight, fg_accum_alpha,
                    premult_b * weight[..., None],
                )
            fg_accum_alpha += coverage_b * weight
            acc_color += premult_b * weight[..., None]
            acc_alpha += coverage_b * weight
        else:
            # Background layers fill only what foreground coverage and
            # earlier background layers left transparent: no penetration.
            remaining = (1.0 - fg_accum_alpha) * (1.0 - acc_alpha)
            if recorder is not None:
                recorder.note(
                    k, r, order_rank, coverage_b, remaining, acc_alpha,
                    premult_b * remaining[..., None],
                )
            acc_color += premult_b * remaining[..., None]
            acc_alpha += coverage_b * remaining

    hole = acc_alpha <= 1e-8
    if hole.any():
        acc_color[hole] = filled_color[hole]
        acc_alpha[hole] = 1.0

    out = np.clip(acc_color, 0.0, None)
    declaration["method"] = "reverse_gather"
    declaration["n_layers"] = n_layers
    declaration["max_radius_px"] = r_cap
    layer_trace = None
    if recorder is not None:
        recorder.finalize(out)
        layer_trace = recorder.layer_trace()
    return DoFResult(
        out, "reverse_gather", masks, camera, declaration, trace=layer_trace
    )
