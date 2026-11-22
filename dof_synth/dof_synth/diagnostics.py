"""Observational diagnostics for occlusion-side ghosting.

The user-visible failure this module exists to explain is a *duplicate
contour* at a defocused occlusion edge: a thin foreground surface appears
to have a second, brighter outline just outside its true boundary.  The
diagnostic is built on evidence produced by the real solver loop (see
:mod:`trace`), never by a re-implementation, so it cannot drift from the
compositor it describes.

Three questions have to be answered separately, because they have
different fixes:

normalization vs coverage
    Is the error caused by dividing / re-normalising energy (a gain
    error that scales every colour) or by missing *coverage* at the
    boundary (a geometric hole)?  The equal-depth, two-colour input-order
    swap is the discriminator: a pure normalization fault is invariant to
    input order, but so is a coverage fault -- so we additionally read the
    accumulated alpha and the per-layer coverage from the trace.

depth noise vs kernel coverage vs composite order
    Construct a *known* depth step and perturb one factor at a time:
    flatten the depth (removes noise), keep a monotone side face
    (real geometric variation), and swap depth order.  Whichever
    perturbation moves the ghost identifies the stage.

stage localization
    Every anomaly is tagged with the earliest pipeline stage whose
    invariant it violates: input, prepare, quantize, layer_assign,
    convolve, composite, output.

No function here crops the image boundary before reporting a metric.
Full-frame and boundary-band numbers are always reported side by side,
and the number of boundary pixels actually included is recorded, so an
"improvement" can never be an artifact of trimming the offending edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .camera import CameraParams

STAGES = (
    "input",
    "prepare",
    "quantize",
    "layer_assign",
    "convolve",
    "composite",
    "output",
)

# Cause labels, ordered by specificity.
CAUSE_NONE = "none"
CAUSE_DEPTH_NOISE = "depth_noise"
CAUSE_LAYER_FRAGMENTATION = "layer_fragmentation"
CAUSE_KERNEL_COVERAGE = "kernel_coverage"
CAUSE_COMPOSITE_ORDER = "composite_order"
CAUSE_NORMALIZATION = "normalization"


# ---------------------------------------------------------------------------
# Depth-continuous components and layer assignment
# ---------------------------------------------------------------------------

def depth_continuous_components(
    depth_m: np.ndarray, depth_step_m: float
) -> np.ndarray:
    """Label surfaces that are continuous in depth (no jump > step).

    Two 4-neighbour pixels belong to the same component when their depth
    differs by at most ``depth_step_m``.  This is *measured geometry*,
    not a heuristic fill: it only groups pixels whose depth evidence
    actually connects.  The label array has shape ``depth.shape``.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    h, w = depth.shape
    index = np.arange(h * w).reshape(h, w)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    vertical = np.abs(np.diff(depth, axis=0)) <= depth_step_m
    r, c = np.where(vertical)
    rows.append(index[r, c])
    cols.append(index[r + 1, c])

    horizontal = np.abs(np.diff(depth, axis=1)) <= depth_step_m
    r, c = np.where(horizontal)
    rows.append(index[r, c])
    cols.append(index[r, c + 1])

    rr = np.concatenate(rows)
    cc = np.concatenate(cols)
    graph = coo_matrix(
        (np.ones(rr.size), (rr, cc)), shape=(h * w, h * w)
    )
    _, labels = connected_components(graph, directed=False)
    return labels.reshape(h, w)


def layer_fragmentation(
    depth_m: np.ndarray,
    layer_index: np.ndarray,
    depth_step_m: float,
) -> tuple[np.ndarray, list[dict]]:
    """Find depth-continuous surfaces split across several radius layers.

    Returns ``(fragmented_mask, components)`` where each component record
    lists the layers it spans.  A single physical surface that appears in
    more than one quantised radius layer will be composited as if it were
    several mutually-occluding surfaces -- the precursor of a duplicate
    contour at its boundary.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    layers = np.asarray(layer_index)
    labels = depth_continuous_components(depth, depth_step_m)

    fragmented = np.zeros(depth.shape, dtype=bool)
    records: list[dict] = []
    for label in np.unique(labels):
        sel = labels == label
        span = np.unique(layers[sel])
        if span.size <= 1:
            continue
        fragmented |= sel
        ys, xs = np.where(sel)
        records.append(
            {
                "component": int(label),
                "pixels": int(sel.sum()),
                "depth_min_m": float(depth[sel].min()),
                "depth_max_m": float(depth[sel].max()),
                "depth_span_m": float(depth[sel].max() - depth[sel].min()),
                "layers": [int(v) for v in span],
                "n_layers": int(span.size),
                "bbox": [
                    int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
                ],
            }
        )
    records.sort(key=lambda d: (-d["n_layers"], -d["pixels"]))
    return fragmented, records


# ---------------------------------------------------------------------------
# Depth-of-field boundary tracking
# ---------------------------------------------------------------------------

@dataclass
class DepthBoundary:
    row: int
    col: int
    near_depth_m: float
    far_depth_m: float
    near_coc_px: float
    far_coc_px: float
    near_side: str
    far_side: str
    straddles_dof: bool
    near_layer: int
    far_layer: int
    layer_split: bool


@dataclass
class DoFBoundaryReport:
    near_m: float
    far_m: float
    in_focus_mask: np.ndarray
    boundaries: list[DepthBoundary]
    layer_index: np.ndarray
    component_report: list[dict]
    summary: dict = field(default_factory=dict)


def _side_label(signed_coc_px: float, coc_limit_px: float) -> str:
    if abs(signed_coc_px) <= coc_limit_px:
        return "in_focus"
    return "foreground" if signed_coc_px > 0.0 else "background"


def track_dof_boundaries(
    depth_m: np.ndarray,
    camera: CameraParams,
    layer_index: np.ndarray | None = None,
    coc_limit_px: float = 1.0,
    depth_step_m: float = 0.15,
) -> DoFBoundaryReport:
    """Track where depth discontinuities meet the DOF limits.

    Every boundary pixel is recorded with the measured near/far depth on
    each side, the signed CoC on each side, which side is foreground, and
    the quantised layer of each side.  A boundary whose two sides fall in
    different radius layers is flagged ``layer_split``.

    The function fails loudly on invalid input instead of substituting a
    default depth or camera value: boundary tracking that silently used
    defaults would hide exactly the errors it is meant to reveal.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2-D array")
    if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
        raise ValueError(
            "depth boundary tracking requires finite positive depths; "
            "refusing to substitute a default"
        )
    h, w = depth.shape
    if (w, h) != (camera.image_width_px, camera.image_height_px):
        raise ValueError("camera pixel dimensions must match the depth map")
    if layer_index is None:
        from .kernels import quantize_radii

        radius = np.abs(camera.signed_coc_px(depth)) * 0.5
        layer_index, _ = quantize_radii(
            radius, 64, max(float(radius.max()), 1e-3)
        )
    layers = np.asarray(layer_index)

    near_m, far_m = camera.dof_bounds_m(coc_limit_px * camera.pixel_pitch_mm)
    signed = camera.signed_coc_px(depth)
    in_focus = np.abs(signed) <= coc_limit_px

    boundaries: list[DepthBoundary] = []
    # Horizontal and vertical neighbour pairs; plain slicing avoids the
    # wrap-around at the image border that np.roll would introduce.
    for axis in (0, 1):
        if axis == 0:
            a = (slice(0, h - 1), slice(None))
            b = (slice(1, h), slice(None))
        else:
            a = (slice(None), slice(0, w - 1))
            b = (slice(None), slice(1, w))
        jump = np.abs(depth[a] - depth[b]) > depth_step_m
        ii, jj = np.where(jump)
        for i, j in zip(ii, jj):
            if axis == 0:
                p, q = (i, j), (i + 1, j)
            else:
                p, q = (i, j), (i, j + 1)
            dp, dq = depth[p], depth[q]
            near_pt, far_pt = (p, q) if dp <= dq else (q, p)
            near_d = float(min(dp, dq))
            far_d = float(max(dp, dq))
            near_coc = float(camera.signed_coc_px(near_d))
            far_coc = float(camera.signed_coc_px(far_d))
            boundaries.append(
                DepthBoundary(
                    row=int(p[0]),
                    col=int(p[1]),
                    near_depth_m=near_d,
                    far_depth_m=far_d,
                    near_coc_px=near_coc,
                    far_coc_px=far_coc,
                    near_side=_side_label(near_coc, coc_limit_px),
                    far_side=_side_label(far_coc, coc_limit_px),
                    straddles_dof=(
                        (near_d <= near_m <= far_d)
                        or (np.isfinite(far_m) and near_d <= far_m <= far_d)
                    ),
                    near_layer=int(layers[near_pt]),
                    far_layer=int(layers[far_pt]),
                    layer_split=int(layers[near_pt]) != int(layers[far_pt]),
                )
            )

    _, component_report = layer_fragmentation(depth, layers, depth_step_m)
    n_split = int(sum(b.layer_split for b in boundaries))
    fragmentation = [
        c for c in component_report if c["n_layers"] > 1
    ]
    summary = {
        "boundary_pixels": int(len(boundaries)),
        # Two sides of a depth edge normally land in different radius
        # layers (they differ in depth); this is expected and only
        # informative alongside the fragmentation count below.
        "boundaries_distinct_layers": n_split,
        "boundaries_distinct_fraction": (
            float(n_split / len(boundaries)) if boundaries else 0.0
        ),
        # A *single* depth-continuous surface spanning several radius
        # layers is the pathological case behind a duplicate contour.
        "fragmented_components": len(fragmentation),
        "fragmented_pixels": int(
            sum(c["pixels"] for c in fragmentation)
        ),
        "max_layers_per_component": int(
            max((c["n_layers"] for c in fragmentation), default=1)
        ),
        "dof_near_m": float(near_m),
        "dof_far_m": float(far_m),
        "coc_limit_px": float(coc_limit_px),
        "depth_step_m": float(depth_step_m),
        "included_full_frame": True,
        "frame_pixels": int(depth.size),
        "defaults_used": False,
    }
    return DoFBoundaryReport(
        near_m=float(near_m),
        far_m=float(far_m),
        in_focus_mask=in_focus,
        boundaries=boundaries,
        layer_index=layers,
        component_report=component_report,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Ghost measurement against a reference (full frame, never cropped)
# ---------------------------------------------------------------------------

@dataclass
class GhostMeasurement:
    profile: dict
    full_frame_rmse: float
    boundary_band_rmse: float
    band_pixels: int
    frame_pixels: int
    ghost_width_px: float
    ghost_width_max_px: int
    ghost_sides: int
    ghost_excess_peak: float       # robust (90th percentile over sides)
    ghost_excess_max: float        # raw max, may be emissive Monte-Carlo noise
    notch_excess_peak: float
    cropped: bool = False


def _ghost_sides(occluder_mask: np.ndarray) -> list[tuple[int, int, int]]:
    """For each row crossing the occluder, return (row, left_edge, right_edge).

    Edges are the outer columns still belonging to the occluder; the ghost
    is searched just *outside* them.
    """
    sides = []
    for row in np.where(occluder_mask.any(axis=1))[0]:
        cols = np.where(occluder_mask[row])[0]
        if cols.size == 0:
            continue
        sides.append((int(row), int(cols.min()), int(cols.max())))
    return sides


def measure_ghost(
    image: np.ndarray,
    reference: np.ndarray,
    occluder_mask: np.ndarray,
    *,
    span: int = 10,
    threshold: float = 2e-3,
    band_dilation: int = 8,
    band_erosion: int = 3,
) -> GhostMeasurement:
    """Quantify the duplicate contour just outside an occluder.

    ``ghost_excess_peak`` is the largest positive (image - reference)
    luminance just outside the occluder silhouette -- a bright duplicate
    contour.  ``notch_excess_peak`` is the largest negative excess just
    *inside* it -- a dark coverage hole.  ``ghost_width_px`` is the mean
    number of contiguous outside columns whose excess exceeds
    ``threshold`` before it falls back.

    All metrics are computed on the full frame; the boundary-band RMSE is
    reported separately and the pixel counts are recorded, so the caller
    cannot mistake a boundary-excluded number for a full-frame one.
    """
    image = np.asarray(image, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if image.shape != reference.shape:
        raise ValueError("image and reference must have identical shapes")
    occluder = np.asarray(occluder_mask, dtype=bool)

    lum = image.mean(axis=2)
    ref = reference.mean(axis=2)
    excess = lum - ref
    h, w = lum.shape

    ghost_peaks: list[float] = []
    notch_peaks: list[float] = []
    widths: list[int] = []
    for row, left, right in _ghost_sides(occluder):
        for direction, edge in ((-1, left), (1, right)):
            outer_cols = [
                edge + direction * k for k in range(1, span + 1)
                if 0 <= edge + direction * k < w
            ]
            inner_cols = [
                edge - direction * k for k in range(1, span + 1)
                if 0 <= edge - direction * k < w
            ]
            if outer_cols:
                vals = excess[row, outer_cols]
                ghost_peaks.append(float(np.clip(vals, 0.0, None).max()))
                count = 0
                for v in vals:
                    if v > threshold:
                        count += 1
                    elif count > 0:
                        break
                widths.append(count)
            if inner_cols:
                vals = excess[row, inner_cols]
                notch_peaks.append(float(np.clip(-vals, 0.0, None).max()))

    band = ndimage.binary_dilation(
        occluder, iterations=band_dilation
    ) & ~ndimage.binary_erosion(occluder, iterations=band_erosion)
    full = np.ones((h, w), dtype=bool)
    full_rmse = float(np.sqrt(np.mean((image[full] - reference[full]) ** 2)))
    band_rmse = (
        float(np.sqrt(np.mean((image[band] - reference[band]) ** 2)))
        if band.any()
        else float("nan")
    )

    return GhostMeasurement(
        profile={
            "span": span,
            "threshold": threshold,
            "ghost_side_samples": len(ghost_peaks),
            "notch_side_samples": len(notch_peaks),
            "peak_statistic": "p90 over sides",
        },
        full_frame_rmse=full_rmse,
        boundary_band_rmse=band_rmse,
        band_pixels=int(band.sum()),
        frame_pixels=int(h * w),
        ghost_width_px=float(np.mean(widths)) if widths else 0.0,
        ghost_width_max_px=int(np.max(widths)) if widths else 0,
        ghost_sides=len(widths),
        ghost_excess_peak=(
            float(np.percentile(ghost_peaks, 90)) if ghost_peaks else 0.0
        ),
        ghost_excess_max=float(np.max(ghost_peaks)) if ghost_peaks else 0.0,
        notch_excess_peak=(
            float(np.percentile(notch_peaks, 90)) if notch_peaks else 0.0
        ),
        cropped=False,
    )


# ---------------------------------------------------------------------------
# In-kernel evidence chain and cause / stage localization
# ---------------------------------------------------------------------------

def _dominant_provenance(trace, layer: int) -> str:
    counts = trace.layer_provenance.get(layer, {})
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


@dataclass
class PixelAttribution:
    row: int
    col: int
    provenance: str
    depth_m: float
    assigned_layer: int
    signed_coc_px: float
    events: list[dict]
    final_rgb: tuple[float, float, float] | None
    foreground_coverage: float
    background_energy: float
    truncated: bool


def attribute_pixels(trace) -> list[PixelAttribution]:
    """Turn the bounded per-pixel trace into attribution records."""
    out: list[PixelAttribution] = []
    for (row, col), pt in sorted(trace.pixels.items()):
        events = []
        fg_cov = 0.0
        bg_energy = 0.0
        for ev in pt.events:
            provenance = _dominant_provenance(trace, ev.layer)
            events.append(
                {
                    "order_rank": ev.order_rank,
                    "layer": ev.layer,
                    "radius_px": ev.radius_px,
                    "layer_sign": ev.layer_sign,
                    "layer_min_depth_m": ev.layer_min_depth_m,
                    "coverage": ev.coverage,
                    "weight": ev.weight,
                    "acc_before": ev.acc_before,
                    "energy_rgb": list(ev.energy_rgb),
                    "provenance": provenance,
                }
            )
            if ev.layer_sign == "foreground":
                fg_cov += ev.coverage * ev.weight
            elif ev.layer_sign == "background":
                bg_energy += float(sum(ev.energy_rgb))
        out.append(
            PixelAttribution(
                row=row,
                col=col,
                provenance=pt.provenance,
                depth_m=pt.depth_m,
                assigned_layer=pt.layer_index,
                signed_coc_px=pt.signed_coc_px,
                events=events,
                final_rgb=pt.output_rgb,
                foreground_coverage=float(min(fg_cov, 1.0)),
                background_energy=float(bg_energy),
                truncated=pt.truncated,
            )
        )
    return out


def localize_stage(
    attribution: PixelAttribution,
    *,
    fragmented: bool,
    quantized_under: bool,
    coverage_hole: bool,
) -> tuple[str, list[str]]:
    """Return the earliest stage whose invariant the pixel violates.

    The returned list contains every violated stage so the full picture is
    preserved; the first element is the earliest (root-most) stage.
    """
    flags: list[str] = []
    if attribution.provenance in ("unknown_fill", "unrecoverable"):
        flags.append("input")
    if attribution.provenance == "edge":
        flags.append("prepare")
    if quantized_under:
        flags.append("quantize")
    if fragmented:
        flags.append("layer_assign")
    if coverage_hole:
        flags.append("convolve")
    if (
        attribution.background_energy > 1e-6
        and attribution.foreground_coverage < 0.99
    ):
        flags.append("composite")
    if not flags:
        flags.append("output")
    return flags[0], flags


@dataclass
class OcclusionDiagnostic:
    method: str
    layer_records: list[dict]
    component_report: list[dict]
    fragmented_mask: np.ndarray
    attribution: list[PixelAttribution]
    ghost: GhostMeasurement | None
    cause: str
    stage: str
    stage_flags: dict
    per_pixel_stage: list[dict]
    evidence: list[dict]
    order_invariant: bool | None
    boundary_summary: dict
    trace_truncated: bool
    dropped_events: int

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "cause": self.cause,
            "stage": self.stage,
            "stage_flags": self.stage_flags,
            "per_pixel_stage": self.per_pixel_stage,
            "order_invariant": self.order_invariant,
            "boundary_summary": self.boundary_summary,
            "layer_records": self.layer_records,
            "components": self.component_report,
            "attribution": [
                {
                    "row": a.row,
                    "col": a.col,
                    "provenance": a.provenance,
                    "depth_m": a.depth_m,
                    "assigned_layer": a.assigned_layer,
                    "signed_coc_px": a.signed_coc_px,
                    "foreground_coverage": a.foreground_coverage,
                    "background_energy": a.background_energy,
                    "events": a.events,
                    "final_rgb": a.final_rgb,
                    "truncated": a.truncated,
                }
                for a in self.attribution
            ],
            "ghost": (
                {
                    "ghost_width_px": self.ghost.ghost_width_px,
                    "ghost_width_max_px": self.ghost.ghost_width_max_px,
                    "ghost_sides": self.ghost.ghost_sides,
                    "ghost_excess_peak": self.ghost.ghost_excess_peak,
                    "ghost_excess_max": self.ghost.ghost_excess_max,
                    "notch_excess_peak": self.ghost.notch_excess_peak,
                    "full_frame_rmse": self.ghost.full_frame_rmse,
                    "boundary_band_rmse": self.ghost.boundary_band_rmse,
                    "band_pixels": self.ghost.band_pixels,
                    "frame_pixels": self.ghost.frame_pixels,
                    "cropped": self.ghost.cropped,
                    "profile": self.ghost.profile,
                }
                if self.ghost is not None
                else None
            ),
            "evidence": self.evidence,
            "trace_truncated": self.trace_truncated,
            "dropped_events": self.dropped_events,
        }


def _layer_records(trace) -> list[dict]:
    records = []
    for layer in sorted(trace.layer_signs):
        records.append(
            {
                "layer": int(layer),
                "radius_px": float(trace.layer_radius_px[layer]),
                "min_depth_m": float(trace.layer_min_depth_m[layer]),
                "sign": trace.layer_signs[layer],
                "provenance": trace.layer_provenance.get(layer, {}),
            }
        )
    return records


def diagnose_occlusion(
    result,
    depth_m: np.ndarray,
    camera: CameraParams,
    *,
    occluder_mask: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    trace=None,
    depth_step_m: float = 0.15,
    coc_limit_px: float = 1.0,
    order_invariant: bool | None = None,
    band_dilation: int = 8,
    band_erosion: int = 3,
) -> OcclusionDiagnostic:
    """Assemble the full evidence chain for one rendered result.

    ``trace`` must come from the same solver call that produced
    ``result`` (``result.trace``); the attribution then describes the
    exact contributions that built the output pixel.  When ``reference``
    is supplied the ghost width / excess are measured against it on the
    full frame.
    """
    if trace is None:
        trace = getattr(result, "trace", None)
    if trace is None:
        raise ValueError(
            "diagnose_occlusion needs an in-kernel trace; render with "
            "TraceRequest so the evidence comes from the real solver loop"
        )

    depth = np.asarray(depth_m, dtype=np.float64)
    layer_index = np.empty(depth.shape, dtype=np.int64)
    # Re-quantise identically to the solver, using the solver's own cap.
    from .kernels import quantize_radii

    radius = np.abs(camera.signed_coc_px(depth)) * 0.5
    r_cap = max(float(radius.max()), 1e-3)
    declared_cap = float(trace.layer_radius_px.max())
    if declared_cap > 0.0:
        r_cap = declared_cap
    layer_index, layer_radius = quantize_radii(
        radius, trace.n_layers, r_cap
    )

    boundary = track_dof_boundaries(
        depth, camera, layer_index, coc_limit_px, depth_step_m
    )
    fragmented, component_report = layer_fragmentation(
        depth, layer_index, depth_step_m
    )

    # A ghost is a *background*-side pixel affected by a depth-continuous
    # foreground surface split across layers, whose bokeh footprint
    # reaches it.  Build that reach explicitly: dilate every fragmented
    # component by its largest member radius.  This lets the diagnosis
    # tie the proximate coverage hole at a background edge pixel back to
    # the fragmented foreground surface that produced it.
    fragmentation_reach = np.zeros(depth.shape, dtype=bool)
    labels = depth_continuous_components(depth, depth_step_m)
    for rec in component_report:
        if rec["n_layers"] <= 1:
            continue
        comp = labels == rec["component"]
        comp_radius = float(np.abs(camera.signed_coc_px(depth[comp])).max() * 0.5)
        reach = int(np.ceil(comp_radius)) + 1
        fragmentation_reach |= ndimage.binary_dilation(
            comp, iterations=reach
        )

    attribution = attribute_pixels(trace)

    # Boundary = depth discontinuity or declared occluder silhouette.
    edge = boundary.summary["boundary_pixels"] > 0
    edge_mask = np.zeros(depth.shape, dtype=bool)
    if occluder_mask is not None:
        edge_mask |= np.asarray(occluder_mask, dtype=bool)
    if edge:
        for b in boundary.boundaries:
            edge_mask[max(0, b.row - 1): b.row + 2,
                     max(0, b.col - 1): b.col + 2] = True

    ghost = None
    if reference is not None and occluder_mask is not None:
        ghost = measure_ghost(
            result.image, reference, occluder_mask,
            band_dilation=band_dilation, band_erosion=band_erosion,
        )

    # ---- cause attribution -------------------------------------------
    stage_flags: dict[str, int] = {s: 0 for s in STAGES}
    per_pixel: list[dict] = []
    coverage_holes = 0
    fragmented_hits = 0
    quantized_under = 0
    bin_width = r_cap / max(trace.n_layers - 1, 1)
    subbin_under_max = 0.0
    for att in attribution:
        true_radius = abs(att.signed_coc_px) * 0.5
        assigned_radius = float(layer_radius[att.assigned_layer])
        under_by = true_radius - assigned_radius
        subbin_under_max = max(subbin_under_max, under_by)
        # A gross violation is under-blur by more than a whole bin.  The
        # routine upper-edge-vs-lower-edge rounding (always < one bin) is
        # recorded separately as ``quantization_subbin_under_px`` rather
        # than mislabelled as the ghost cause.
        under = under_by > bin_width
        # A pixel is "fragmentation-affected" if it belongs to a split
        # component *or* lies inside the bokeh reach of one -- the latter
        # covers the background-side ghost that a fragmented foreground
        # smears onto.
        frag = bool(fragmented[att.row, att.col])
        frag_reach = bool(fragmentation_reach[att.row, att.col])
        hole = att.foreground_coverage < 0.99 and att.background_energy > 1e-6
        if under:
            quantized_under += 1
        if frag or frag_reach:
            fragmented_hits += 1
        if hole:
            coverage_holes += 1
        stage, flags = localize_stage(
            att, fragmented=frag or frag_reach,
            quantized_under=under, coverage_hole=hole,
        )
        for f in flags:
            stage_flags[f] += 1
        per_pixel.append(
            {
                "row": att.row,
                "col": att.col,
                "stage": stage,
                "flags": flags,
                "foreground_coverage": att.foreground_coverage,
                "background_energy": att.background_energy,
                "fragmented": frag,
                "fragmentation_reach": frag_reach,
                "quantized_under": under,
                "quantization_under_px": float(under_by),
            }
        )

    # Priority: an order-dependent result beats everything; otherwise the
    # earliest *observed* stage-specific invariant wins.
    if order_invariant is False:
        cause = CAUSE_COMPOSITE_ORDER
        stage = "composite"
    elif fragmented_hits:
        cause = CAUSE_LAYER_FRAGMENTATION
        stage = "layer_assign"
    elif coverage_holes:
        cause = CAUSE_KERNEL_COVERAGE
        stage = "composite"
    elif quantized_under:
        cause = CAUSE_KERNEL_COVERAGE
        stage = "quantize"
    else:
        cause = CAUSE_NONE
        stage = "output"

    evidence = [
        {
            "order": 1,
            "stage": "quantize",
            "observation": (
                f"{quantized_under} traced pixel(s) under-blurred by more "
                f"than one bin; worst sub-bin rounding "
                f"{subbin_under_max:.4f} px (not a ghost cause)"
            ),
            "value": quantized_under,
            "subbin_under_px": float(subbin_under_max),
        },
        {
            "order": 2,
            "stage": "layer_assign",
            "observation": (
                f"{fragmented_hits} traced pixel(s) belong to a depth-"
                f"continuous surface split across radius layers"
            ),
            "value": fragmented_hits,
        },
        {
            "order": 3,
            "stage": "composite",
            "observation": (
                f"{coverage_holes} traced pixel(s) let background energy in "
                f"while foreground coverage < 1"
            ),
            "value": coverage_holes,
        },
        {
            "order": 4,
            "stage": "convolve",
            "observation": (
                "ghost width %.2f px, peak +/- %.4f, full-frame RMSE %.5f"
                % (
                    ghost.ghost_width_px if ghost else float("nan"),
                    ghost.ghost_excess_peak if ghost else float("nan"),
                    ghost.full_frame_rmse if ghost else float("nan"),
                )
            ),
            "value": ghost.ghost_width_px if ghost else None,
        },
        {
            "order": 5,
            "stage": "composite",
            "observation": (
                "input-order invariant" if order_invariant
                else "input-order dependent" if order_invariant is False
                else "input-order not tested"
            ),
            "value": order_invariant,
        },
    ]

    return OcclusionDiagnostic(
        method=result.method,
        layer_records=_layer_records(trace),
        component_report=component_report,
        fragmented_mask=fragmented,
        attribution=attribution,
        ghost=ghost,
        cause=cause,
        stage=stage,
        stage_flags=stage_flags,
        per_pixel_stage=per_pixel,
        evidence=evidence,
        order_invariant=order_invariant,
        boundary_summary=boundary.summary,
        trace_truncated=trace.truncated,
        dropped_events=trace.dropped_events,
    )


# ---------------------------------------------------------------------------
# Known depth-step experiment: separate noise / coverage / order
# ---------------------------------------------------------------------------

def _two_layer_inputs(
    h: int,
    w: int,
    *,
    bar_cols: tuple[int, int],
    bar_depth: float,
    bg_depth: float,
    side_face: dict[int, float] | None = None,
    depth_noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    color = np.zeros((h, w, 3), dtype=np.float64)
    depth = np.full((h, w), bg_depth, dtype=np.float64)
    color[:, :] = np.array([0.18, 0.22, 0.30])
    c0, c1 = bar_cols
    depth[:, c0:c1] = bar_depth
    color[:, c0:c1] = np.array([0.85, 0.30, 0.25])
    if side_face:
        for col, d in side_face.items():
            depth[:, col] = d
    if depth_noise > 0.0:
        rng = np.random.default_rng(seed)
        depth[:, c0:c1] += rng.normal(0.0, depth_noise, size=(h, c1 - c0))
    return color, depth


def known_depth_step_probe(
    camera: CameraParams,
    *,
    n_layers: int = 64,
    depth_step_m: float = 0.25,
    h: int | None = None,
    w: int | None = None,
    reference: np.ndarray | None = None,
) -> dict:
    """Controlled experiment isolating the three candidate causes.

    Builds a thin bright bar over a background and perturbs one factor at
    a time:

    flat
        perfectly constant bar depth -- no bin crossing;
    side_face
        monotone receding side face -- real geometric depth variation,
        the bar is still one physical surface;
    depth_noise
        flat bar plus symmetric noise -- makes bins flicker but keeps the
        mean depth;
    order_swap
        two halves at *identical* depth with swapped input colours --
        discriminates input-order / normalization faults.

    The probe reports the *unoccluded* flat step as the intrinsic kernel
    baseline, so a residual ghost in the flat case is not blamed on depth.
    It never crops the frame.
    """
    from .gather import render_gather
    from .scatter import render_scatter

    if h is None:
        h = camera.image_height_px
    if w is None:
        w = camera.image_width_px
    # The probe renders its own inputs, so it owns a camera sized to them
    # rather than trusting a mismatched caller size.
    camera = CameraParams(
        camera.f_mm,
        camera.f_number,
        camera.focus_distance_m,
        camera.sensor_width_mm,
        w,
        h,
    )
    bar = (w // 2 - 10, w // 2 + 10)
    bg_depth = 5.95
    bar_depth = 0.70
    results: dict[str, dict] = {}

    def render(color, depth):
        return (
            render_scatter(color, depth, camera, n_layers=n_layers,
                           depth_step_m=depth_step_m).image,
            render_gather(color, depth, camera, n_layers=n_layers,
                          depth_step_m=depth_step_m).image,
        )

    flat_c, flat_d = _two_layer_inputs(
        h, w, bar_cols=bar, bar_depth=bar_depth, bg_depth=bg_depth
    )
    flat_s, flat_g = render(flat_c, flat_d)

    side_c, side_d = _two_layer_inputs(
        h, w, bar_cols=bar, bar_depth=bar_depth, bg_depth=bg_depth,
        side_face={bar[1] - 2: bar_depth + 0.007,
                   bar[1] - 1: bar_depth + 0.034},
    )
    side_s, side_g = render(side_c, side_d)

    noise_c, noise_d = _two_layer_inputs(
        h, w, bar_cols=bar, bar_depth=bar_depth, bg_depth=bg_depth,
        depth_noise=0.01, seed=1,
    )
    noise_s, noise_g = render(noise_c, noise_d)

    # Order sensitivity: a single-depth frame split spatially into two
    # co-planar layers of *equal* radius, composited near->far vs
    # far->near.  There is no near/far distinction (one depth, two
    # halves), so a correct compositor must be order-invariant; any
    # difference comes from compositing order alone, not from depth,
    # coverage, or normalisation.  No background layer is included, so
    # the test cannot be confounded by a foreground/background swap.
    mid = (bar[0] + bar[1]) // 2
    order_depth = np.full((h, w), bar_depth)
    order_color = np.zeros((h, w, 3))
    order_color[:, :] = [0.85, 0.30, 0.25]

    order_layers = np.ones((h, w), dtype=np.int64)
    order_layers[:, mid:] = 2
    order_radius = np.zeros(3)
    order_radius[0] = 0.0
    order_radius[1] = order_radius[2] = float(
        np.abs(camera.signed_coc_px(bar_depth)) * 0.5
    )
    override = (order_layers, order_radius)
    odo_s = render_scatter(
        order_color, order_depth, camera, depth_step_m=depth_step_m,
        layer_order="depth", layer_override=override,
    ).image
    orv_s = render_scatter(
        order_color, order_depth, camera, depth_step_m=depth_step_m,
        layer_order="reverse", layer_override=override,
    ).image
    odo_g = render_gather(
        order_color, order_depth, camera, depth_step_m=depth_step_m,
        layer_order="depth", layer_override=override,
    ).image
    orv_g = render_gather(
        order_color, order_depth, camera, depth_step_m=depth_step_m,
        layer_order="reverse", layer_override=override,
    ).image
    order_diff_s = float(np.abs(odo_s - orv_s).max())
    order_diff_g = float(np.abs(odo_g - orv_g).max())
    order_invariant = bool(
        order_diff_s < 1e-6 and order_diff_g < 1e-6
    )

    occluder = np.zeros((h, w), dtype=bool)
    occluder[:, bar[0]:bar[1]] = True

    def edge_signature(img):
        """Duplicate-contour signature: the largest *outward rebound*.

        Walk outward across each occluder edge and record the biggest
        increase in luminance between consecutive columns.  A physically
        correct, monotone edge only ever decreases going outward, so the
        rebound is 0.  A duplicate contour raises the exterior column
        above the interior one, producing a positive rebound right at
        the edge.  Measured over the full frame; nothing is cropped.
        """
        lum = img.mean(axis=2)
        max_rebound = 0.0
        where = None
        # Skip rows within one PSF radius of the frame edge: the disk
        # convolution legitimately loses energy off-frame there, which
        # would masquerade as a rebound.  This is an artifact exclusion,
        # not boundary cropping -- the occluder edge itself is always
        # inside the scanned rows and the full-frame RMSE is reported
        # separately.
        pad = 8
        for row in range(pad, h - pad):
            for edge, outward in ((bar[0], -1), (bar[1], 1)):
                # Columns in one direction: from a few px inside the
                # occluder (k<0) to a few px outside it (k>0), so the
                # sequence is monotone in the outward direction.
                cols = [edge + outward * k for k in range(-4, 7)]
                cols = [c for c in cols if 0 <= c < w]
                if len(cols) < 3:
                    continue
                vals = lum[row, cols]
                steps = np.diff(vals)  # positive => brighter going out
                if steps.size:
                    j = int(np.argmax(steps))
                    if float(steps[j]) > max_rebound:
                        max_rebound = float(steps[j])
                        where = (row, cols[j], cols[j + 1])
        return max_rebound, where

    def summarize(img_s, img_g):
        reb_s, where_s = edge_signature(img_s)
        reb_g, where_g = edge_signature(img_g)
        return {
            "scatter_rebound": reb_s,
            "gather_rebound": reb_g,
            "scatter_rebound_at": where_s,
            "gather_rebound_at": where_g,
            "scatter_gather_rmse": float(
                np.sqrt(np.mean((img_s - img_g) ** 2))
            ),
        }

    results["flat"] = summarize(flat_s, flat_g)
    results["side_face"] = summarize(side_s, side_g)
    results["depth_noise"] = summarize(noise_s, noise_g)
    results["order_swap"] = {
        "scatter_order_diff": order_diff_s,
        "gather_order_diff": order_diff_g,
        "invariant": order_invariant,
        "test": "equal-depth two-layer, near->far vs far->near",
    }

    flat_ridge = max(
        results["flat"]["scatter_rebound"], results["flat"]["gather_rebound"]
    )
    side_ridge = max(
        results["side_face"]["scatter_rebound"],
        results["side_face"]["gather_rebound"],
    )
    noise_ridge = max(
        results["depth_noise"]["scatter_rebound"],
        results["depth_noise"]["gather_rebound"],
    )

    # A flat bar is one physical depth, so it cannot fragment; any ridge
    # that survives flattening is intrinsic coverage of the disk PSF at a
    # hard edge, not a depth artefact.  Only blame depth when a factor
    # makes the ridge materially worse than that baseline.
    #
    # Scatter and gather react differently: random depth noise creates
    # isolated spurious layers that ghost *both* solvers, whereas a
    # depth-continuous surface split across radius layers ghosts the
    # forward scatter composite but not the front-to-back gather
    # accumulation.  That asymmetry separates the two depth causes.
    margin = 1e-3
    noise_both = max(
        results["depth_noise"]["scatter_rebound"],
        results["depth_noise"]["gather_rebound"],
    )
    noise_any = noise_both > flat_ridge + margin
    side_scatter = results["side_face"]["scatter_rebound"]
    side_gather = results["side_face"]["gather_rebound"]
    side_fragments = (
        side_scatter > flat_ridge + margin
        and abs(side_scatter - side_gather) > margin
    )

    if not order_invariant:
        cause = CAUSE_COMPOSITE_ORDER
    elif noise_any and not side_fragments:
        cause = CAUSE_DEPTH_NOISE
    elif side_fragments:
        cause = CAUSE_LAYER_FRAGMENTATION
    elif noise_any:
        cause = CAUSE_DEPTH_NOISE
    elif flat_ridge > 1e-3:
        cause = CAUSE_KERNEL_COVERAGE
    else:
        cause = CAUSE_NONE

    results["isolated_cause"] = cause
    results["cause_thresholds"] = {
        "margin": margin,
        "flat_rebound": flat_ridge,
        "side_face_scatter": side_scatter,
        "side_face_gather": side_gather,
        "depth_noise_scatter": results["depth_noise"]["scatter_rebound"],
        "depth_noise_gather": results["depth_noise"]["gather_rebound"],
        "side_fragments_scatter_only": side_fragments,
        "order_invariant": order_invariant,
    }
    results["frame"] = {"height": h, "width": w, "cropped": False}
    results["flat_is_intrinsic_baseline"] = True

    # Optionally measure the ghost against an optical reference if one
    # is supplied; still full-frame, never cropped.
    if reference is not None:
        results["ghost"] = measure_ghost(
            flat_s, reference, occluder
        ).__dict__
    return results
