"""Tests for the occlusion-ghost diagnostics.

These assert the *evidence*, not just that code runs: every claim the
diagnostic makes must be reproducible from the real solver output.  The
tests deliberately keep the full frame (no boundary cropping) so a pass
cannot come from trimming the offending edge.
"""

import numpy as np
import pytest
from scipy import ndimage

from dof_synth.camera import CameraParams
from dof_synth.diagnostics import (
    CAUSE_KERNEL_COVERAGE,
    CAUSE_LAYER_FRAGMENTATION,
    CAUSE_NONE,
    depth_continuous_components,
    diagnose_occlusion,
    known_depth_step_probe,
    layer_fragmentation,
    measure_ghost,
    track_dof_boundaries,
)
from dof_synth.gather import render_gather
from dof_synth.kernels import quantize_radii
from dof_synth.reference import render_pinhole
from dof_synth.scatter import render_scatter
from dof_synth.trace import TraceRequest


def _camera(focus=2.75, f_number=1.4, size=160, height=None):
    h = size if height is None else height
    return CameraParams(35.0, f_number, focus, 36.0, size, h)


def _bar_scene(h=120, w=160, bar_depth=0.70, bg_depth=5.95, side=False):
    color = np.zeros((h, w, 3))
    depth = np.full((h, w), bg_depth)
    color[:, :] = [0.18, 0.22, 0.30]
    c0, c1 = w // 2 - 12, w // 2 + 12
    depth[:, c0:c1] = bar_depth
    color[:, c0:c1] = [0.85, 0.30, 0.25]
    if side:
        depth[:, c1 - 2] = bar_depth + 0.007
        depth[:, c1 - 1] = bar_depth + 0.034
    return color, depth, (c0, c1)


# ---------------------------------------------------------------------------
# Depth-continuous components and fragmentation
# ---------------------------------------------------------------------------

def test_components_join_continuous_surface_and_split_at_step():
    depth = np.full((20, 20), 5.0)
    depth[:, 10:] = 0.7
    labels = depth_continuous_components(depth, depth_step_m=0.25)
    # Two surfaces -> exactly two components.
    assert len(np.unique(labels)) == 2
    assert labels[0, 0] == labels[19, 0]      # far wall is one surface
    assert labels[0, 15] == labels[19, 15]    # bar is one surface
    assert labels[0, 0] != labels[0, 15]


def test_fragmentation_detects_one_surface_split_across_layers():
    # A bar whose depth drifts just enough to cross a radius bin.
    depth = np.full((80, 80), 0.70)
    depth[:, -3:] = 0.745            # still within the bar, no jump > step
    cam = _camera(size=80)
    radius = np.abs(cam.signed_coc_px(depth)) * 0.5
    layer_idx, _ = quantize_radii(radius, 64, float(radius.max()))
    fragmented, records = layer_fragmentation(depth, layer_idx, 0.25)
    assert fragmented.any()
    assert records and records[0]["n_layers"] >= 2
    # The bar must be reported as a single component spanning those layers.
    assert records[0]["depth_span_m"] < 0.25


# ---------------------------------------------------------------------------
# Boundary tracking refuses defaults and records full frame
# ---------------------------------------------------------------------------

def test_boundary_tracking_reports_layers_and_full_frame():
    color, depth, (c0, c1) = _bar_scene()
    cam = _camera(size=depth.shape[1], height=depth.shape[0])
    report = track_dof_boundaries(depth, cam, coc_limit_px=1.0,
                                  depth_step_m=0.25)
    assert report.summary["included_full_frame"] is True
    assert report.summary["frame_pixels"] == depth.size
    assert report.summary["defaults_used"] is False
    assert report.summary["boundary_pixels"] > 0
    # The bar/background edge must be among the tracked boundaries.
    assert any(abs(b.near_depth_m - 0.70) < 1e-6
               or abs(b.far_depth_m - 0.70) < 1e-6
               for b in report.boundaries)


def test_boundary_tracking_rejects_invalid_depth_instead_of_defaulting():
    cam = _camera(size=32)
    bad = np.full((32, 32), 2.0)
    bad[4, 4] = np.nan
    with pytest.raises(ValueError):
        track_dof_boundaries(bad, cam)
    bad2 = np.full((32, 32), -1.0)
    with pytest.raises(ValueError):
        track_dof_boundaries(bad2, cam)


# ---------------------------------------------------------------------------
# In-kernel trace must match the actual solver output
# ---------------------------------------------------------------------------

def test_trace_events_reconstruct_the_actual_output_pixel():
    color, depth, (c0, c1) = _bar_scene()
    cam = _camera(size=depth.shape[1], height=depth.shape[0])
    row = depth.shape[0] // 2
    cols = [c1 - 1, c1, c1 + 1]
    req = TraceRequest(
        pixels=tuple((row, c) for c in cols),
        max_events_per_pixel=256,
        max_total_events=4096,
    )
    res = render_scatter(color, depth, cam, n_layers=64,
                         depth_step_m=0.25, trace=req)
    assert res.trace is not None
    for c in cols:
        pt = res.trace.pixels[(row, c)]
        # Sum of events must reproduce the pixel the solver produced.
        recon = np.zeros(3)
        for ev in pt.events:
            recon += np.array(ev.energy_rgb)
        assert np.allclose(recon, res.image[row, c], atol=1e-6), (
            f"trace at ({row},{c}) does not reconstruct the output"
        )
        assert pt.output_rgb is not None


def test_trace_is_bounded_and_reports_drops():
    color, depth, (c0, c1) = _bar_scene()
    cam = _camera(size=depth.shape[1], height=depth.shape[0])
    row = depth.shape[0] // 2
    pixels = tuple((row, c) for c in range(c0 - 3, c1 + 3))
    # Tiny budget forces truncation; the recorder must say so.
    req = TraceRequest(
        pixels=pixels, max_events_per_pixel=2, max_total_events=6
    )
    res = render_scatter(color, depth, cam, n_layers=64,
                         depth_step_m=0.25, trace=req)
    tr = res.trace
    assert tr.truncated is True
    assert tr.dropped_events > 0
    assert tr.event_count() <= req.max_total_events
    for pt in tr.pixels.values():
        assert len(pt.events) <= req.max_events_per_pixel


def test_trace_rejects_out_of_bounds_pixels():
    color, depth, _ = _bar_scene()
    cam = _camera(size=depth.shape[1], height=depth.shape[0])
    req = TraceRequest(pixels=((0, 0), (10_000, 10_000)))
    with pytest.raises(ValueError):
        render_scatter(color, depth, cam, n_layers=16,
                       depth_step_m=0.25, trace=req)


# ---------------------------------------------------------------------------
# Ghost measurement is full-frame and detects a known duplicate contour
# ---------------------------------------------------------------------------

def test_ghost_measurement_full_frame_reports_band_separately():
    h, w = 80, 80
    occluder = np.zeros((h, w), dtype=bool)
    occluder[:, 34:46] = True
    reference = np.zeros((h, w, 3))
    reference[:, :] = 0.1
    image = reference.copy()
    # A bright duplicate contour just outside the right edge.
    image[:, 46:48] += 0.05
    m = measure_ghost(image, reference, occluder)
    assert m.cropped is False
    assert m.frame_pixels == h * w
    assert m.ghost_width_px >= 1.0
    assert m.ghost_excess_max > 0.0
    assert m.band_pixels < m.frame_pixels


# ---------------------------------------------------------------------------
# The known-depth-step experiment isolates the causes correctly
# ---------------------------------------------------------------------------

def test_known_step_isolates_layer_fragmentation_and_order_invariance():
    cam = _camera()
    probe = known_depth_step_probe(
        cam, n_layers=48, depth_step_m=0.25, h=120, w=120
    )
    # Equal-depth two-layer order swap must be invariant: this rules out
    # normalization / compositing-order faults.
    assert probe["order_swap"]["invariant"] is True
    # A flat bar has a clean (monotone) edge -- the intrinsic baseline.
    assert probe["flat"]["scatter_rebound"] < 1e-3
    # A receding side face (one physical surface crossing bins) ghosts
    # the forward scatter more than the gather.
    assert probe["side_face"]["scatter_rebound"] > 1e-3
    assert (
        probe["side_face"]["scatter_rebound"]
        > probe["side_face"]["gather_rebound"] + 1e-3
    )
    assert probe["isolated_cause"] == CAUSE_LAYER_FRAGMENTATION
    assert probe["frame"]["cropped"] is False


def test_known_step_flat_scene_has_no_fragmentation():
    # A truly constant-depth bar never fragments, so the probe must not
    # blame depth for it.
    cam = _camera()
    probe = known_depth_step_probe(
        cam, n_layers=48, depth_step_m=0.25, h=100, w=100
    )
    assert probe["flat"]["scatter_rebound"] < 1e-3
    assert probe["isolated_cause"] != CAUSE_LAYER_FRAGMENTATION


# ---------------------------------------------------------------------------
# End-to-end diagnosis ties the ghost to a fragmented foreground
# ---------------------------------------------------------------------------

def test_diagnosis_attributes_ghost_to_fragmented_foreground_layer():
    h, w = 120, 160
    color, depth, (c0, c1) = _bar_scene(h, w, side=True)
    cam = _camera(size=w, height=h)
    row = h // 2
    # Trace background-side pixels just outside the bar.
    pixels = tuple((row, c) for c in range(c1 - 2, c1 + 4))
    req = TraceRequest(pixels=pixels, max_events_per_pixel=256,
                       max_total_events=4096)
    res = render_scatter(color, depth, cam, n_layers=64,
                         silhouette_m=None, depth_step_m=0.25, trace=req)
    diag = diagnose_occlusion(
        res, depth, cam,
        occluder_mask=(depth < 1.0),
        trace=res.trace,
        depth_step_m=0.25,
    )
    d = diag.to_dict()
    assert d["cause"] == CAUSE_LAYER_FRAGMENTATION
    assert d["stage"] == "layer_assign"
    # The proximate coverage hole at the composite stage must be visible.
    assert d["stage_flags"]["composite"] > 0
    # At least one attributed pixel must be inside the fragmentation reach.
    assert any(
        p["fragmented"] or p["fragmentation_reach"]
        for p in d["per_pixel_stage"]
    )
    # Full frame, nothing cropped, no defaults substituted.
    assert d["boundary_summary"]["included_full_frame"] is True
    assert d["boundary_summary"]["defaults_used"] is False
    assert d["trace_truncated"] is False


def test_diagnosis_requires_trace_and_refuses_to_invent_one():
    color, depth, _ = _bar_scene()
    cam = _camera(size=depth.shape[1], height=depth.shape[0])
    res = render_scatter(color, depth, cam, n_layers=32,
                         depth_step_m=0.25)
    assert res.trace is None
    with pytest.raises(ValueError):
        diagnose_occlusion(res, depth, cam)


def test_diagnosis_does_not_report_improvement_by_cropping_boundary():
    # Ghost metrics must be measured on every frame pixel; the band is a
    # subset, never a replacement.
    h, w = 100, 120
    color, depth, (c0, c1) = _bar_scene(h, w, side=True)
    cam = _camera(size=w, height=h)
    row = h // 2
    req = TraceRequest(pixels=((row, c1),), max_total_events=256)
    res = render_scatter(color, depth, cam, n_layers=64,
                         depth_step_m=0.25, trace=req)
    ref = res.image + 0.0
    diag = diagnose_occlusion(
        res, depth, cam, occluder_mask=(depth < 1.0),
        reference=ref, trace=res.trace, depth_step_m=0.25,
    )
    g = diag.to_dict()["ghost"]
    assert g["cropped"] is False
    assert g["frame_pixels"] == h * w
    assert g["band_pixels"] <= g["frame_pixels"]
