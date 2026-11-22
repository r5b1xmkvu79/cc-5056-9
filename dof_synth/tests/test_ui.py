"""Headless checks for the interactive viewer (Agg backend, no display)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from dof_synth.camera import CameraParams
from dof_synth.ui import DoFViewer


def _scene():
    h, w = 60, 80
    color = np.full((h, w, 3), 0.2)
    depth = np.full((h, w), 4.0)
    depth[:, 30:36] = 0.8
    color[:, 30:36] = np.array([0.9, 0.1, 0.05])
    cam = CameraParams(35.0, 1.0, 2.0, 36.0, w, h)
    return color, depth, cam


def test_viewer_builds_and_recomputes_on_focus_drag():
    color, depth, cam = _scene()
    color0, depth0 = color.copy(), depth.copy()
    viewer = DoFViewer(color, depth, cam)
    first = viewer.result.image.copy()
    # Simulate dragging the focus slider to a new plane.
    viewer.focus_slider.set_val(1.0)
    second = viewer.result.image
    assert viewer.result.camera.focus_distance_m == 1.0
    assert not np.allclose(first, second)
    # Raw measurements stay available and untouched for recomputation.
    np.testing.assert_array_equal(viewer.color, color0)
    np.testing.assert_array_equal(viewer.depth_m, depth0)


def test_viewer_mask_views_and_method_switch():
    color, depth, cam = _scene()
    viewer = DoFViewer(color, depth, cam)
    for label, _ in [
        ("signed CoC (px)", None),
        ("foreground", None),
        ("edge contaminated", None),
        ("unknown background", None),
        ("unrecoverable", None),
        ("in focus", None),
    ]:
        viewer._on_view(label)
    viewer._on_method("gather (reverse)")
    assert viewer.method == "gather"
    assert viewer.result.method == "reverse_gather"
    viewer._on_method("scatter (near→far)")
    assert viewer.result.method == "forward_scatter"


def test_viewer_diagnose_links_canvas_to_kernel_evidence():
    # A receding side face on the bar splits one depth-continuous surface
    # across radius layers -- the duplicate-contour root cause.  The
    # viewer must run the real trace and report it, not a placeholder.
    h, w = 120, 160
    color = np.full((h, w, 3), [0.18, 0.22, 0.30])
    depth = np.full((h, w), 5.95)
    c0, c1 = w // 2 - 12, w // 2 + 12
    depth[:, c0:c1] = 0.70
    color[:, c0:c1] = [0.85, 0.30, 0.25]
    depth[:, c1 - 2] = 0.707
    depth[:, c1 - 1] = 0.734
    cam = CameraParams(35.0, 1.4, 2.75, 36.0, w, h)
    viewer = DoFViewer(color, depth, cam)

    assert viewer.diagnostic is None
    viewer._on_diagnose()
    diag = viewer.diagnostic
    assert diag is not None
    assert diag.cause == "layer_fragmentation"
    assert diag.stage == "layer_assign"
    # Evidence actually came from the kernel, on the full frame.
    assert diag.boundary_summary["included_full_frame"] is True
    assert diag.boundary_summary["defaults_used"] is False
    assert diag.per_pixel_stage
    # The diagnosis view must render without a recursion error and the
    # overlay must be present.
    viewer._on_view("occlusion diagnosis")
    assert viewer.diagnostic is not None

    # Moving the focus plane invalidates the cached diagnosis, and the
    # next Diagnose recomputes it from the new raw solve.
    focus_before = viewer.result.camera.focus_distance_m
    viewer.focus_slider.set_val(1.2)
    assert viewer.diagnostic is None
    assert viewer.result.camera.focus_distance_m == 1.2
    viewer._on_diagnose()
    assert viewer.diagnostic is not None
    assert viewer.result.camera.focus_distance_m == 1.2 != focus_before


def test_viewer_diagnose_surfaces_failure_not_a_default():
    # A constant-depth frame has no occlusion edge; the viewer must say
    # so rather than inventing a diagnosis.
    h, w = 40, 40
    color = np.full((h, w, 3), 0.2)
    depth = np.full((h, w), 3.0)
    cam = CameraParams(35.0, 1.4, 2.75, 36.0, w, h)
    viewer = DoFViewer(color, depth, cam)
    viewer._on_diagnose()
    assert viewer.diagnostic is None
    assert "No occlusion edge" in viewer.diagnosis_note
