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
