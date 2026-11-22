import numpy as np
import pytest

from dof_synth.camera import CameraParams
from dof_synth.geometry import edge_contamination_mask, nearest_valid_depth
from dof_synth.kernels import convolve_disk, disk_kernel
from dof_synth.gather import render_gather
from dof_synth.scatter import render_scatter


def make_camera(focus=2.0, f_number=2.0, w=80, h=60):
    return CameraParams(
        f_mm=35.0,
        f_number=f_number,
        focus_distance_m=focus,
        sensor_width_mm=36.0,
        image_width_px=w,
        image_height_px=h,
    )


def test_coc_sign_convention_and_zero_at_focus():
    cam = make_camera(focus=2.0)
    assert cam.signed_coc_mm(2.0) == pytest.approx(0.0, abs=1e-12)
    assert cam.signed_coc_mm(1.0) > 0.0   # foreground
    assert cam.signed_coc_mm(4.0) < 0.0   # background


def test_coc_matches_thin_lens_derivation():
    cam = make_camera(focus=2.5, f_number=2.0)
    f, s = 0.035, 2.5
    d = f / 2.0
    z = 1.2
    expected = d * f * (s - z) / (z * (s - f))
    assert cam.signed_coc_mm(z) == pytest.approx(expected * 1000.0, rel=1e-9)


def test_wider_aperture_gives_larger_coc():
    open_cam = make_camera(f_number=1.0)
    stopped = make_camera(f_number=8.0)
    z = 1.0
    assert abs(open_cam.signed_coc_px(z)) > abs(stopped.signed_coc_px(z))


def test_dof_bounds_bracket_focus_and_hyperfocal_infinity():
    cam = make_camera(focus=2.0)
    near, far = cam.dof_bounds_m(0.03)
    assert near < 2.0 < far
    h = cam.hyperfocal_m(0.03)
    # Focusing beyond the hyperfocal distance puts the far limit at infinity.
    cam_h = cam.with_focus(h * 1.5)
    _, far_inf = cam_h.dof_bounds_m(0.03)
    assert np.isinf(far_inf)


def test_disk_kernel_is_normalised_and_symmetric():
    k = disk_kernel(5.0)
    assert k.sum() == pytest.approx(1.0)
    assert np.allclose(k, k.T)


def test_convolution_preserves_total_energy():
    rng = np.random.default_rng(0)
    img = rng.random((48, 64, 3))
    blurred = convolve_disk(img, 6.0)
    # Same-kernel convolution on a finite window preserves total flux
    # up to the energy lost across the image border.
    border = 8
    inner = np.s_[border:-border, border:-border]
    assert blurred[inner].sum() == pytest.approx(
        img[inner].sum(), rel=2e-2
    )


def test_edge_mask_detects_silhouette_without_border_artifacts():
    depth = np.full((20, 20), 3.0)
    depth[:, :8] = 0.8
    mask = edge_contamination_mask(depth, depth_step_m=0.15)
    assert not mask[0, :].all() and not mask[-1, :].all()
    # Jump at column 7/8 must be marked.
    assert mask[:, 7:10].any()
    assert not mask[:, 2].any()


def test_nearest_valid_depth_fills_foreground_conservatively():
    depth = np.full((10, 10), 4.0)
    invalid = np.zeros((10, 10), dtype=bool)
    invalid[4:6, 4:6] = True
    depth[3, 4:6] = 1.0  # nearby foreground evidence
    filled = nearest_valid_depth(depth, invalid)
    assert filled[4:6, 4:6].max() <= 4.0
    assert np.isfinite(filled).all()


def _two_layer_inputs(w=80, h=60):
    color = np.zeros((h, w, 3))
    depth = np.full((h, w), 4.0)
    # Opaque thin foreground bar, 6 px wide, bright red.
    depth[:, 30:36] = 0.8
    color[:, 30:36] = np.array([0.9, 0.1, 0.05])
    # A bright point light in the background.
    color[30, 60] = np.array([4.0, 4.0, 4.0])
    return color, depth


def test_foreground_bokeh_covers_background_pixels():
    cam = make_camera(focus=6.0, f_number=0.5, w=80, h=60)
    color, depth = _two_layer_inputs()
    res = render_scatter(color, depth, cam, n_layers=24,
                         depth_step_m=0.15)
    radius = float(res.masks.radius_px[30, 33])
    assert radius > 2.5
    # A pixel just outside the bar's pinhole footprint must receive
    # foreground red energy -- a per-pixel-radius blur leaves it dark.
    px = res.image[30, 37]
    assert px[0] > px[2]


def test_scatter_and_gather_agree_on_layered_scene():
    cam = make_camera(focus=2.0, f_number=1.4, w=80, h=60)
    color, depth = _two_layer_inputs()
    s = render_scatter(color, depth, cam, n_layers=40,
                       depth_step_m=0.15)
    g = render_gather(color, depth, cam, n_layers=40,
                      depth_step_m=0.15)
    assert s.image.shape == color.shape
    np.testing.assert_allclose(s.image, g.image, atol=2e-2)


def test_bright_point_conserves_energy_under_defocus():
    from dof_synth.metrics import energy_ratio
    cam = make_camera(focus=0.9, f_number=1.0, w=80, h=60)
    color, depth = _two_layer_inputs()
    res = render_scatter(color, depth, cam, n_layers=24,
                         depth_step_m=0.15)
    point_mask = np.zeros((60, 80), dtype=bool)
    point_mask[30, 60] = True
    ratio = energy_ratio(color, res.image, point_mask, 12)
    assert ratio == pytest.approx(1.0, abs=0.02)


def test_unknown_background_is_declared_when_foreground_occludes():
    cam = make_camera(focus=4.0, f_number=0.5, w=80, h=60)
    color, depth = _two_layer_inputs()
    res = render_gather(color, depth, cam, n_layers=24,
                        depth_step_m=0.15)
    # The foreground footprint beyond the bar's own columns is unknown.
    assert res.masks.unknown_background.any()
    # In this scene background samples are always within reach, so the
    # unrecoverable subset must stay empty.
    assert not res.masks.unrecoverable.any()


def test_unrecoverable_region_is_flagged_for_enclosed_occlusion():
    # Foreground ring whose bokeh footprint covers background pixels.
    # A 1 px declaration radius means any fill farther than one pixel from
    # measured background is honestly flagged as unrecoverable.
    h, w = 80, 80
    color = np.full((h, w, 3), 0.2)
    depth = np.full((h, w), 5.0)
    yy, xx = np.mgrid[0:h, 0:w]
    ring = ((yy - 40) ** 2 + (xx - 40) ** 2 <= 26**2) & (
        (yy - 40) ** 2 + (xx - 40) ** 2 >= 20**2
    )
    depth[ring] = 0.8
    color[ring] = 0.9
    cam = make_camera(focus=5.0, f_number=0.5, w=w, h=h)
    res = render_scatter(
        color, depth, cam, n_layers=24,
        declaration_radius_px=1.0, depth_step_m=0.15,
    )
    assert res.masks.unknown_background.any()
    assert res.masks.unrecoverable.any()
    assert res.masks.unrecoverable.sum() <= res.masks.unknown_background.sum()


def test_thin_foreground_bokeh_energy_beats_per_pixel_baseline():
    # A 3 px foreground bar, heavily defocused.  A per-pixel-radius blur
    # only reweights the few foreground pixels and under-deposits energy
    # into the surrounding bokeh footprint; the occlusion-aware solver
    # must spread the full disk energy beyond the bar's own columns.
    from dof_synth.metrics import naive_radius_blur
    h, w = 80, 100
    color = np.zeros((h, w, 3))
    depth = np.full((h, w), 5.0)
    depth[:, 48:51] = 0.8
    color[:, 48:51] = np.array([1.0, 0.0, 0.0])
    cam = make_camera(focus=5.0, f_number=0.5, w=w, h=h)
    res = render_scatter(color, depth, cam, n_layers=32,
                         depth_step_m=0.15)
    naive = np.clip(
        naive_radius_blur(color, res.masks.radius_px, n_bins=32), 0, None
    )
    outside = np.ones((h, w), dtype=bool)
    outside[:, 48:51] = False
    proper = res.image[outside, 0].sum()
    baseline = naive[outside, 0].sum()
    assert proper > baseline * 1.15


def test_occlusion_aware_solver_beats_naive_on_thin_foreground():
    # Thin foreground over a textured background.  The naive per-pixel
    # radius blur cannot reason about occlusion and must score worse on
    # the foreground bokeh footprint than the layered solver.
    from dof_synth.metrics import naive_radius_blur, rmse
    from dof_synth.reference import render_thin_lens
    from dof_synth.scene import build_validation_scene
    from dof_synth.reference import render_pinhole
    from dof_synth.scene import analytic_masks
    from scipy import ndimage

    w = h = 128
    cam = CameraParams(35.0, 1.4, 2.75, 36.0, w, h)
    scene = build_validation_scene()
    color, depth = render_pinhole(scene, cam)
    meta = analytic_masks(scene, cam, depth)
    res = render_scatter(
        color, depth, cam, n_layers=48,
        silhouette_m=meta.thin_slab_mask, depth_step_m=0.25,
    )
    naive = np.clip(naive_radius_blur(color, res.masks.radius_px), 0, None)
    reference = np.clip(
        render_thin_lens(scene, cam, lens_samples=48, seed=3), 0, None
    )
    footprint = ndimage.binary_dilation(meta.thin_slab_mask, iterations=5)
    assert rmse(res.image, reference, footprint) < rmse(
        naive, reference, footprint
    )


def test_raw_inputs_remain_unmodified_for_recomputation():
    cam = make_camera(focus=2.0)
    color, depth = _two_layer_inputs(80, 60)
    color0, depth0 = color.copy(), depth.copy()
    render_scatter(color, depth, cam, depth_step_m=0.15)
    np.testing.assert_array_equal(color, color0)
    np.testing.assert_array_equal(depth, depth0)


def test_invalid_dimensions_raise():
    cam = make_camera(w=32, h=32)
    with pytest.raises(ValueError):
        render_scatter(np.zeros((30, 32, 3)), np.full((30, 32), 2.0),
                       cam, depth_step_m=0.15)
